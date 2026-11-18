"""Tune decoder on held-out training trips, then evaluate on student-test trips."""

import json

import joblib
import numpy as np

from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.data_processing.loader import load_trip
from src.models.data import load_split
from src.models.evaluate import load_networks
from src.models.folds import get_folds
from src.models.bayesian_opt import fit_on_mask, CalibratedForest
from .hmm_decoder import decoder


OUT = RESULTS_DIR / "random_forest_tuning"


def transition_f1(predicted, reference):
    predicted = predicted[np.r_[True, predicted[1:] != predicted[:-1]]]
    truth = set(zip(reference[:-1], reference[1:]))
    found = set(zip(predicted[:-1], predicted[1:]))
    return 2 * len(truth & found) / (len(truth) + len(found)) if truth or found else 1.0


def evaluate(data, model, trip_indices, settings):
    networks = load_networks(data)
    totals = {key: 0 for key in ("points", "raw_hits", "decoded_hits", "raw_pairs", "decoded_pairs", "pairs", "worse_trips")}
    trips = []
    for trip_index in trip_indices:
        trip_id = data.trip_ids[trip_index]
        obs = np.flatnonzero(data.trip_of_obs == trip_index)
        candidates = data.arc_ids[obs]
        truth = data.y[data.row_mask(data.obs_mask([trip_index]))].reshape(-1, data.K)
        probability = model.predict_proba(data.X[data.row_mask(data.obs_mask([trip_index]))])[:, 1]
        probability = probability.reshape(-1, data.K)
        raw = candidates[np.arange(len(obs)), probability.argmax(axis=1)]
        trip = load_trip(DATA_DIR / data.split / trip_id)
        reference = np.loadtxt(
            DATA_DIR / data.split / trip_id / f"{trip_id}.route",
            dtype=np.int32, ndmin=1,
        )
        starts = np.flatnonzero(data.block_start[obs])
        decoded = decoder(
            trip, candidates, probability, trip.track[:, 2], starts, **settings,
        )
        raw_hits = int((truth[np.arange(len(obs)), probability.argmax(axis=1)]).sum())
        decoded_hits = int(((candidates == decoded[:, None]) & truth).any(axis=1).sum())
        frm, to = networks[trip_index]
        within = ~data.block_start[obs][1:]
        raw_pairs = int((((raw[:-1] == raw[1:]) | (to[raw[:-1]] == frm[raw[1:]])) & within).sum())
        decoded_pairs = int((((decoded[:-1] == decoded[1:]) | (to[decoded[:-1]] == frm[decoded[1:]])) & within).sum())
        pairs = int(within.sum())
        trips.append({
            "trip_id": trip_id, "observations": len(obs),
            "raw_top1": raw_hits / len(obs), "decoded_top1": decoded_hits / len(obs),
            "raw_continuity": raw_pairs / pairs if pairs else None,
            "decoded_continuity": decoded_pairs / pairs if pairs else None,
            "raw_transition_f1": transition_f1(raw, reference),
            "decoded_transition_f1": transition_f1(decoded, reference),
        })
        for key, value in (
            ("points", len(obs)), ("raw_hits", raw_hits), ("decoded_hits", decoded_hits),
            ("raw_pairs", raw_pairs), ("decoded_pairs", decoded_pairs), ("pairs", pairs),
            ("worse_trips", decoded_hits < raw_hits),
        ):
            totals[key] += value
        print(data.split, trip_id, trips[-1], flush=True)
    return {
        "raw_top1": totals["raw_hits"] / totals["points"],
        "decoded_top1": totals["decoded_hits"] / totals["points"],
        "raw_continuity": totals["raw_pairs"] / totals["pairs"],
        "decoded_continuity": totals["decoded_pairs"] / totals["pairs"],
        "worse_trips": totals["worse_trips"],
        "trips": trips,
    }


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    train = load_split("train")
    folds = get_folds(train)
    heldout = np.flatnonzero(folds == 0)
    mask = ~train.obs_mask(heldout)
    forest = fit_on_mask(train, mask, summary["best_params"], summary["final_cap_fixes"], 42)
    model = forest
    if summary["calibration_used"]:
        final = joblib.load(MODELS_DIR / "random_forest_final.joblib")
        model = CalibratedForest(forest, final.calibrator)

    settings = [
        {"distance_scale": distance, "disconnect_penalty": penalty}
        for distance in (20.0, 50.0) for penalty in (4.0, 8.0)
    ]
    # Use two complete training trips for decoder tuning. The classifier has
    # never fitted on these trips, and student-test remains held out.
    tune_ids = heldout[:2]
    trials = []
    for parameters in settings:
        result = evaluate(train, model, tune_ids, parameters)
        trials.append({"settings": parameters, "top1": result["decoded_top1"],
                       "continuity": result["decoded_continuity"]})
    winner = max(trials, key=lambda row: (row["top1"], row["continuity"]))
    test = load_split("student_test")
    final = joblib.load(MODELS_DIR / "random_forest_final.joblib")
    test_result = evaluate(test, final, range(len(test.trip_ids)), winner["settings"])
    (OUT / "decoder_results.json").write_text(json.dumps({
        "tuning_split": "two held-out training trips",
        "trials": trials, "selected": winner, "student_test": test_result,
    }, indent=2) + "\n")
    (MODELS_DIR / "decoder_config.json").write_text(
        json.dumps(winner["settings"], indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
