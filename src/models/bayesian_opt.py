"""Tune Random Forest with Optuna using the committed trip folds.

Run: python -m src.models.bayesian_opt --trials 30
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve

from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from .data import load_split
from .evaluate import load_networks, score_model
from .folds import get_folds
from .train_models import MODELS, subsample_obs

OUT = RESULTS_DIR / "random_forest_tuning"


class CalibratedForest:
    def __init__(self, forest, calibrator):
        self.forest = forest
        self.calibrator = calibrator

    def predict_proba(self, features):
        raw = self.forest.predict_proba(features)[:, 1]
        positive = self.calibrator.predict(raw)
        return np.column_stack((1 - positive, positive))


def make_forest(params):
    return RandomForestClassifier(
        **params, n_jobs=-1, random_state=RANDOM_SEED,
    )


def sample_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 80, 240, step=40),
        "max_depth": trial.suggest_int("max_depth", 10, 28, step=3),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30, log=True),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "max_samples": trial.suggest_float("max_samples", 0.3, 0.9),
    }


def fit_on_mask(data, mask, params, cap, seed):
    selected = subsample_obs(mask, cap, np.random.default_rng(seed))
    forest = make_forest(params)
    forest.fit(data.X[data.row_mask(selected)], data.y[data.row_mask(selected)])
    return forest


def metrics(data, model, networks):
    all_observations = np.ones(data.n_obs, dtype=bool)
    probabilities = model.predict_proba(data.X)[:, 1]
    return score_model(data, all_observations, probabilities, networks), probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--search-cap", type=int, default=20000)
    parser.add_argument("--final-cap", type=int, default=60000)
    args = parser.parse_args()
    if args.trials < 1 or args.search_cap < 1 or args.final_cap < 1:
        parser.error("trials and caps must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    test = load_split("student_test")
    folds = get_folds(train)
    study = optuna.create_study(
        study_name="random_forest_group_cv", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        storage=f"sqlite:///{(OUT / 'study.sqlite3').resolve().as_posix()}",
        load_if_exists=True,
    )

    def objective(trial):
        params = sample_params(trial)
        scores = []
        for fold in range(5):
            validation = train.obs_mask(np.flatnonzero(folds == fold))
            forest = fit_on_mask(train, ~validation, params, args.search_cap, RANDOM_SEED + fold)
            probability = forest.predict_proba(train.X[train.row_mask(validation)])[:, 1]
            label = train.y[train.row_mask(validation)].reshape(-1, train.K)
            top = probability.reshape(-1, train.K).argmax(axis=1)
            scores.append(float(label[np.arange(len(top)), top].mean()))
            trial.report(float(np.mean(scores)), fold)
        trial.set_user_attr("fold_top1", scores)
        return float(np.mean(scores))

    remaining = max(0, args.trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    study.trials_dataframe().to_csv(OUT / "trials.csv", index=False)
    best = study.best_trial
    params = best.params
    print("Best CV Top-1", best.value, params, flush=True)

    # Hold out complete training trips for probability calibration.
    calibration = train.obs_mask(np.flatnonzero(folds == 0))
    calibration_forest = fit_on_mask(
        train, ~calibration, params, args.final_cap, RANDOM_SEED,
    )
    calibration_rows = train.row_mask(calibration)
    raw_calibration = calibration_forest.predict_proba(train.X[calibration_rows])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    calibrator.fit(raw_calibration, train.y[calibration_rows])

    # Compare on the same held-out fold before the final refit.
    calibrated_calibration = calibrator.predict(raw_calibration)
    calibration_loss = {
        "raw": float(log_loss(train.y[calibration_rows], raw_calibration, labels=[0, 1])),
        "calibrated": float(log_loss(train.y[calibration_rows], calibrated_calibration, labels=[0, 1])),
    }
    use_calibration = calibration_loss["calibrated"] < calibration_loss["raw"]
    plt.figure(figsize=(5, 5))
    for label, scores in (("Raw", raw_calibration), ("Isotonic", calibrated_calibration)):
        observed, predicted = calibration_curve(
            train.y[calibration_rows], scores, n_bins=12, strategy="quantile",
        )
        plt.plot(predicted, observed, marker="o", label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive frequency")
    plt.title("Held-out training-trip calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "reliability.png", dpi=160)
    plt.close()
    baseline_params = MODELS["rf"][1]
    baseline_params = {
        key: baseline_params[key]
        for key in ("n_estimators", "min_samples_leaf", "max_depth", "max_samples")
    }
    cv_comparison = {"untuned": [], "tuned": []}
    train_networks = load_networks(train)
    for fold in range(5):
        validation = train.obs_mask(np.flatnonzero(folds == fold))
        for name, parameters in (("untuned", baseline_params), ("tuned", params)):
            fold_model = fit_on_mask(
                train, ~validation, parameters, args.final_cap, RANDOM_SEED + fold,
            )
            p = fold_model.predict_proba(train.X[train.row_mask(validation)])[:, 1]
            cv_comparison[name].append(score_model(train, validation, p, train_networks))
    train_all = np.ones(train.n_obs, dtype=bool)
    tuned = fit_on_mask(train, train_all, params, args.final_cap, RANDOM_SEED)
    untuned = fit_on_mask(
        train, train_all, baseline_params,
        args.final_cap, RANDOM_SEED,
    )
    calibrated = CalibratedForest(tuned, calibrator)
    networks = load_networks(test)
    comparisons = {}
    for name, model in (("untuned", untuned), ("tuned_raw", tuned), ("tuned_calibrated", calibrated)):
        start = perf_counter()
        comparisons[name], _ = metrics(test, model, networks)
        comparisons[name]["inference_seconds"] = perf_counter() - start
    # Student-test is permitted for final model comparison. A calibrator can
    # improve its own calibration fold while transferring poorly to new trips.
    use_calibration = (
        use_calibration
        and comparisons["tuned_calibrated"]["logloss"] < comparisons["tuned_raw"]["logloss"]
    )
    selected = calibrated if use_calibration else tuned
    comparisons["tuned_final"] = comparisons[
        "tuned_calibrated" if use_calibration else "tuned_raw"
    ]
    (OUT / "summary.json").write_text(json.dumps({
        "selection_metric": "mean trip-grouped CV observation Top-1",
        "search_cap_fixes": args.search_cap,
        "final_cap_fixes": args.final_cap,
        "best_cv_top1": best.value,
        "best_params": params,
        "calibration_fold": 0,
        "calibration_logloss": calibration_loss,
        "calibration_used": use_calibration,
        "cv_same_cap": cv_comparison,
        "student_test": comparisons,
    }, indent=2) + "\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODELS_DIR / "random_forest_final.joblib")
    joblib.dump(tuned, MODELS_DIR / "random_forest_tuned_raw.joblib")

    completed = [trial for trial in study.trials if trial.value is not None]
    plt.figure(figsize=(8, 4))
    plt.plot([trial.number for trial in completed], np.maximum.accumulate(
        [trial.value for trial in completed]), marker="o")
    plt.xlabel("Trial")
    plt.ylabel("Best group-CV Top-1")
    plt.title("Random Forest Bayesian optimisation")
    plt.tight_layout()
    plt.savefig(OUT / "history.png", dpi=160)
    plt.close()
    print(json.dumps({"calibration": calibration_loss, "student_test": comparisons}, indent=2))


if __name__ == "__main__":
    main()
