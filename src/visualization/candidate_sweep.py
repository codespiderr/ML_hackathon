"""Measure the training-only trade-off between top-K coverage and candidate rows."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import DATA_DIR, RESULTS_DIR
from src.data_processing.loader import load_trip
from src.data_processing.spatial_utils import prepare_geometry
from src.features.candidate_gen import nearest_arcs


CANDIDATE_COUNTS = np.array([1, 3, 5, 8, 12, 16, 24, 32, 48, 64])
BASELINE_K = 12
RECALL_TOLERANCE_PP = 0.5


def measure_trip(directory: Path, archive: Path):
    with np.load(archive, allow_pickle=False) as prepared:
        baseline = prepared["arc_ids"]
        reference = prepared["reference_arc_ids"]
        distances = prepared["alignment_distance_m"]
    if baseline.shape != (len(reference), BASELINE_K):
        raise ValueError(f"{archive}: expected an archive generated with K={BASELINE_K}")

    # A rank beyond the sweep range represents a miss at every tested K.
    max_k = int(CANDIDATE_COUNTS[-1])
    rank = np.full(len(reference), max_k + 1)
    observed, candidate = np.where(baseline == reference[:, None])
    rank[observed] = candidate + 1
    missed = np.flatnonzero(rank > BASELINE_K)
    trip = load_trip(directory)
    tested_counts = np.minimum(CANDIDATE_COUNTS, len(trip.arcs))
    if len(missed):
        geometry = prepare_geometry(trip)
        expanded = nearest_arcs(geometry, int(tested_counts[-1]), points=geometry.gps[missed])
        # Exact nearest-neighbour search must reproduce the baseline prefix.
        if not np.array_equal(expanded[:, :BASELINE_K], baseline[missed]):
            raise ValueError(f"{archive}: candidate order differs from the raw training trip")
        recovered, candidate = np.where(expanded == reference[missed, None])
        rank[missed[recovered]] = candidate + 1

    hits = np.count_nonzero(rank[:, None] <= CANDIDATE_COUNTS, axis=0)
    return hits, tested_counts * len(reference), distances


def plot_tradeoff(summary: dict, distances: np.ndarray, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    rows = summary["counts"]
    counts = [row["k"] for row in rows]
    recalls = [row["recall_pct"] for row in rows]
    selected_k = summary["selected_k"]
    selected = next(row for row in rows if row["k"] == selected_k)
    baseline = next(row for row in rows if row["k"] == BASELINE_K)

    fig, (tradeoff, distance_plot) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    fig.suptitle(
        f"PathSnap candidate sweep | {summary['trip_count']} training trips, "
        f"{len(distances):,} GPS fixes", fontsize=15, fontweight="bold",
    )
    tradeoff.plot(counts, recalls, "o-", color="#146477", linewidth=2)
    tradeoff.scatter(selected_k, selected["recall_pct"], zorder=4, color="#a24518", s=55)
    tradeoff.set(
        xscale="log", xticks=counts, xticklabels=counts,
        xlabel="Candidates per GPS fix (K)", ylabel="Reference arc coverage",
        title="Larger candidate sets improve coverage",
    )
    tradeoff.yaxis.set_major_formatter(PercentFormatter(100))
    tradeoff.text(
        0.97, 0.08,
        f"K={BASELINE_K}: {baseline['recall_pct']:.2f}%\n"
        f"Selected K={selected_k}: {selected['recall_pct']:.2f}%",
        transform=tradeoff.transAxes, ha="right", va="bottom", fontsize=10,
    )

    sorted_distances = np.sort(distances)
    cumulative = np.arange(1, len(distances) + 1) / len(distances) * 100
    distance_plot.plot(sorted_distances, cumulative, color="#146477", linewidth=2)
    distance_plot.set(
        xlim=(0, summary["reference_distance_m"]["p99"]), ylim=(0, 100),
        xlabel="Distance to aligned reference arc (m)", ylabel="Share of GPS fixes",
        title="Distance distribution (through the 99th percentile)",
    )
    distance_plot.yaxis.set_major_formatter(PercentFormatter(100))
    for name in ("p50", "p95"):
        distance_plot.axvline(
            summary["reference_distance_m"][name], color="#a24518", linestyle="--", alpha=0.7,
        )
    for axis in (tradeoff, distance_plot):
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--prepared", type=Path, default=RESULTS_DIR / "preprocessing")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "baseline" / "preprocessing")
    args = parser.parse_args()

    trips = sorted(path for path in (args.data / "train").iterdir() if path.is_dir())
    if not trips:
        parser.error("no trips found in the training split")

    total_hits = np.zeros(len(CANDIDATE_COUNTS), dtype=np.int64)
    total_rows = np.zeros_like(total_hits)
    trip_results = []
    distance_batches = []
    for trip in trips:
        hits, rows, distances = measure_trip(
            trip, args.prepared / "train" / f"{trip.name}.npz",
        )
        total_hits += hits
        total_rows += rows
        distance_batches.append(distances)
        trip_results.append({
            "trip_id": trip.name,
            "observations": len(distances),
            "hits_by_k": hits.tolist(),
        })
        print(f"train/{trip.name}: {len(distances):,} fixes", flush=True)

    distances = np.concatenate(distance_batches)
    recall = total_hits / len(distances) * 100
    selected_index = np.flatnonzero(recall >= recall.max() - RECALL_TOLERANCE_PP)[0]
    percentiles = (50, 90, 95, 99)
    summary = {
        "split": "train",
        "trip_count": len(trips),
        "observations": len(distances),
        "selection_rule": "Smallest K within 0.5 percentage points of best tested recall",
        "selected_k": int(CANDIDATE_COUNTS[selected_index]),
        "reference_distance_m": {
            f"p{percentile}": float(distance)
            for percentile, distance in zip(percentiles, np.percentile(distances, percentiles))
        },
        "counts": [
            {"k": int(k), "hits": int(hits), "recall_pct": float(coverage),
             "miss_rate_pct": float(100 - coverage), "candidate_rows": int(rows)}
            for k, hits, coverage, rows in zip(CANDIDATE_COUNTS, total_hits, recall, total_rows)
        ],
        "trips": trip_results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidate_sweep.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    plot_tradeoff(summary, distances, args.output / "candidate_sweep.png")
    print(f"Selected K={summary['selected_k']} at {recall[selected_index]:.2f}% recall.")


if __name__ == "__main__":
    main()
