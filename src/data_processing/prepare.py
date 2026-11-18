"""Build per-trip model inputs and data quality summaries for development splits."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

from src.config import DATA_DIR, RESULTS_DIR, TOP_K_CANDIDATES
from src.features.feature_extractor import FEATURE_NAMES, generate_candidates

from .labels import align_reference, load_reference
from .loader import load_trip
from .spatial_utils import prepare_geometry


def graph_counts(nodes, arcs):
    parent = list(range(len(nodes)))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for start, end in arcs.tolist():
        parent[root(start)] = root(end)
    components = sum(index == ancestor for index, ancestor in enumerate(parent))
    return components, len(nodes) - np.unique(arcs).size


def prepare_trip(directory: Path, k: int, gap_seconds: float):
    trip = load_trip(directory)
    geometry = prepare_geometry(trip, gap_seconds)
    candidates = generate_candidates(geometry, k)
    route = load_reference(directory, len(trip.arcs))
    aligned, positions, alignment_distance = align_reference(geometry, route)
    labels = candidates.arc_ids == aligned[:, None]
    hit = labels.any(axis=1)
    components, isolated = graph_counts(trip.nodes, trip.arcs)
    dt = np.diff(trip.track[:, 2])
    reference_endpoints = trip.arcs[route]
    payload = {
        "arc_ids": candidates.arc_ids,
        "features": candidates.features,
        "feature_names": np.asarray(FEATURE_NAMES),
        "labels": labels,
        "reference_arc_ids": aligned,
        "reference_positions": positions,
        "alignment_distance_m": alignment_distance.astype(np.float32),
        "reference_route": route,
        "block_starts": geometry.block_starts,
        "timestamps": trip.track[:, 2],
    }
    stats = {
        "trip_id": directory.name,
        "observations": len(aligned),
        "nodes": len(trip.nodes),
        "arcs": len(trip.arcs),
        "reference_arcs": len(route),
        "raw_file_bytes": {
            suffix: (directory / f"{directory.name}.{suffix}").stat().st_size
            for suffix in ("track", "nodes", "arcs", "route")
        },
        "gps_min_lon_lat": trip.track[:, :2].min(axis=0).tolist(),
        "gps_max_lon_lat": trip.track[:, :2].max(axis=0).tolist(),
        "weak_components": components,
        "isolated_nodes": isolated,
        "zero_length_arcs": int(np.count_nonzero(geometry.length == 0)),
        "noncontiguous_reference_hops": int(np.count_nonzero(
            reference_endpoints[:-1, 1] != reference_endpoints[1:, 0],
        )),
        "gps_blocks": len(geometry.block_starts),
        "max_gap_seconds": float(dt.max(initial=0)),
        "one_second_steps": int(np.count_nonzero(dt == 1)),
        "candidate_hits": int(np.count_nonzero(hit)),
        "nearest_arc_hits": int(np.count_nonzero(labels[:, 0])),
        "candidate_miss_rate": float(1 - hit.mean()),
        "alignment_distance_m_p50": float(np.median(alignment_distance)),
        "alignment_distance_m_p95": float(np.percentile(alignment_distance, 95)),
        "alignment_distance_over_50m": int(np.count_nonzero(alignment_distance > 50)),
    }
    return payload, stats


def save_trip(path: Path, payload: dict):
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        np.savez_compressed(temp_path, **payload)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "preprocessing")
    parser.add_argument("--split", choices=["train", "student_test", "both"], default="both")
    parser.add_argument("--trip", action="append", help="Process only the specified trip ID. Repeat as needed.")
    parser.add_argument("--candidates", type=int, default=TOP_K_CANDIDATES)
    parser.add_argument("--gap-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.candidates < 1 or not np.isfinite(args.gap_seconds) or args.gap_seconds <= 0:
        parser.error("use at least one candidate and a positive, finite gap threshold")

    splits = ["train", "student_test"] if args.split == "both" else [args.split]
    directories = {
        split: [path for path in sorted((args.data / split).iterdir())
                if path.is_dir() and (args.trip is None or path.name in args.trip)]
        for split in splits
    }
    if args.trip:
        found = {path.name for paths in directories.values() for path in paths}
        missing = set(args.trip) - found
        if missing:
            parser.error(f"trip IDs not found in the selected split(s): {', '.join(sorted(missing))}")
    for split, paths in directories.items():
        if not paths:
            continue
        output = args.output / split
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        started = perf_counter()
        for directory in paths:
            payload, stats = prepare_trip(directory, args.candidates, args.gap_seconds)
            save_trip(output / f"{directory.name}.npz", payload)
            rows.append(stats)
            print(f"{split}/{directory.name}: {stats['observations']} fixes, "
                  f"{1 - stats['candidate_miss_rate']:.1%} candidate recall", flush=True)
        observations = sum(row["observations"] for row in rows)
        summary = {
            "split": split,
            "trip_count": len(rows),
            "observations": observations,
            "candidate_count": args.candidates,
            "gap_seconds": args.gap_seconds,
            "candidate_recall": sum(row["candidate_hits"] for row in rows) / observations,
            "nearest_arc_top1": sum(row["nearest_arc_hits"] for row in rows) / observations,
            "alignment_distance_over_50m": sum(row["alignment_distance_over_50m"] for row in rows),
            "elapsed_seconds": round(perf_counter() - started, 2),
            "trips": rows,
        }
        report = output / ("summary.json" if args.trip is None else "sample_summary.json")
        report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"{split}: {summary['candidate_recall']:.2%} candidate recall, "
              f"{summary['nearest_arc_top1']:.2%} nearest-arc Top-1 ({summary['elapsed_seconds']}s)",
              flush=True)


if __name__ == "__main__":
    main()
