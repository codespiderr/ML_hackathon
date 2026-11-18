"""Development-only alignment of GPS fixes to a supplied reference route."""

from pathlib import Path

import numpy as np

from .loader import read_array
from .spatial_utils import Geometry, project_to_arcs


def load_reference(directory: Path, arc_count: int) -> np.ndarray:
    route = read_array(directory, "route", 1, int)[:, 0]
    if (route < 0).any() or (route >= arc_count).any():
        raise ValueError(f"{directory}: reference arc outside arc table")
    return route


def align_reference(
    geometry: Geometry, route: np.ndarray, heading_weight_m: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align to nondecreasing reference positions, independently per GPS block.

    Repeated arc IDs are distinct route positions. Each fix can stay on the
    current arc or advance across any number of arcs between observations.
    This geometric alignment gives proxy labels for pointwise experiments.
    """
    start = geometry.start[route]
    vector = geometry.vector[route]
    length = geometry.length[route]
    unit = np.divide(vector, length[:, None], out=np.zeros_like(vector),
                     where=length[:, None] > 0)
    route_indices = np.arange(len(route))
    positions = np.empty(len(geometry.gps), dtype=np.int32)
    block_ends = np.r_[geometry.block_starts[1:], len(geometry.gps)]
    for first, stop in zip(geometry.block_starts, block_ends):
        predecessors = np.empty((stop - first - 1, len(route)), dtype=np.int32)
        previous_cost = np.zeros(len(route))
        for index in range(first, stop):
            distance, _ = project_to_arcs(geometry.gps[index], start, vector, length)
            if geometry.heading_available[index]:
                cosine = np.clip(np.sum(unit * geometry.heading[index], axis=1), -1, 1)
                distance += heading_weight_m * (1 - cosine) * (length > 0)
            if index == first:
                previous_cost = distance
                continue

            # At route position j, only positions 0 through j can precede it.
            best_cost = np.minimum.accumulate(previous_cost)
            # Keep the latest position when two eligible positions have equal cost.
            best_positions = np.where(previous_cost == best_cost, route_indices, 0)
            predecessors[index - first - 1] = np.maximum.accumulate(
                best_positions,
            )
            previous_cost = best_cost + distance

        # Trace back from the lowest-cost final position in this GPS block.
        positions[stop - 1] = np.argmin(previous_cost)
        for index in range(stop - 2, first - 1, -1):
            positions[index] = predecessors[index - first, positions[index + 1]]
    aligned = route[positions]
    distances, _ = project_to_arcs(
        geometry.gps, geometry.start[aligned], geometry.vector[aligned],
        geometry.length[aligned],
    )
    return aligned, positions, distances
