"""Find the closest directed arcs for each GPS observation."""

from collections import defaultdict

import numpy as np

from src.data_processing.spatial_utils import Geometry, project_to_arcs


def _index_arcs(geometry: Geometry, cell_size: float):
    """Index each segment in every cell touched by its bounding box."""
    end = geometry.start + geometry.vector
    lower = np.floor(np.minimum(geometry.start, end) / cell_size).astype(int)
    upper = np.floor(np.maximum(geometry.start, end) / cell_size).astype(int)
    cells = defaultdict(list)
    long_arcs = []
    for arc_id, ((x0, y0), (x1, y1)) in enumerate(zip(lower.tolist(), upper.tolist())):
        # Check unusually long segments separately to limit grid memory.
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 64:
            long_arcs.append(arc_id)
            continue
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                cells[x, y].append(arc_id)
    return cells, long_arcs


def _rank_arcs(geometry: Geometry, points: np.ndarray, arc_ids: np.ndarray, count: int):
    """Return the nearest arc IDs and the distance to the last one."""
    distances, _ = project_to_arcs(
        points[:, None, :],
        geometry.start[arc_ids],
        geometry.vector[arc_ids],
        geometry.length[arc_ids],
    )
    # Arc IDs arrive in ascending order, so stable sorting breaks ties by ID.
    order = np.argsort(distances, axis=1, kind="stable")[:, :count]
    furthest = np.take_along_axis(distances, order[:, -1:], axis=1)[:, 0]
    return arc_ids[order], furthest


def nearest_arcs(geometry: Geometry, count: int, cell_size: float = 100) -> np.ndarray:
    """Look in nearby cells first and fall back to the full graph as needed."""
    cells, long_arcs = _index_arcs(geometry, cell_size)
    observations = defaultdict(list)
    for index, cell in enumerate(np.floor(geometry.gps / cell_size).astype(int)):
        observations[tuple(cell)].append(index)
    nearest = np.empty((len(geometry.gps), count), dtype=np.int32)
    all_arcs = np.arange(len(geometry.start))

    for (x, y), indices in observations.items():
        pending = np.asarray(indices)
        for radius in range(5):
            use_full_graph = radius == 4
            if use_full_graph:
                arc_ids = all_arcs
            else:
                local = set(long_arcs)
                for col in range(x - radius, x + radius + 1):
                    for row in range(y - radius, y + radius + 1):
                        local.update(cells.get((col, row), ()))
                if len(local) < count:
                    continue
                arc_ids = np.asarray(sorted(local))

            lower_edge = np.asarray([x - radius, y - radius]) * cell_size
            upper_edge = np.asarray([x + radius + 1, y + radius + 1]) * cell_size
            remaining = []
            # Bound the size of the temporary point-by-segment distance matrix.
            batch_size = max(1, 200_000 // len(arc_ids))
            for batch_start in range(0, len(pending), batch_size):
                batch = pending[batch_start:batch_start + batch_size]
                points = geometry.gps[batch]
                ranked_ids, furthest = _rank_arcs(geometry, points, arc_ids, count)
                if use_full_graph:
                    nearest[batch] = ranked_ids
                    continue

                # An unsearched segment cannot be closer than the window's edge.
                edge_distance = np.minimum(points - lower_edge, upper_edge - points).min(axis=1)
                done = furthest < edge_distance
                nearest[batch[done]] = ranked_ids[done]
                remaining.extend(batch[~done])
            if not remaining:
                break
            pending = np.asarray(remaining)
    return nearest
