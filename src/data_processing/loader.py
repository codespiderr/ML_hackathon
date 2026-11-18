"""Read and validate raw trip files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    nodes: np.ndarray
    arcs: np.ndarray
    track: np.ndarray


def read_array(directory: Path, suffix: str, columns: int, dtype: type) -> np.ndarray:
    path = directory / f"{directory.name}.{suffix}"
    rows = np.loadtxt(path, dtype=dtype, ndmin=2)
    if rows.size == 0 or rows.shape[1] != columns or not np.isfinite(rows).all():
        raise ValueError(f"{path}: expected non-empty, finite rows with {columns} columns")
    return rows


def load_trip(directory: Path) -> Trip:
    """Read only inputs allowed at inference time."""
    nodes = read_array(directory, "nodes", 2, float)
    arcs = read_array(directory, "arcs", 2, int)
    track = read_array(directory, "track", 3, float)
    for coordinates in (nodes, track[:, :2]):
        if (np.abs(coordinates) > [180, 90]).any():
            raise ValueError(f"{directory}: invalid longitude or latitude")
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError(f"{directory}: arc endpoint outside node table")
    if (np.diff(track[:, 2]) <= 0).any():
        raise ValueError(f"{directory}: timestamps must be strictly increasing")
    return Trip(nodes, arcs, track)
