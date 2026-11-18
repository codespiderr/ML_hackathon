"""Trip-local metric projection and point-to-segment geometry."""

from dataclasses import dataclass

import numpy as np

from .loader import Trip


EARTH_RADIUS_M = 6_371_008.8


@dataclass
class Geometry:
    gps: np.ndarray
    start: np.ndarray
    vector: np.ndarray
    length: np.ndarray
    heading: np.ndarray
    heading_available: np.ndarray
    speed: np.ndarray
    block_starts: np.ndarray


def prepare_geometry(trip: Trip, gap_seconds: float = 30.0) -> Geometry:
    """Project the trip to a local metre grid and estimate motion within GPS blocks."""
    if not np.isfinite(gap_seconds) or gap_seconds <= 0:
        raise ValueError("gap_seconds must be positive and finite")
    origin = np.median(trip.track[:, :2], axis=0)
    scale = EARTH_RADIUS_M * np.pi / 180 * np.array([np.cos(np.deg2rad(origin[1])), 1])
    gps = (trip.track[:, :2] - origin) * scale
    nodes = (trip.nodes - origin) * scale
    start = nodes[trip.arcs[:, 0]]
    vector = nodes[trip.arcs[:, 1]] - start
    length = np.linalg.norm(vector, axis=1)

    time_steps = np.diff(trip.track[:, 2])
    in_same_block = time_steps <= gap_seconds
    displacements = np.diff(gps, axis=0) * in_same_block[:, None]
    elapsed_steps = time_steps * in_same_block
    # Use both neighbouring steps when available and one at a block boundary.
    motion = np.zeros_like(gps)
    elapsed = np.zeros(len(gps))
    motion[:-1] += displacements
    motion[1:] += displacements
    elapsed[:-1] += elapsed_steps
    elapsed[1:] += elapsed_steps
    movement = np.linalg.norm(motion, axis=1)
    speed = np.divide(movement, elapsed, out=np.zeros_like(elapsed), where=elapsed > 0)
    heading_available = (movement >= 3) & (speed >= 0.75)
    heading = np.divide(
        motion, movement[:, None], out=np.zeros_like(motion), where=heading_available[:, None],
    )
    return Geometry(
        gps=gps,
        start=start,
        vector=vector,
        length=length,
        heading=heading,
        heading_available=heading_available,
        speed=speed,
        block_starts=np.r_[0, np.flatnonzero(~in_same_block) + 1],
    )


def project_to_arcs(points, start, vector, length):
    """Return Euclidean distance and position along each segment, with broadcasting."""
    offset = points - start
    fraction = np.divide(
        np.sum(offset * vector, axis=-1), length**2,
        out=np.zeros(offset.shape[:-1]), where=length > 0,
    )
    np.clip(fraction, 0, 1, out=fraction)
    distance = np.linalg.norm(offset - vector * fraction[..., None], axis=-1)
    return distance, fraction
