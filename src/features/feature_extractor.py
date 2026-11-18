"""Build a feature tensor for the candidate arcs of each GPS observation."""

from dataclasses import dataclass

import numpy as np

from src.config import TOP_K_CANDIDATES
from src.data_processing.spatial_utils import Geometry, project_to_arcs

from .candidate_gen import nearest_arcs


FEATURE_NAMES = (
    "distance_m", "heading_cosine", "heading_available", "speed_mps",
    "arc_length_m", "projection_fraction", "distance_rank",
    "distance_above_nearest_m",
)


@dataclass
class Candidates:
    arc_ids: np.ndarray
    features: np.ndarray


def generate_candidates(geometry: Geometry, k: int = TOP_K_CANDIDATES) -> Candidates:
    """Return the exact nearest K directed arcs per fix and their feature tensor."""
    if k < 1:
        raise ValueError("k must be at least 1")
    arc_ids = nearest_arcs(geometry, min(k, len(geometry.start)))
    length = geometry.length[arc_ids]
    vector = geometry.vector[arc_ids]
    distance, fraction = project_to_arcs(
        geometry.gps[:, None, :], geometry.start[arc_ids], vector, length,
    )
    heading_dot = np.sum(geometry.heading[:, None, :] * vector, axis=-1)
    heading_cosine = np.divide(heading_dot, length, out=np.zeros_like(length), where=length > 0)
    np.clip(heading_cosine, -1, 1, out=heading_cosine)
    heading_available = geometry.heading_available[:, None] & (length > 0)
    speed = np.broadcast_to(geometry.speed[:, None], distance.shape)
    rank = np.broadcast_to(np.arange(1, distance.shape[1] + 1), distance.shape)
    distance_above_nearest = distance - distance[:, :1]
    features = np.stack((
        distance,
        heading_cosine,
        heading_available,
        speed,
        length,
        fraction,
        rank,
        distance_above_nearest,
    ), axis=2)
    return Candidates(arc_ids, features.astype(np.float32))
