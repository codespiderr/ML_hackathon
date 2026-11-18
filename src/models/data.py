"""Load the per-trip .npz archives written by src.data_processing.prepare.

Layout of one archive: features (obs, K, F), labels (obs, K), arc_ids (obs, K),
block_starts, ... Only `features` is ever used as model input. Everything else is for scoring.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import DATA_DIR, RESULTS_DIR

PRE_DIR = RESULTS_DIR / "preprocessing"          # change with --pre-dir if you regenerated elsewhere
CACHE_DIR = Path("scratch") / "arcs_cache"       # scratch/ is git-ignored


@dataclass
class SplitData:
    split: str
    trip_ids: list
    X: np.ndarray            # (n_obs*K, F) float32, trip-major, then time, then candidate
    y: np.ndarray            # (n_obs*K,)   int8   1 = aligned reference arc
    arc_ids: np.ndarray      # (n_obs, K)   trip-local arc ids
    trip_of_obs: np.ndarray  # (n_obs,)     index into trip_ids
    block_start: np.ndarray  # (n_obs,)     True where a new GPS block (gap > threshold) begins
    K: int
    feature_names: list

    @property
    def n_obs(self):
        return len(self.trip_of_obs)

    def obs_mask(self, trip_indices):
        return np.isin(self.trip_of_obs, list(trip_indices))

    def row_mask(self, obs_mask):
        return np.repeat(obs_mask, self.K)


def load_split(split, pre_dir=PRE_DIR, max_trips=None):
    files = sorted((Path(pre_dir) / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files in {Path(pre_dir) / split}; run src.data_processing.prepare")
    if max_trips:
        files = files[:max_trips]
    X, Y, A, T, B, ids, names, K = [], [], [], [], [], [], None, None
    for t, f in enumerate(files):
        with np.load(f) as z:
            feats, lab = z["features"], z["labels"]
            if K is None:
                K, names = feats.shape[1], [str(n) for n in z["feature_names"]]
            assert feats.shape[1] == K, f"{f.name}: candidate count differs"
            n = len(feats)
            bs = np.zeros(n, dtype=bool); bs[z["block_starts"]] = True
            X.append(feats.reshape(-1, feats.shape[2])); Y.append(lab.reshape(-1).astype(np.int8))
            A.append(z["arc_ids"]); T.append(np.full(n, t)); B.append(bs); ids.append(f.stem)
    return SplitData(split, ids, np.concatenate(X), np.concatenate(Y), np.concatenate(A),
                     np.concatenate(T), np.concatenate(B), K, names)


def load_network(split, trip_id):
    """(from_node, to_node) int32 arrays indexed by arc id, read from the raw .arcs file (cached)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{split}_{trip_id}.npy"
    if cache.exists():
        return tuple(np.load(cache))
    arcs = np.loadtxt(DATA_DIR / split / trip_id / f"{trip_id}.arcs", dtype=np.int32, ndmin=2)
    out = np.stack([arcs[:, 0], arcs[:, 1]])
    np.save(cache, out)
    return tuple(out)
