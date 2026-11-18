"""Group folds by TRIP, balanced by number of GPS fixes, seeded. Create once, commit, never regenerate.

Trips differ a lot in size (hundreds to ~19k fixes), so round-robin gives lopsided folds.
Greedy rule: biggest trips first, each into the currently lightest fold; ties broken by a seeded shuffle.
"""
import json
from pathlib import Path

import numpy as np

from src.config import RANDOM_SEED, RESULTS_DIR

FOLDS_PATH = RESULTS_DIR / "model_comparison" / "folds.json"


def make_folds(trip_ids, obs_counts, n_splits=5, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trip_ids))                       # seeded tie-breaking
    order = sorted(order, key=lambda i: -obs_counts[i])          # stable: ties keep shuffled order
    load = np.zeros(n_splits); assignment = {}
    for i in order:
        f = int(np.argmin(load)); assignment[trip_ids[i]] = f; load[f] += obs_counts[i]
    return assignment


def get_folds(split_data, n_splits=5, path=FOLDS_PATH):
    """Load folds.json, or create it on first use. Returns trip_index -> fold array."""
    path = Path(path)
    if path.exists():
        d = json.loads(path.read_text())
        assert d["n_splits"] == n_splits, f"{path} was made with {d['n_splits']} folds"
        asg = d["assignment"]
        missing = [t for t in split_data.trip_ids if t not in asg]
        assert not missing, f"trips missing from {path}: {missing[:3]} (delete it only if you restart ALL experiments)"
    else:
        counts = np.bincount(split_data.trip_of_obs, minlength=len(split_data.trip_ids))
        asg = make_folds(split_data.trip_ids, counts, n_splits)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seed": RANDOM_SEED, "n_splits": n_splits, "assignment": asg}, indent=1))
        print(f"created {path}")
    return np.array([asg[t] for t in split_data.trip_ids])
