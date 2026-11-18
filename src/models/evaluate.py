"""Shared metrics + results logging. Every model is scored by these functions only.

Conventions: scores are P(candidate is the aligned reference arc) per (fix, candidate) row.
Top-1 = argmax over a fix's K candidates vs. label. A fix whose reference arc is not among
its K candidates (a candidate miss) counts as WRONG, so Top-1 <= candidate recall.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from src.config import RESULTS_DIR
from .data import load_network

OUT_DIR = RESULTS_DIR / "model_comparison"
COLUMNS = ["model", "fold", "top1", "macro_f1", "logloss", "route_continuity", "candidate_recall",
           "fit_s", "infer_s", "config", "seed"]


def top1_and_recall(labels_ok, p_obs):
    """labels_ok: (n_obs, K) 0/1; p_obs: (n_obs, K). Returns (top1, candidate_recall)."""
    top = p_obs.argmax(axis=1)
    return float(labels_ok[np.arange(len(top)), top].mean()), float(labels_ok.any(axis=1).mean())


def safe_log_loss(y, p, eps=1e-6):
    return float(log_loss(y, np.clip(p, eps, 1 - eps), labels=[0, 1]))


def macro_f1(y, p, thr=0.5):
    return float(f1_score(y, (p >= thr).astype(int), average="macro", zero_division=0))


def route_continuity(pred_arcs, trip_of_obs, block_start, networks):
    """Share of consecutive fixes (inside one GPS block) whose predicted arcs are identical
    or directly connected (to_node of arc i == from_node of arc i+1). Reuse for HMM output."""
    ok = total = 0
    for t in np.unique(trip_of_obs):
        idx = np.where(trip_of_obs == t)[0]
        a = pred_arcs[idx]
        within = ~block_start[idx][1:]                       # skip pairs that straddle a gap
        frm, to = networks[t]
        good = (a[:-1] == a[1:]) | (to[a[:-1]] == frm[a[1:]])
        ok += int((good & within).sum()); total += int(within.sum())
    return ok / total if total else float("nan")


def score_model(sd, obs_mask, p_rows, networks=None):
    """Full metric set for the fixes selected by obs_mask. p_rows aligns with sd.row_mask(obs_mask)."""
    rm = sd.row_mask(obs_mask)
    y = sd.y[rm]
    p_obs = p_rows.reshape(-1, sd.K)
    top1, recall = top1_and_recall(y.reshape(-1, sd.K), p_obs)
    out = dict(top1=top1, candidate_recall=recall, macro_f1=macro_f1(y, p_rows),
               logloss=safe_log_loss(y, p_rows), route_continuity=float("nan"))
    if networks is not None:
        pred = sd.arc_ids[obs_mask][np.arange(p_obs.shape[0]), p_obs.argmax(axis=1)]
        trips = sd.trip_of_obs[obs_mask]
        nets = {t: networks[t] for t in np.unique(trips)}
        out["route_continuity"] = route_continuity(pred, trips, sd.block_start[obs_mask], nets)
    return out


def load_networks(sd):
    return {i: load_network(sd.split, t) for i, t in enumerate(sd.trip_ids)}


def log_result(filename, model, fold, metrics, fit_s, infer_s, config, seed):
    """Upsert one row (same model+fold replaces the old one, so re-runs don't duplicate)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    row = dict(model=model, fold=fold, fit_s=round(fit_s, 2), infer_s=round(infer_s, 2),
               config=json.dumps(config, sort_keys=True, default=str), seed=seed,
               **{k: round(v, 6) for k, v in metrics.items()})
    df = pd.DataFrame([row])[COLUMNS]
    if path.exists():
        old = pd.read_csv(path, dtype={"fold": str})
        old = old[~((old.model == model) & (old.fold == str(fold)))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
