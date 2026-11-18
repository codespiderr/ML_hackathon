"""Group-CV comparison of the five models, plus the one-shot student-test evaluation.

    python -m src.models.train_models                          # all 5 models, 5-fold group CV by trip
    python -m src.models.train_models --models logreg hgb      # subset (e.g. one person's models)
    python -m src.models.train_models --models hgb --student-test   # refit on all 80 trips, score the 10
    python -m src.models.train_models --max-trips 20 --folds 3 # quick smoke test

Rows go to results/model_comparison/cv_results.csv and student_test_results.csv.
Only the 8 `features` are model inputs; labels/arc ids are used for scoring only.
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR

from .data import PRE_DIR, load_split
from .evaluate import OUT_DIR, load_networks, log_result, score_model
from .folds import get_folds

SEED = RANDOM_SEED

# name -> (factory, config dict logged to CSV, max training FIXES (None = all))
MODELS = {
    "logreg": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)),
               {"family": "linear", "C": 1.0, "max_iter": 500, "scaler": "standard"}, None),
    "knn":    (lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
               {"family": "instance", "k": 25, "weights": "distance", "scaler": "standard"}, 40_000),
    "rf":     (lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_depth=20, max_samples=0.3,
                                               n_jobs=-1, random_state=SEED),
               {"family": "bagging", "n_estimators": 100, "min_samples_leaf": 20, "max_depth": 20, "max_samples": 0.3}, 60_000),
    "hgb":    (lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, early_stopping=False, random_state=SEED),
               {"family": "boosting", "learning_rate": 0.1, "max_iter": 200, "early_stopping": False}, None),
    "mlp":    (lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), batch_size=1024, early_stopping=True,
                                                                     n_iter_no_change=5, max_iter=40, random_state=SEED)),
               {"family": "neural", "hidden": [64, 32], "batch_size": 1024, "early_stopping": True,
                "n_iter_no_change": 5, "max_iter": 40, "scaler": "standard"}, 60_000),
}


def subsample_obs(train_obs_mask, cap, rng):
    """Cap training size by sampling whole FIXES (all K candidates stay together). Evaluation is never subsampled."""
    idx = np.where(train_obs_mask)[0]
    if cap is None or len(idx) <= cap:
        return train_obs_mask
    keep = np.zeros_like(train_obs_mask); keep[rng.choice(idx, cap, replace=False)] = True
    return keep


CAP_OVERRIDE = None   # set from --cap


def fit_predict(name, sd, train_obs, eval_obs):
    factory, config, cap = MODELS[name]
    cap = CAP_OVERRIDE or cap
    rng = np.random.default_rng(SEED)
    fit_mask = subsample_obs(train_obs, cap, rng)
    Xtr, ytr = sd.X[sd.row_mask(fit_mask)], sd.y[sd.row_mask(fit_mask)]
    model = factory()
    t0 = time.time(); model.fit(Xtr, ytr); fit_s = time.time() - t0
    t0 = time.time(); p = model.predict_proba(sd.X[sd.row_mask(eval_obs)])[:, 1]; infer_s = time.time() - t0
    return model, p, fit_s, infer_s, {**config, "n_train_fixes": int(fit_mask.sum())}


def save_loss_curve(model, tag):
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "loss_curve_"):
        d = {"train_loss": est.loss_curve_}
        if getattr(est, "validation_scores_", None) is not None:
            d["val_score"] = est.validation_scores_
        pd.DataFrame(d).rename_axis("epoch").to_csv(OUT_DIR / f"{tag}_loss_curve.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pre-dir", default=PRE_DIR, help="folder holding train/ and student_test/ .npz archives")
    ap.add_argument("--cap", type=int, default=None,
                    help="max training FIXES per model (each fix = K candidate rows); overrides per-model defaults")
    ap.add_argument("--max-trips", type=int, default=None, help="debug: use only the first N training trips")
    ap.add_argument("--no-continuity", action="store_true", help="skip route continuity (avoids reading raw .arcs files)")
    ap.add_argument("--student-test", action="store_true", help="refit on ALL training trips, score the student-test trips once")
    ap.add_argument("--save", action="store_true", help="with --student-test: save fitted model to saved_models/")
    a = ap.parse_args()
    global CAP_OVERRIDE
    CAP_OVERRIDE = a.cap
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train", a.pre_dir, a.max_trips)
    print(f"train: {len(train.trip_ids)} trips, {train.n_obs:,} fixes, {len(train.X):,} candidate rows")
    nets = None if a.no_continuity else load_networks(train)
    trip_fold = get_folds(train, a.folds) if a.max_trips is None else \
        np.arange(len(train.trip_ids)) % a.folds                     # debug runs never touch folds.json
    all_obs = np.ones(train.n_obs, dtype=bool)

    if not a.student_test:
        for name in a.models:
            for fold in range(a.folds):
                va = train.obs_mask(np.where(trip_fold == fold)[0]); tr = all_obs & ~va
                model, p, fit_s, inf_s, cfg = fit_predict(name, train, tr, va)
                m = score_model(train, va, p, nets)
                log_result("cv_results.csv", name, fold, m, fit_s, inf_s, cfg, SEED)
                if name == "mlp": save_loss_curve(model, f"mlp_fold{fold}")
                print(f"{name:7s} fold {fold}: top1={m['top1']:.4f} f1={m['macro_f1']:.4f} "
                      f"ll={m['logloss']:.4f} cont={m['route_continuity']:.4f} fit={fit_s:.0f}s", flush=True)
    else:
        test = load_split("student_test", a.pre_dir)
        tnets = None if a.no_continuity else load_networks(test)
        for name in a.models:
            model, p, fit_s, inf_s, cfg = fit_predict_split(name, train, test)
            m = score_model(test, np.ones(test.n_obs, dtype=bool), p, tnets)
            log_result("student_test_results.csv", name, "student_test", m, fit_s, inf_s, cfg, SEED)
            print(f"{name:7s} student-test: top1={m['top1']:.4f} f1={m['macro_f1']:.4f} ll={m['logloss']:.4f} "
                  f"cont={m['route_continuity']:.4f}", flush=True)
            if name == "mlp": save_loss_curve(model, "mlp_final")
            if a.save:
                MODELS_DIR.mkdir(exist_ok=True); joblib.dump(model, MODELS_DIR / f"{name}.joblib")


def fit_predict_split(name, train, test):
    """Fit on every training trip, predict every student-test row."""
    factory, config, cap = MODELS[name]
    cap = CAP_OVERRIDE or cap
    rng = np.random.default_rng(SEED)
    fit_mask = subsample_obs(np.ones(train.n_obs, dtype=bool), cap, rng)
    rows = train.row_mask(fit_mask)
    model = factory()
    t0 = time.time(); model.fit(train.X[rows], train.y[rows]); fit_s = time.time() - t0
    t0 = time.time(); p = model.predict_proba(test.X)[:, 1]; infer_s = time.time() - t0
    return model, p, fit_s, infer_s, {**config, "n_train_fixes": int(fit_mask.sum())}


if __name__ == "__main__":
    main()
