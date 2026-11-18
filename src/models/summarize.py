"""Aggregate the CSVs into the comparison table + plots.   python -m src.models.summarize"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PLOTS_DIR
from .evaluate import OUT_DIR

METRICS = ["top1", "macro_f1", "logloss", "route_continuity", "candidate_recall"]


def main():
    cv = pd.read_csv(OUT_DIR / "cv_results.csv")
    agg = cv.groupby("model")[METRICS + ["fit_s", "infer_s"]].agg(["mean", "std"]).round(4)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.sort_values("top1_mean", ascending=False)
    print("\nCV (mean/std over folds)\n", agg[["top1_mean", "top1_std", "macro_f1_mean", "logloss_mean",
                                              "route_continuity_mean", "fit_s_mean", "infer_s_mean"]])
    wide = cv.pivot(index="fold", columns="model", values="top1")
    best = agg.index[0]
    print(f"\nfold-by-fold wins of {best} (top1):",
          {m: int((wide[best] > wide[m]).sum()) for m in wide.columns if m != best}, "of", len(wide))
    table = agg.copy()
    st_path = OUT_DIR / "student_test_results.csv"
    if st_path.exists():
        st = pd.read_csv(st_path).set_index("model")[["top1", "macro_f1", "logloss", "route_continuity", "infer_s"]]
        st.columns = [f"student_{c}" for c in st.columns]
        table = table.join(st)
        print("\nstudent-test\n", st.round(4))
    table.to_csv(OUT_DIR / "comparison_table.csv")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (m, title) in zip(ax, [("top1", "CV Top-1 accuracy"), ("macro_f1", "CV macro F1"), ("logloss", "CV log loss")]):
        a.bar(agg.index, agg[f"{m}_mean"], yerr=agg[f"{m}_std"], capsize=4); a.set_title(title)
        a.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "cv_model_comparison.png", dpi=150)
    print("saved", PLOTS_DIR / "cv_model_comparison.png")


if __name__ == "__main__":
    main()
