#!/usr/bin/env python3
"""
HDFS no-drift baseline, group A primary detector: PCA + Q-statistic threshold (fully label-free).

Pre-registered rule (fixed before this script was first run, 3 Nov 2026):
  * Same Drain3 counts and the same chronological split as hdfs_baseline_v2.py
    (train = first 60% of blocks, validation = next 20% (unused here), test = final 20%).
  * PCA fitted on ALL training blocks, no labels (TF-IDF weighting, components explaining 95%
    of variance, identical to pca_spe_fn in hdfs_baseline_v2.py).
  * Anomaly score = squared prediction error (SPE) in the residual subspace.
  * Threshold = Q-statistic of Jackson & Mudholkar (1979), as used by Xu et al. (2009),
    at alpha = 0.001, computed from the eigenvalues of the training data only.
  * Labels are read only to compute test metrics.

Usage:
  python src/hdfs_baseline_pca_q.py --counts results/counts.csv --labels data/anomaly_label.csv \
      --out results/results_pca_q.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdfs_baseline_v2 import bootstrap_ci, diagnostics, full_metrics, load_data, make_splits  # noqa: E402

ALPHA = 0.001
VAR_KEPT = 0.95


def fit_pca_q(Xfit, alpha=ALPHA, var_kept=VAR_KEPT):
    df = (Xfit > 0).sum(axis=0)
    idf = np.log((len(Xfit) + 1) / (df + 1)) + 1.0
    Z = Xfit * idf
    full = PCA(svd_solver="full").fit(Z)
    lam = full.explained_variance_
    k = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), var_kept) + 1)
    pca = PCA(n_components=k, svd_solver="full").fit(Z)

    resid = lam[k:]
    resid = resid[resid > 1e-12]
    th1, th2, th3 = (float((resid ** i).sum()) for i in (1, 2, 3))
    h0 = 1.0 - 2.0 * th1 * th3 / (3.0 * th2 ** 2)
    c = norm.ppf(1.0 - alpha)
    q = th1 * (c * np.sqrt(2.0 * th2 * h0 ** 2) / th1 + 1.0 + th2 * h0 * (h0 - 1.0) / th1 ** 2) ** (1.0 / h0)

    def spe(X):
        Zx = X * idf
        R = pca.inverse_transform(pca.transform(Zx))
        return ((Zx - R) ** 2).sum(axis=1)

    info = {"n_components_kept": k, "n_features": int(Xfit.shape[1]), "theta1": th1, "theta2": th2,
            "theta3": th3, "h0": float(h0), "alpha": alpha, "q_threshold": float(q)}
    return spe, float(q), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--npz")
    ap.add_argument("--out", default="results/results_pca_q.json")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    X, y, _ = load_data(args)
    tr, va, te = make_splits(len(y))
    diag = diagnostics(X, y, tr, va, te)

    spe, q, info = fit_pca_q(X[tr])          # no labels used
    s_tr, s_te = spe(X[tr]), spe(X[te])
    pred = s_te > q
    m = full_metrics(y[te], s_te, pred)
    ci = bootstrap_ci(y[te], s_te, {"q_stat": pred}, args.n_boot)

    result = {
        "settings": {"rule": "PCA SPE, Q-statistic (Jackson & Mudholkar 1979)", "alpha": ALPHA,
                     "variance_kept": VAR_KEPT, "fit_on": "all training blocks, no labels",
                     "split": "chronological 60/20/20 (validation unused)"},
        "pca": info,
        "train_fraction_flagged": float((s_tr > q).mean()),
        "test_fraction_flagged": float(pred.mean()),
        "diagnostics": diag,
        "test_metrics": m,
        "bootstrap_ci95_test_blocks": ci,
        "total_seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"PCA: kept {info['n_components_kept']} of {info['n_features']} components; "
          f"Q threshold (alpha={ALPHA}) = {q:.4f}")
    print(f"flagged: {result['train_fraction_flagged']:.3%} of train, {result['test_fraction_flagged']:.3%} of test")
    print(f"TEST  P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} FPR={m['fpr']:.5f} "
          f"PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    c = ci["q_stat"]
    print(f"95% CI  P={c['precision']} R={c['recall']} F1={c['f1']}")
    print(f"wrote {args.out} in {result['total_seconds']}s")


if __name__ == "__main__":
    main()
