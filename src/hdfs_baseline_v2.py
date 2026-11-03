#!/usr/bin/env python3
"""
HDFS no-drift baseline, v2.

Changes versus the first baseline:
  * Chronological train / validation / test split. The test set is the same final 20% of blocks as
    before (int(n * 0.8) onward); the earlier 80% is split 75/25 into train (60% overall) and
    validation (20% overall).
  * Thresholds are chosen on VALIDATION scores only, then frozen before the test set is scored.
    Two rules: "f1_val" (maximise F1 on validation) and "fpr_val" (target false-positive rate on
    validation normals).
  * Several fixed detectors: Isolation Forest (fit on all train blocks / normal train blocks only),
    PCA reconstruction error (deterministic, Xu et al. 2009 style), plus supervised logistic
    regression and random forest as reference ceilings.
  * Threshold-free metrics (PR-AUC, ROC-AUC) next to P / R / F1 / FPR.
  * 95% CIs: t-interval across seeds for stochastic detectors, bootstrap over test blocks.
  * Diagnostics: how many anomalous test blocks have a count vector identical to a normal training
    block (no count-vector detector can separate those), and window sizes for recovery metrics.

Input (rows MUST be in chronological order, i.e. the order blocks were first seen):
  --counts  CSV with a BlockId column and one numeric column per Drain template (E1..E46)
  --labels  CSV with BlockId and Label (Anomaly/Normal or 1/0), e.g. LogHub anomaly_label.csv
  or --npz  file with arrays X (n_blocks x n_templates) and y (0/1)

Example:
  python hdfs_baseline_v2.py --counts counts.csv --labels anomaly_label.csv --out results_v2.json
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

UNSUP_PERCENTILES = [95.0, 97.0, 98.0, 99.0, 99.5]   # label-free thresholds (percentile of TRAIN scores)
DROP_COLS = {"blockid", "label", "type", "timeinterval", "latency", "date", "time", "block_id"}
STOCHASTIC = {"iforest_all": True, "iforest_normal": True, "pca_normal": False, "pca_unsupervised": False,
              "logreg_supervised": False, "rf_supervised": True}


# ----------------------------------------------------------------------------- data
def load_data(args):
    if args.npz:
        d = np.load(args.npz)
        return d["X"].astype(np.float64), d["y"].astype(int), None
    counts = pd.read_csv(args.counts)
    labels = pd.read_csv(args.labels)
    cmap = {c.lower(): c for c in counts.columns}
    lmap = {c.lower(): c for c in labels.columns}
    bid_c = cmap.get("blockid") or cmap.get("block_id")
    bid_l = lmap.get("blockid") or lmap.get("block_id")
    if not bid_c or not bid_l or "label" not in lmap:
        sys.exit("counts needs BlockId; labels needs BlockId and Label")
    lab = labels[[bid_l, lmap["label"]]].rename(columns={bid_l: "BlockId", lmap["label"]: "Label"})
    counts = counts.rename(columns={bid_c: "BlockId"})
    counts = counts.drop(columns=[c for c in counts.columns if c.lower() == "label"])
    df = counts.merge(lab, on="BlockId", how="inner")
    if len(df) < len(counts):
        print(f"warning: {len(counts) - len(df)} blocks had no label and were dropped", file=sys.stderr)
    y = df["Label"].map(lambda v: 1 if str(v).strip().lower() in ("anomaly", "1", "true") else 0).to_numpy()
    feat = [c for c in df.columns
            if c.lower() not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    return df[feat].to_numpy(dtype=np.float64), y, feat


def make_splits(n, shuffle_seed=None):
    idx = np.arange(n)
    if shuffle_seed is not None:
        idx = np.random.default_rng(shuffle_seed).permutation(n)
    test_start = int(n * 0.8)
    dev, test = idx[:test_start], idx[test_start:]
    cut = int(len(dev) * 0.75)
    return dev[:cut], dev[cut:], test


# ----------------------------------------------------------------------------- detectors
def fit_score(name, Xtr, ytr, Xval, Xte, seed):
    """Return (val_scores, test_scores); higher score = more anomalous."""
    if name in ("iforest_all", "iforest_normal"):
        Xfit = Xtr if name == "iforest_all" else Xtr[ytr == 0]
        m = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xfit)
        return -m.score_samples(Xval), -m.score_samples(Xte)
    if name == "pca_normal":
        spe = pca_spe_fn(Xtr[ytr == 0])          # uses labels to pick normal training blocks
        return spe(Xval), spe(Xte)
    if name == "logreg_supervised":
        m = LogisticRegression(class_weight="balanced", max_iter=2000).fit(np.log1p(Xtr), ytr)
        return m.predict_proba(np.log1p(Xval))[:, 1], m.predict_proba(np.log1p(Xte))[:, 1]
    if name == "rf_supervised":
        m = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        return m.predict_proba(Xval)[:, 1], m.predict_proba(Xte)[:, 1]
    raise ValueError(name)


def pca_spe_fn(Xfit):
    """Fit TF-IDF-weighted PCA (95% variance) on Xfit; return squared-prediction-error scorer."""
    df = (Xfit > 0).sum(axis=0)
    idf = np.log((len(Xfit) + 1) / (df + 1)) + 1.0
    pca = PCA(n_components=0.95, svd_solver="full").fit(Xfit * idf)

    def spe(X):
        Z = X * idf
        R = pca.inverse_transform(pca.transform(Z))
        return ((Z - R) ** 2).sum(axis=1)
    return spe


# ----------------------------------------------------------------------------- thresholds
def pick_thresholds(sv, yv, target_fpr):
    """Both rules use validation data only. Returns {rule: (threshold, inclusive)}."""
    if yv.sum() == 0 or (yv == 0).sum() == 0:
        sys.exit("validation split needs both classes")
    p, r, t = precision_recall_curve(yv, sv)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    s_norm = np.sort(sv[yv == 0])
    k = min(max(int(math.ceil((1 - target_fpr) * len(s_norm))) - 1, 0), len(s_norm) - 1)
    return {"f1_val": (float(t[i]), True), "fpr_val": (float(s_norm[k]), False)}


def predict(scores, thr, inclusive):
    return scores >= thr if inclusive else scores > thr


# ----------------------------------------------------------------------------- metrics
def prf(y, pred):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def full_metrics(y, scores, pred):
    m = prf(y, pred)
    m["pr_auc"] = float(average_precision_score(y, scores))
    m["roc_auc"] = float(roc_auc_score(y, scores))
    return m


def bootstrap_ci(y, scores, preds, n_boot, seed=123):
    """Percentile 95% CI over test blocks. preds: {rule: bool array}."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ["precision", "recall", "f1", "fpr"]
    store = {r: {k: [] for k in keys} for r in preds}
    store["_auc"] = {"pr_auc": [], "roc_auc": []}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy, ss = y[i], scores[i]
        if yy.sum() == 0:
            continue
        for r, p in preds.items():
            m = prf(yy, p[i])
            for k in keys:
                store[r][k].append(m[k])
        store["_auc"]["pr_auc"].append(average_precision_score(yy, ss))
        store["_auc"]["roc_auc"].append(roc_auc_score(yy, ss))
    out = {}
    for r, d in store.items():
        out[r] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
    return out


def agg(vals):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return {"mean": float(a.mean()), "std": 0.0, "ci95_halfwidth": 0.0, "n": len(a)}
    sd = float(a.std(ddof=1))
    return {"mean": float(a.mean()), "std": sd,
            "ci95_halfwidth": float(stats.t.ppf(0.975, len(a) - 1) * sd / math.sqrt(len(a))), "n": len(a)}


# ----------------------------------------------------------------------------- diagnostics
def row_hash(X):
    return pd.util.hash_pandas_object(pd.DataFrame(X.astype(np.int64)), index=False).to_numpy()


def diagnostics(X, y, tr, va, te):
    h = row_hash(X)
    h_tr_norm = set(h[tr][y[tr] == 0].tolist())
    te_anom = h[te][y[te] == 1]
    same = int(np.isin(te_anom, list(h_tr_norm)).sum()) if len(te_anom) else 0
    d = pd.DataFrame({"h": h[tr], "y": y[tr]}).groupby("h")["y"].agg(["min", "max"])
    rate = float(y[te].mean())
    return {
        "blocks": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "anomaly_rate": {"train": float(y[tr].mean()), "val": float(y[va].mean()), "test": rate},
        "unique_count_vectors": {"train": int(len(np.unique(h[tr]))), "test": int(len(np.unique(h[te])))},
        "train_vectors_with_both_labels": int(((d["min"] == 0) & (d["max"] == 1)).sum()),
        "test_anomalies": int(len(te_anom)),
        "test_anomalies_identical_to_a_normal_train_vector": same,
        "test_anomalies_identical_fraction": float(same / len(te_anom)) if len(te_anom) else 0.0,
        "window_blocks_for_target_anomalies": {str(k): int(math.ceil(k / rate)) if rate else None for k in (30, 50, 100)},
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts"); ap.add_argument("--labels"); ap.add_argument("--npz")
    ap.add_argument("--out", default="results_v2.json")
    ap.add_argument("--detectors", nargs="+", default=list(STOCHASTIC))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--target-fpr", type=float, default=0.001)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--random-split", action="store_true",
                    help="shuffle before splitting (comparison only; not for drift experiments)")
    args = ap.parse_args()
    if not args.npz and not (args.counts and args.labels):
        sys.exit("give --npz, or --counts and --labels")

    t0 = time.time()
    X, y, feats = load_data(args)
    tr, va, te = make_splits(len(y), 0 if args.random_split else None)
    diag = diagnostics(X, y, tr, va, te)
    print(json.dumps(diag, indent=2))

    results = {"settings": vars(args), "diagnostics": diag, "detectors": {}}
    for name in args.detectors:
        seeds = args.seeds if STOCHASTIC[name] else [args.seeds[0]]
        per_seed, first = [], None
        for sd in seeds:
            if name == "pca_unsupervised":       # no labels anywhere in fit or thresholding
                spe = pca_spe_fn(X[tr])
                s_tr, sv, st = spe(X[tr]), spe(X[va]), spe(X[te])
                thr = {f"train_p{p:g}": (float(np.percentile(s_tr, p)), True) for p in UNSUP_PERCENTILES}
            else:
                sv, st = fit_score(name, X[tr], y[tr], X[va], X[te], sd)
                thr = pick_thresholds(sv, y[va], args.target_fpr)
            row = {"seed": sd, "rules": {}}
            preds = {}
            for rule, (t, inc) in thr.items():
                preds[rule] = predict(st, t, inc)
                row["rules"][rule] = {"threshold": t, **full_metrics(y[te], st, preds[rule])}
            per_seed.append(row)
            if first is None:
                first = (st, preds)
        summary = {}
        rules = list(per_seed[0]["rules"])
        for rule in rules:
            summary[rule] = {k: agg([r["rules"][rule][k] for r in per_seed])
                             for k in ("precision", "recall", "f1", "fpr", "pr_auc", "roc_auc")}
        ci = bootstrap_ci(y[te], first[0], first[1], args.n_boot) if args.n_boot else None
        results["detectors"][name] = {"per_seed": per_seed, "summary_across_seeds": summary,
                                      "bootstrap_ci95_test_blocks_seed0": ci}
        print(f"\n== {name} ({len(seeds)} seed(s)) ==")
        for rule in rules:
            s = summary[rule]
            print(f"  [{rule:>10}] P={s['precision']['mean']:.3f} R={s['recall']['mean']:.3f}±{s['recall']['std']:.3f} "
                  f"F1={s['f1']['mean']:.3f}±{s['f1']['std']:.3f} FPR={s['fpr']['mean']:.5f} "
                  f"PR-AUC={s['pr_auc']['mean']:.3f} ROC-AUC={s['roc_auc']['mean']:.3f}")

    results["total_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out} in {results['total_seconds']}s")


if __name__ == "__main__":
    main()
