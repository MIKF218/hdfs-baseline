"""
HDFS no-drift baseline: Drain3 + Isolation Forest (block level)
Thesis: Streaming Log Anomaly Detection under Format Drift (Group A, no drift)

Usage:
    python hdfs_baseline.py --log HDFS.log --labels anomaly_label.csv --out results

Inputs (LogHub HDFS_v1):
    HDFS.log            raw log, one entry per line
    anomaly_label.csv   columns: BlockId, Label  (Label = Normal / Anomaly)

Protocol:
    1. Parse every line with Drain3 in original order (online, single pass).
    2. Group lines by block ID -> per-block template count vectors.
    3. Split blocks by first appearance: first TRAIN_FRACTION = train, rest = test.
    4. Templates never seen in training blocks -> one shared UNKNOWN feature.
    5. Fit Isolation Forest on training blocks WITHOUT labels.
    6. Threshold = percentile of training anomaly scores (no test labels used).
    7. Score test blocks; labels are used only for evaluation.
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.ensemble import IsolationForest

# ----------------------------------------------------------------------------
# Settings (record these in the report; reuse them unchanged in drift runs)
# ----------------------------------------------------------------------------
DRAIN_SIM_TH = 0.5            # Drain similarity threshold
DRAIN_DEPTH = 4               # Drain parse-tree depth
DRAIN_MAX_CHILDREN = 100
TRAIN_FRACTION = 0.8          # share of blocks (by first appearance) used for training
IF_N_ESTIMATORS = 100
IF_MAX_SAMPLES = "auto"
THRESHOLD_PERCENTILE = 99.0   # flag test blocks scoring at or above this percentile of training scores
SENSITIVITY_PERCENTILES = [95.0, 97.0, 99.0, 99.5]   # reported separately, for transparency only
SEEDS = [0, 1, 2, 3, 4]

# HDFS_v1 line: <Date> <Time> <Pid> <Level> <Component>: <Content>
LINE_RE = re.compile(r"^(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([^:]+):\s(.*)$")
BLOCK_RE = re.compile(r"(blk_-?\d+)")


def build_miner():
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = DRAIN_SIM_TH
    cfg.drain_depth = DRAIN_DEPTH
    cfg.drain_max_children = DRAIN_MAX_CHILDREN
    # Mask obvious variables before parsing (Drain's "domain knowledge" step)
    cfg.masking_instructions = [
        MaskingInstruction(r"blk_-?\d+", "BLK"),
        MaskingInstruction(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?", "IP"),
        MaskingInstruction(r"(?<![A-Za-z])-?\d+(?![A-Za-z])", "NUM"),
    ]
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def parse_log(log_path, report_every=1_000_000):
    """Single pass over the log. Returns per-block counts, first-seen index and miner."""
    miner = build_miner()
    block_counts = defaultdict(Counter)
    block_first = {}
    n_lines = n_bad = 0
    t0 = time.time()
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f):
            line = raw.rstrip("\n")
            m = LINE_RE.match(line)
            if not m:
                n_bad += 1
                continue
            content = m.group(6)
            blocks = set(BLOCK_RE.findall(content))
            if not blocks:
                continue
            cid = miner.add_log_message(content)["cluster_id"]
            for b in blocks:
                block_counts[b][cid] += 1
                if b not in block_first:
                    block_first[b] = i
            n_lines += 1
            if report_every and n_lines % report_every == 0:
                print(f"  parsed {n_lines:,} lines ({time.time() - t0:.0f}s)", flush=True)
    stats = {"lines_parsed": n_lines, "lines_unmatched": n_bad,
             "parse_seconds": round(time.time() - t0, 1)}
    return block_counts, block_first, miner, stats


def build_matrix(blocks, block_counts, feature_ids):
    """Count matrix; templates outside feature_ids go to the last (UNKNOWN) column."""
    col = {cid: j for j, cid in enumerate(feature_ids)}
    unk = len(feature_ids)
    X = np.zeros((len(blocks), len(feature_ids) + 1), dtype=np.float32)
    for r, b in enumerate(blocks):
        for cid, c in block_counts[b].items():
            X[r, col.get(cid, unk)] += c
    return X


def metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": p, "recall": r, "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    print("1/5 Parsing log with Drain3 ...")
    block_counts, block_first, miner, parse_stats = parse_log(args.log)

    print("2/5 Loading labels and splitting ...")
    lab = pd.read_csv(args.labels)
    label_map = dict(zip(lab["BlockId"], (lab["Label"] == "Anomaly").astype(int)))
    blocks = [b for b in block_first if b in label_map]
    blocks.sort(key=lambda b: block_first[b])           # order by first appearance
    n_train = int(len(blocks) * TRAIN_FRACTION)
    train_b, test_b = blocks[:n_train], blocks[n_train:]
    y_train = np.array([label_map[b] for b in train_b])
    y_test = np.array([label_map[b] for b in test_b])

    seen = sorted({cid for b in train_b for cid in block_counts[b]})
    all_ids = {c.cluster_id for c in miner.drain.clusters}
    unseen_in_test = sorted({cid for b in test_b for cid in block_counts[b]} - set(seen))

    print("3/5 Building count matrices ...")
    X_train = build_matrix(train_b, block_counts, seen)
    X_test = build_matrix(test_b, block_counts, seen)

    print("4/5 Fitting Isolation Forest (no labels) over seeds ...")
    runs, sens = [], {p: [] for p in SENSITIVITY_PERCENTILES}
    last_scores = None
    for seed in SEEDS:
        model = IsolationForest(n_estimators=IF_N_ESTIMATORS, max_samples=IF_MAX_SAMPLES,
                                random_state=seed, n_jobs=-1)
        model.fit(X_train)
        s_train = -model.score_samples(X_train)   # higher = more anomalous
        s_test = -model.score_samples(X_test)
        thr = np.percentile(s_train, THRESHOLD_PERCENTILE)
        m = metrics(y_test, (s_test >= thr).astype(int))
        m["seed"] = seed
        runs.append(m)
        for p in SENSITIVITY_PERCENTILES:
            sens[p].append(metrics(y_test, (s_test >= np.percentile(s_train, p)).astype(int))["f1"])
        last_scores = s_test

    print("5/5 Writing results ...")
    keys = ["precision", "recall", "f1", "fpr"]
    summary = {k: {"mean": float(np.mean([r[k] for r in runs])),
                   "std": float(np.std([r[k] for r in runs]))} for k in keys}
    result = {
        "settings": {"drain_sim_th": DRAIN_SIM_TH, "drain_depth": DRAIN_DEPTH,
                     "drain_max_children": DRAIN_MAX_CHILDREN,
                     "train_fraction": TRAIN_FRACTION,
                     "if_n_estimators": IF_N_ESTIMATORS,
                     "threshold_percentile": THRESHOLD_PERCENTILE, "seeds": SEEDS},
        "data": {**parse_stats,
                 "templates_total": len(all_ids),
                 "templates_seen_in_train": len(seen),
                 "templates_unseen_in_test": len(unseen_in_test),
                 "blocks_train": len(train_b), "blocks_test": len(test_b),
                 "anomaly_rate_train": float(y_train.mean()),
                 "anomaly_rate_test": float(y_test.mean())},
        "test_metrics_mean_std": summary,
        "test_metrics_per_seed": runs,
        "threshold_sensitivity_f1_mean": {str(p): float(np.mean(v)) for p, v in sens.items()},
        "total_seconds": round(time.time() - t_start, 1),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame([{"cluster_id": c.cluster_id, "size": c.size,
                   "seen_in_train": c.cluster_id in set(seen),
                   "template": c.get_template()} for c in miner.drain.clusters]
                 ).sort_values("cluster_id").to_csv(os.path.join(args.out, "templates.csv"), index=False)
    pd.DataFrame({"BlockId": test_b, "label": y_test, "score_last_seed": last_scores}
                 ).to_csv(os.path.join(args.out, "test_scores.csv"), index=False)

    print("\n=== Test results (mean ± std over seeds) ===")
    for k in keys:
        print(f"  {k:9s} {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
    d = result["data"]
    print(f"  templates: {d['templates_total']} total, {d['templates_seen_in_train']} seen in train, "
          f"{d['templates_unseen_in_test']} unseen in test")
    print(f"  blocks: {d['blocks_train']:,} train / {d['blocks_test']:,} test; "
          f"test anomaly rate {d['anomaly_rate_test']:.3%}")
    print(f"  total time {result['total_seconds']}s  ->  {args.out}/results.json")


if __name__ == "__main__":
    main()
