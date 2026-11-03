"""
Export the block x template count matrix using the SAME Drain3 parse as hdfs_baseline.py.
Rows are in first-appearance order (the order hdfs_baseline.py uses for its chronological split).

Usage (from the repo root):
    python src/export_counts.py --log data/HDFS.log --labels data/anomaly_label.csv --out results/counts.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdfs_baseline import parse_log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results/counts.csv")
    a = ap.parse_args()

    block_counts, block_first, miner, stats = parse_log(a.log)
    print(stats)
    lab = pd.read_csv(a.labels)
    label_map = dict(zip(lab["BlockId"], lab["Label"]))
    blocks = sorted((b for b in block_first if b in label_map), key=lambda b: block_first[b])
    ids = sorted(c.cluster_id for c in miner.drain.clusters)
    rows = []
    for b in blocks:
        c = block_counts[b]
        rows.append([b, label_map[b]] + [c.get(i, 0) for i in ids])
    df = pd.DataFrame(rows, columns=["BlockId", "Label"] + [f"E{i}" for i in ids])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} blocks x {len(ids)} templates")


if __name__ == "__main__":
    main()
