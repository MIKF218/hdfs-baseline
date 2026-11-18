#!/usr/bin/env python3
"""
Leakage check for the HDFS baseline (group A): can anything from the test period reach the detector
through the parser or the feature construction?

Drain3 is run once over the whole log, before the train/test split. This script checks what that implies:

  1. Parser causality. Re-parse only the lines written before the test period with a fresh Drain3
     instance and compare template IDs line by line with the full-log parse. Identical IDs mean the
     parse of pre-test lines does not depend on anything that comes later.
  2. Template timing. When Drain3 created each template, relative to the split boundaries.
  3. Session overlap. Blocks are assigned to train/validation/test by FIRST appearance, but a block can
     keep receiving lines later. Count the training-block lines written after each boundary.
  4. Strict re-runs of the primary detector (PCA + Q-statistic, same code as hdfs_baseline_pca_q.py):
       a. as reported (full training sessions, all template columns);
       b. full training sessions, features restricted to templates seen in training, every other
          template mapped to one shared UNKNOWN feature;
       c. training vectors built only from lines written before the test period starts;
       d. training vectors built only from lines written before the training period ends;
       e. only training blocks that are complete before the training period ends.
     The test set is identical in every variant (final 20% of blocks, full sessions).

Labels are used only to compute metrics.

Usage (from the repo root):
    python src/check_parser_leakage.py --log data/HDFS.log --labels data/anomaly_label.csv \
        --out results/leakage_check.json
"""
import argparse
import json
import os
import sys
import time
from array import array

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdfs_baseline import BLOCK_RE, LINE_RE, build_miner  # noqa: E402
from hdfs_baseline_pca_q import ALPHA, fit_pca_q  # noqa: E402
from hdfs_baseline_v2 import full_metrics  # noqa: E402


class ParseState:
    """Resumable single pass of Drain3 over the log, with the same rules as hdfs_baseline.parse_log."""

    def __init__(self):
        self.miner = build_miner()
        self.offset = 0                 # byte offset reached in the log
        self.done = False
        self.cid = array("H")           # template ID of every parsed line, in log order
        self.pair_pos = array("i")      # (parsed-line position, block index) for every line/block pair
        self.pair_blk = array("i")
        self.block_index = {}           # BlockId -> index in first-appearance order
        self.block_first = array("i")   # parsed-line position of each block's first line
        self.created = {}               # template ID -> parsed-line position at which Drain3 created it
        self.unmatched = 0
        self.seconds = 0.0

    def advance(self, log_path, stop_at=None, deadline=None, progress=True):
        """Parse from the current offset to the end, to `stop_at` parsed lines, or until `deadline`."""
        t0 = time.time()
        with open(log_path, "rb") as f:
            f.seek(self.offset)
            for raw in f:
                if stop_at is not None and len(self.cid) >= stop_at:
                    self.done = True
                    break
                self.offset += len(raw)
                m = LINE_RE.match(raw.decode("utf-8", "replace").rstrip("\r\n"))
                if not m:
                    self.unmatched += 1
                    continue
                content = m.group(6)
                blocks = set(BLOCK_RE.findall(content))
                if not blocks:
                    continue
                c = self.miner.add_log_message(content)["cluster_id"]
                pos = len(self.cid)
                self.cid.append(c)
                if c not in self.created:
                    self.created[c] = pos
                for b in blocks:
                    j = self.block_index.get(b)
                    if j is None:
                        j = len(self.block_index)
                        self.block_index[b] = j
                        self.block_first.append(pos)
                    self.pair_pos.append(pos)
                    self.pair_blk.append(j)
                if (pos + 1) % 20000 == 0:
                    if progress and (pos + 1) % 1_000_000 == 0:
                        print(f"  parsed {pos + 1:,} lines", flush=True)
                    if deadline is not None and time.time() > deadline:
                        break
            else:
                self.done = True
        self.seconds += time.time() - t0
        return self.done


def boundaries(st, labels_path):
    """Chronological 60/20/20 block split (same as hdfs_baseline_v2.make_splits) and its line positions."""
    lab = pd.read_csv(labels_path)
    label_map = dict(zip(lab["BlockId"], (lab["Label"] == "Anomaly").astype(int)))
    y_all = np.array([label_map.get(b, -1) for b in st.block_index], dtype=np.int64)
    keep = y_all >= 0
    first = np.frombuffer(st.block_first, dtype=np.int32)[keep]
    n = int(keep.sum())
    test_start = int(n * 0.8)
    cut = int(test_start * 0.75)
    return {"keep": keep, "y": y_all[keep], "n": n, "cut": cut, "test_start": test_start,
            "pos_train_end": int(first[cut]), "pos_test_start": int(first[test_start])}


def analyse(st, b, prefix_check=None):
    y, n, cut, ts = b["y"], b["n"], b["cut"], b["test_start"]
    p_val, p_test = b["pos_train_end"], b["pos_test_start"]
    n_lines = len(st.cid)

    cid = np.frombuffer(st.cid, dtype=np.uint16)
    pos = np.frombuffer(st.pair_pos, dtype=np.int32)
    blk_raw = np.frombuffer(st.pair_blk, dtype=np.int32)
    remap = np.cumsum(b["keep"]) - 1                    # drop unlabelled blocks, if any
    ok = b["keep"][blk_raw]
    pos, blk = pos[ok], remap[blk_raw[ok]]

    tids = sorted(st.created)
    K = len(tids)
    lut = np.zeros(max(tids) + 1, dtype=np.int64)
    lut[tids] = np.arange(K)
    col = lut[cid[pos]]
    created = np.array([st.created[t] for t in tids])
    text = {c.cluster_id: c.get_template() for c in st.miner.drain.clusters}

    def counts(mask=None):
        key = blk * K + col if mask is None else blk[mask] * K + col[mask]
        return np.bincount(key, minlength=n * K).reshape(n, K)

    X = counts()
    is_train = blk < cut

    # ---- 2. template timing
    tr_tot, va_tot, te_tot = X[:cut].sum(0), X[cut:ts].sum(0), X[ts:].sum(0)
    late = [{"template_id": int(t), "created_at_line": int(created[j]),
             "period": "validation" if created[j] < p_test else "test",
             "events_in_train_blocks": int(tr_tot[j]), "events_in_validation_blocks": int(va_tot[j]),
             "events_in_test_blocks": int(te_tot[j]), "template": text.get(t, "")}
            for j, t in enumerate(tids) if created[j] >= p_val]
    timing = {
        "templates_total": K,
        "created_during_training_period": int((created < p_val).sum()),
        "created_during_validation_period": int(((created >= p_val) & (created < p_test)).sum()),
        "created_during_test_period": int((created >= p_test).sum()),
        "templates_absent_from_training_blocks": int((tr_tot == 0).sum()),
        "templates_absent_from_training_blocks_but_present_in_test_blocks": int(((tr_tot == 0) & (te_tot > 0)).sum()),
        "late_templates": late,
    }

    # ---- 3. session overlap
    def overlap(p):
        m = is_train & (pos >= p)
        blocks_hit = np.unique(blk[m])
        by_t = np.bincount(col[m], minlength=K)
        top = [{"template_id": int(tids[j]), "lines": int(by_t[j]), "template": text.get(tids[j], "")}
               for j in np.argsort(-by_t)[:5] if by_t[j] > 0]
        return {"training_block_lines_after_boundary": int(m.sum()),
                "share_of_training_block_lines": float(m.sum() / max(int(is_train.sum()), 1)),
                "training_blocks_affected": int(len(blocks_hit)),
                "share_of_training_blocks": float(len(blocks_hit) / cut),
                "affected_blocks_anomalous": int(y[blocks_hit].sum()),
                "top_templates_of_late_lines": top}, blocks_hit

    ov_test, _ = overlap(p_test)
    ov_val, hit_val = overlap(p_val)

    # ---- 4. strict re-runs
    Xte, yte = X[ts:], y[ts:]

    def run(Xtr, restrict=True):
        seen = Xtr.sum(0) > 0 if restrict else np.ones(K, dtype=bool)

        def proj(M):
            M = M.astype(np.float64)
            return np.hstack([M[:, seen], M[:, ~seen].sum(1, keepdims=True)]) if restrict else M

        spe, q, info = fit_pca_q(proj(Xtr))
        T = proj(Xte)
        s = spe(T)
        pred = s > q
        m = full_metrics(yte, s, pred)
        out = {"training_blocks": int(len(Xtr)), "features": int(seen.sum()) + int(restrict),
               "pca_components_kept": info["n_components_kept"], "q_threshold": info["q_threshold"],
               **{k: m[k] for k in ("precision", "recall", "f1", "fpr", "pr_auc", "roc_auc", "tp", "fp", "fn", "tn")}}
        if restrict:
            unk = T[:, -1] > 0
            out["test_blocks_with_unknown_template"] = int(unk.sum())
            out["of_which_anomalous"] = int((unk & (yte == 1)).sum())
        return out

    complete = np.ones(cut, dtype=bool)
    complete[hit_val] = False
    variants = {
        "a_as_reported": run(X[:cut], restrict=False),
        "b_unknown_feature": run(X[:cut]),
        "c_train_lines_before_test_period": run(counts(is_train & (pos < p_test))[:cut]),
        "d_train_lines_before_training_period_end": run(counts(is_train & (pos < p_val))[:cut]),
        "e_complete_training_blocks_only": run(X[:cut][complete]),
    }

    return {
        "log": {"lines_parsed": n_lines, "lines_unmatched": st.unmatched, "blocks": n,
                "line_block_pairs": int(len(pos)), "parse_seconds": round(st.seconds, 1)},
        "split": {"train_blocks": cut, "validation_blocks": ts - cut, "test_blocks": n - ts,
                  "training_period_ends_at_line": p_val, "test_period_starts_at_line": p_test,
                  "share_of_lines_before_training_period_end": p_val / n_lines,
                  "share_of_lines_before_test_period": p_test / n_lines},
        "1_parser_causality": prefix_check,
        "2_template_timing": timing,
        "3_session_overlap": {"after_test_period_starts": ov_test, "after_training_period_ends": ov_val},
        "4_pca_q_variants": {"alpha": ALPHA, "test_blocks": int(len(yte)), "test_anomalies": int(yte.sum()),
                             "variants": variants},
    }, X


def prefix_compare(st_full, st_prefix, p_test):
    a = np.frombuffer(st_prefix.cid, dtype=np.uint16)[:p_test]
    bfull = np.frombuffer(st_full.cid, dtype=np.uint16)[:p_test]
    return {"lines_compared": int(len(a)),
            "template_id_mismatches": int((a != bfull).sum()) if len(a) == len(bfull) else None,
            "templates_known_to_fresh_parser_at_boundary": int(len(st_prefix.created)),
            "templates_created_before_boundary_in_full_parse": int(sum(v < p_test for v in st_full.created.values()))}


def summarise(res):
    c, t, o, v = (res["1_parser_causality"], res["2_template_timing"], res["3_session_overlap"],
                  res["4_pca_q_variants"]["variants"])
    print("\n=== Leakage check ===")
    if c:
        print(f"1. Parser causality: {c['template_id_mismatches']} mismatches in {c['lines_compared']:,} pre-test lines")
    print(f"2. Templates: {t['templates_total']} total; created in training/validation/test period: "
          f"{t['created_during_training_period']}/{t['created_during_validation_period']}/{t['created_during_test_period']}; "
          f"absent from training blocks but present in test blocks: "
          f"{t['templates_absent_from_training_blocks_but_present_in_test_blocks']}")
    for name, key in (("test period starts", "after_test_period_starts"), ("training period ends", "after_training_period_ends")):
        d = o[key]
        print(f"3. Training-block lines after {name}: {d['training_block_lines_after_boundary']:,} "
              f"({d['share_of_training_block_lines']:.3%}) in {d['training_blocks_affected']:,} blocks "
              f"({d['share_of_training_blocks']:.3%})")
    print("4. PCA + Q-statistic on the same test set:")
    for k, m in v.items():
        print(f"   {k:42s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} FPR={m['fpr']:.5f} "
              f"PR-AUC={m['pr_auc']:.4f} tp={m['tp']} fp={m['fp']} fn={m['fn']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results/leakage_check.json")
    ap.add_argument("--skip-prefix-check", action="store_true", help="skip check 1 (saves a second parse)")
    args = ap.parse_args()

    print("Pass 1/2: parsing the whole log with Drain3 ...")
    st = ParseState()
    st.advance(args.log)
    b = boundaries(st, args.labels)

    check = None
    if not args.skip_prefix_check:
        print("Pass 2/2: re-parsing only the lines before the test period with a fresh parser ...")
        st2 = ParseState()
        st2.advance(args.log, stop_at=b["pos_test_start"])
        check = prefix_compare(st, st2, b["pos_test_start"])

    res, _ = analyse(st, b, check)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    summarise(res)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
