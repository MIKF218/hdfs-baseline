#!/usr/bin/env python3
"""
Controlled format-drift experiments, group A (Drain3 + PCA with Q-statistic threshold), HDFS.

Protocol (fixed, approved settings; nothing here is tuned on test results):
  * Training data is never changed. PCA and the Q threshold are fitted once on the training blocks
    (first 60%) of results/counts.csv, without labels. Templates not seen in training, and every
    template Drain3 creates after training, go to one shared UNKNOWN feature.
  * The parser is warmed up on the log up to the start of the test period (the first line of the
    first test block), then saved. Each scenario reloads that state and parses the test period online,
    exactly as in production. Drift is injected into the raw message text before parsing.
  * Drift only rewords eligible normal-event statements (src/paraphrases.py); parameters, line
    headers and anomaly-only messages are never changed. Drift is applied by time to every eligible
    line after onset, whichever block it belongs to; only test blocks are scored.
  * sudden : every eligible line is reworded from the onset point on.
  * gradual: from the onset point, each eligible line is reworded with a probability that rises
             linearly from 0 to 1 over the ramp interval, then stays at 1.
  * prevalence: a random subset of the 13 statements is reworded (statement share), others unchanged.
  * Labels are used only to compute metrics.

Scenarios (default "core"):
  none                       no drift; must reproduce the baseline (1,616 caught, 5 false alarms)
  sudden_{low,medium,high}   all 13 statements
  gradual_{low,medium,high}  all 13 statements
  prev{25,50}_{sudden,gradual}_medium_s{0,1,2}   prevalence runs (scenario set "prevalence")

Unseen templates (--unseen):
  unknown (default, pre-registered main design): every template not seen in training goes to one shared
          UNKNOWN feature, which is always 0 in training.
  drop    (robustness check): lines with an unseen template are not counted at all, so the model sees
          the block without them. Results go to a separate folder (default results/drift_drop).

Usage:
  python src/run_drift.py --log data/HDFS.log --labels data/anomaly_label.csv --counts results/counts.csv
  python src/run_drift.py ... --unseen drop
  python src/run_drift.py ... --scenarios prevalence
  python src/run_drift.py ... --scenarios sudden_high gradual_low

Outputs (results/drift/):
  parser_state_test_start.pkl   cached parser state at the start of the test period (built once)
  <scenario>.json               settings, parser statistics, metrics overall / by phase / per window
  summary.csv                   one row per scenario (rewritten after every scenario)
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdfs_baseline import BLOCK_RE, LINE_RE, build_miner  # noqa: E402
from hdfs_baseline_pca_q import fit_pca_q  # noqa: E402
from hdfs_baseline_v2 import full_metrics, prf  # noqa: E402
import paraphrases as PARA  # noqa: E402

ONSET = 0.25            # drift starts 25% into the test period (share of test-period lines)
RAMP = 0.50             # gradual drift reaches 100% after a further 50% of the test period
WINDOW_BLOCKS = 5000    # window size for metrics over time
PREVALENCE = (0.25, 0.50)
SUBSET_SEEDS = (0, 1, 2)
STATEMENT_IDS = [s["id"] for s in PARA.STATEMENTS]


# ----------------------------------------------------------------------------- scenarios
def scenario_list(names):
    core = ["none"] + [f"{t}_{l}" for t in ("sudden", "gradual") for l in PARA.LEVELS]
    prev = [f"prev{int(p * 100)}_{t}_medium_s{s}" for p in PREVALENCE for t in ("sudden", "gradual")
            for s in SUBSET_SEEDS]
    out = []
    for n in names:
        out += core if n == "core" else prev if n == "prevalence" else core + prev if n == "all" else [n]
    return list(dict.fromkeys(out))


def parse_scenario(name):
    if name == "none":
        return {"name": name, "type": "none", "level": None, "prevalence": 1.0, "subset_seed": None,
                "statements": []}
    parts = name.split("_")
    prevalence, subset_seed = 1.0, None
    if parts[0].startswith("prev"):
        prevalence = int(parts[0][4:]) / 100.0
        subset_seed = int(parts[3][1:])
        parts = parts[1:3]
    dtype, level = parts
    if dtype not in ("sudden", "gradual") or level not in PARA.LEVELS:
        sys.exit(f"unknown scenario: {name}")
    if prevalence < 1.0:
        k = max(1, math.floor(prevalence * len(STATEMENT_IDS) + 0.5))
        rng = np.random.default_rng(subset_seed)
        stmts = sorted(rng.choice(STATEMENT_IDS, size=k, replace=False).tolist())
    else:
        stmts = list(STATEMENT_IDS)
    return {"name": name, "type": dtype, "level": level, "prevalence": prevalence,
            "subset_seed": subset_seed, "statements": stmts}


# ----------------------------------------------------------------------------- data and model
def load_counts(path):
    d = pd.read_csv(path)
    ecols = sorted((c for c in d.columns if c.startswith("E")), key=lambda c: int(c[1:]))
    X = d[ecols].to_numpy(dtype=np.float64)
    y = (d["Label"] == "Anomaly").to_numpy().astype(int)
    n = len(d)
    test_start = int(n * 0.8)
    cut = int(test_start * 0.75)
    return d["BlockId"].to_numpy(), X, y, [int(c[1:]) for c in ecols], cut, test_start


def fit_model(X, cut, ids, unseen="unknown"):
    seen = X[:cut].sum(0) > 0
    col_of = {}
    j = 0
    for i, cid in enumerate(ids):
        if seen[i]:
            col_of[cid] = j
            j += 1
    if unseen == "drop":
        n_feat = j
        Xtr = X[:cut][:, seen]
    else:
        n_feat = j + 1                  # last column = UNKNOWN
        Xtr = np.hstack([X[:cut][:, seen], X[:cut][:, ~seen].sum(1, keepdims=True)])
    spe, q, info = fit_pca_q(Xtr)
    return spe, q, info, col_of, n_feat


# ----------------------------------------------------------------------------- parser state
def warm_parser(log_path, first_test_block, cache):
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            st = pickle.load(f)
        print(f"loaded parser state: {len(st['miner'].drain.clusters)} templates, "
              f"test period = {st['test_lines']:,} lines")
        return st
    print("warming up the parser on the log before the test period (once, about 4 minutes) ...")
    t0 = time.time()
    miner = build_miner()
    parsed, offset = 0, 0
    with open(log_path, "rb") as f:
        for raw in f:
            m = LINE_RE.match(raw.decode("utf-8", "replace").rstrip("\r\n"))
            if m:
                content = m.group(6)
                blocks = BLOCK_RE.findall(content)
                if blocks:
                    if first_test_block in blocks:
                        break
                    miner.add_log_message(content)
                    parsed += 1
                    if parsed % 1_000_000 == 0:
                        print(f"  {parsed:,} lines", flush=True)
            offset += len(raw)
        else:
            sys.exit("first test block not found in the log")
        test_lines = 0
        f.seek(offset)
        for raw in f:
            m = LINE_RE.match(raw.decode("utf-8", "replace").rstrip("\r\n"))
            if m and BLOCK_RE.search(m.group(6)):
                test_lines += 1
    st = {"miner": miner, "offset": offset, "lines_before_test": parsed, "test_lines": test_lines}
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache + ".tmp", "wb") as f:
        pickle.dump(st, f, protocol=4)
    os.replace(cache + ".tmp", cache)
    print(f"parser state saved ({time.time() - t0:.0f}s): {len(miner.drain.clusters)} templates, "
          f"{parsed:,} lines before the test period, {test_lines:,} in it")
    return st


# ----------------------------------------------------------------------------- one scenario
def run_scenario(sc, a, st_bytes, st, test_index, col_of, n_feat, base_ids):
    drop = a.unseen == "drop"
    miner = pickle.loads(st_bytes)["miner"]
    L = st["test_lines"]
    onset = int(a.onset * L)
    ramp = max(int(a.ramp * L), 1)
    stmts = set(sc["statements"])
    rng = np.random.default_rng(a.gradual_seed)
    n_test = len(test_index)
    C = np.zeros((n_test, n_feat), dtype=np.float64)
    first_pos = np.full(n_test, -1, dtype=np.int64)
    last_pos = np.full(n_test, -1, dtype=np.int64)
    touched = np.zeros(n_test, dtype=bool)          # block received at least one reworded line
    unk = n_feat - 1
    stats = {"lines_parsed": 0, "eligible_lines_after_onset": 0, "lines_reworded": 0,
             "reworded_lines_absorbed": 0, "reworded_lines_new_template": 0,
             "test_block_lines_not_counted": 0}
    new_templates, generalised = {}, {}
    sha = hashlib.sha1()                            # fingerprint of which lines were reworded (shared with group B)

    pos = 0
    t0 = time.time()
    with open(a.log, "rb") as f:
        f.seek(st["offset"])
        for raw in f:
            m = LINE_RE.match(raw.decode("utf-8", "replace").rstrip("\r\n"))
            if not m:
                continue
            content = m.group(6)
            blocks = set(BLOCK_RE.findall(content))
            if not blocks:
                continue
            reworded = False
            if sc["type"] != "none" and pos >= onset:
                s, params = PARA.match(content)
                if s is not None and s["id"] in stmts:
                    stats["eligible_lines_after_onset"] += 1
                    if sc["type"] == "sudden" or rng.random() < min((pos - onset) / ramp, 1.0):
                        content = s[sc["level"]].format(**params)
                        reworded = True
                        sha.update(f"{pos}:{s['id']};".encode())
            r = miner.add_log_message(content)
            cid = r["cluster_id"]
            if reworded:
                stats["lines_reworded"] += 1
                stats["reworded_lines_absorbed" if cid in base_ids else "reworded_lines_new_template"] += 1
            if cid not in base_ids and cid not in new_templates:
                new_templates[cid] = pos
            if cid in base_ids and r["change_type"] == "cluster_template_changed":
                generalised[cid] = r["template_mined"]
            col = col_of.get(cid, None if drop else unk)
            for b in blocks:
                i = test_index.get(b)
                if i is not None:
                    if col is None:
                        stats["test_block_lines_not_counted"] += 1
                    else:
                        C[i, col] += 1
                    if first_pos[i] < 0:
                        first_pos[i] = pos
                    last_pos[i] = pos
                    if reworded:
                        touched[i] = True
            pos += 1
            if a.max_test_lines and pos >= a.max_test_lines:
                break
    stats["lines_parsed"] = pos
    stats["parse_seconds"] = round(time.time() - t0, 1)
    stats["reworded_lines_sha1"] = sha.hexdigest()
    text = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    stats["new_templates"] = [{"template_id": int(c), "first_seen_at_test_line": int(p), "template": text[c]}
                              for c, p in sorted(new_templates.items())]
    stats["original_templates_generalised"] = [{"template_id": int(c), "template_now": t}
                                               for c, t in sorted(generalised.items())]
    return C, first_pos, last_pos, touched, stats, onset, ramp


def metrics_block(y, s, pred, unk_mask):
    m = prf(y, pred)
    out = {k: m[k] for k in ("precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn")}
    out["blocks"] = int(len(y))
    out["anomalous_blocks"] = int(y.sum())
    out["normal_blocks_with_unknown_feature"] = int((unk_mask & (y == 0)).sum())
    if 0 < y.sum() < len(y):
        fm = full_metrics(y, s, pred)
        out["pr_auc"], out["roc_auc"] = fm["pr_auc"], fm["roc_auc"]
    return out


def evaluate(C, first_pos, last_pos, touched, y, spe, q, onset, ramp, dtype, window, has_unknown=True):
    seen = first_pos >= 0
    s = spe(C)
    pred = s > q
    unk_mask = C[:, -1] > 0 if has_unknown else np.zeros(len(C), dtype=bool)
    res = {"blocks_scored": int(seen.sum()), "overall": metrics_block(y[seen], s[seen], pred[seen], unk_mask[seen])}
    # Phases by when a block's lines occur. Blocks live for hours (e.g. deletion lines long after
    # allocation), so a block that starts before the onset can still receive reworded lines.
    phases = {"before_onset": last_pos < onset,                       # every line before drift starts
              "spanning_onset": (first_pos < onset) & (last_pos >= onset)}
    if dtype == "gradual":
        phases["during_ramp"] = (first_pos >= onset) & (first_pos < onset + ramp)
        phases["after_ramp"] = first_pos >= onset + ramp
    else:
        phases["after_onset"] = first_pos >= onset
    res["by_phase"] = {k: metrics_block(y[m & seen], s[m & seen], pred[m & seen], unk_mask[m & seen])
                       for k, m in phases.items() if (m & seen).any()}
    # Direct split: blocks that received at least one reworded line vs blocks that did not.
    res["by_exposure"] = {k: metrics_block(y[m & seen], s[m & seen], pred[m & seen], unk_mask[m & seen])
                          for k, m in (("with_reworded_lines", touched), ("without_reworded_lines", ~touched))
                          if (m & seen).any()}
    order = np.argsort(np.where(seen, first_pos, np.iinfo(np.int64).max), kind="stable")[: int(seen.sum())]
    wins = []
    for k in range(0, len(order), window):
        idx = order[k:k + window]
        mm = prf(y[idx], pred[idx])
        wins.append({"window": k // window, "first_test_line": int(first_pos[idx[0]]),
                     "blocks": int(len(idx)), "anomalous": int(y[idx].sum()),
                     "fpr": mm["fpr"], "recall": mm["recall"], "f1": mm["f1"], "precision": mm["precision"],
                     "normal_blocks_with_unknown": int((unk_mask[idx] & (y[idx] == 0)).sum())})
    res["windows"] = wins
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--labels", required=True, help="kept for symmetry; labels are read from --counts")
    ap.add_argument("--counts", default="results/counts.csv")
    ap.add_argument("--out-dir", default=None, help="default results/drift, or results/drift_drop with --unseen drop")
    ap.add_argument("--unseen", choices=("unknown", "drop"), default="unknown")
    ap.add_argument("--parser-state", default="results/drift/parser_state_test_start.pkl",
                    help="cached parser state, shared by both --unseen modes")
    ap.add_argument("--scenarios", nargs="+", default=["core"])
    ap.add_argument("--onset", type=float, default=ONSET)
    ap.add_argument("--ramp", type=float, default=RAMP)
    ap.add_argument("--window-blocks", type=int, default=WINDOW_BLOCKS)
    ap.add_argument("--gradual-seed", type=int, default=0)
    ap.add_argument("--max-test-lines", type=int, default=0, help="debug only: stop early")
    a = ap.parse_args()
    if a.out_dir is None:
        a.out_dir = "results/drift_drop" if a.unseen == "drop" else "results/drift"
    os.makedirs(a.out_dir, exist_ok=True)

    block_ids, X, y_all, ids, cut, ts = load_counts(a.counts)
    spe, q, info, col_of, n_feat = fit_model(X, cut, ids, a.unseen)
    print(f"model ({a.unseen}): PCA kept {info['n_components_kept']} components, Q = {q:.4f}, "
          f"{len(col_of)} known templates" + (" + UNKNOWN" if a.unseen == "unknown" else ", unseen lines dropped"))
    test_ids = block_ids[ts:]
    test_index = {b: i for i, b in enumerate(test_ids)}
    y = y_all[ts:]
    base_ids = set(ids)

    st = warm_parser(a.log, test_ids[0], a.parser_state)
    st_bytes = pickle.dumps(st, protocol=4)

    summary_path = os.path.join(a.out_dir, "summary.csv")
    rows = {r["scenario"]: r for r in pd.read_csv(summary_path).to_dict("records")} \
        if os.path.exists(summary_path) else {}
    for name in scenario_list(a.scenarios):
        sc = parse_scenario(name)
        print(f"\n== {name}: statements {sc['statements'] or '-'}")
        C, first_pos, last_pos, touched, stats, onset, ramp = run_scenario(
            sc, a, st_bytes, st, test_index, col_of, n_feat, base_ids)
        res = evaluate(C, first_pos, last_pos, touched, y, spe, q, onset, ramp, sc["type"], a.window_blocks,
                       has_unknown=a.unseen == "unknown")
        out = {"scenario": sc,
               "settings": {"unseen": a.unseen, "onset": a.onset, "ramp": a.ramp, "onset_test_line": onset, "ramp_lines": ramp,
                            "test_period_lines": st["test_lines"], "gradual_seed": a.gradual_seed,
                            "window_blocks": a.window_blocks, "q_threshold": q,
                            "pca_components": info["n_components_kept"]},
               "parser": stats, **res}
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as f:
            json.dump(out, f, indent=2)

        o, ph, ex = res["overall"], res["by_phase"], res["by_exposure"]
        post = ph.get("after_ramp" if sc["type"] == "gradual" else "after_onset", {})
        hit, clean = ex.get("with_reworded_lines", {}), ex.get("without_reworded_lines", {})
        row = {"scenario": name, "unseen": a.unseen, "type": sc["type"], "level": sc["level"], "prevalence": sc["prevalence"],
               "subset_seed": sc["subset_seed"], "statements": " ".join(sc["statements"]),
               "lines_reworded": stats["lines_reworded"], "reworded_absorbed": stats["reworded_lines_absorbed"],
               "reworded_new_template": stats["reworded_lines_new_template"],
               "new_templates": len(stats["new_templates"]),
               **{f"all_{k}": o.get(k) for k in ("precision", "recall", "f1", "fpr", "pr_auc", "tp", "fp", "fn")},
               "before_onset_fpr": ph.get("before_onset", {}).get("fpr"),
               **{f"after_{k}": post.get(k) for k in ("blocks", "precision", "recall", "f1", "fpr", "pr_auc")},
               "blocks_with_reworded_lines": hit.get("blocks", 0),
               "with_reworded_fpr": hit.get("fpr"), "with_reworded_recall": hit.get("recall"),
               "without_reworded_fpr": clean.get("fpr"), "without_reworded_recall": clean.get("recall")}
        rows[name] = row
        pd.DataFrame(list(rows.values())).to_csv(summary_path, index=False)
        if a.unseen == "drop":
            print(f"   test-block lines not counted (unseen template): {stats['test_block_lines_not_counted']:,}")
        print(f"   reworded {stats['lines_reworded']:,} lines (absorbed {stats['reworded_lines_absorbed']:,}, "
              f"new template {stats['reworded_lines_new_template']:,}); {len(stats['new_templates'])} new templates; "
              f"parse {stats['parse_seconds']}s")
        print(f"   ALL   P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f} FPR={o['fpr']:.5f} "
              f"tp={o['tp']} fp={o['fp']} fn={o['fn']}")
        for k, v in list(ph.items()) + list(ex.items()):
            print(f"   {k:22s} blocks={v['blocks']:>7,} P={v['precision']:.4f} R={v['recall']:.4f} "
                  f"F1={v['f1']:.4f} FPR={v['fpr']:.5f} fp={v['fp']}")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
