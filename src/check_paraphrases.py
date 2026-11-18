#!/usr/bin/env python3
"""
Review table for the paraphrases in paraphrases.py.

For every statement and level it reports:
  * magnitude: token-level edit distance between the masked original and the masked paraphrase,
    divided by the longer length (0 = identical, 1 = nothing in common);
  * Drain3 reaction: a parser that has seen the original wording (warm-up on the first lines of the
    log) receives one reworded line. "absorbed" = assigned to the original template (possibly by
    generalising it), "new template" = Drain3 creates a new cluster;
  * coverage: how many lines of TEST BLOCKS each statement's pattern matches, next to the number of
    events of its templates in test blocks according to Drain3 (results/counts.csv). Equal numbers mean
    the pattern catches every eligible test line and nothing else.

Usage:
  python src/check_paraphrases.py --log data/HDFS.log --counts results/counts.csv \
      --out results/drift/paraphrases_review.csv
"""
import argparse
import copy
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdfs_baseline import BLOCK_RE, LINE_RE, build_miner  # noqa: E402
import paraphrases as P  # noqa: E402

MASKS = [(re.compile(r"blk_-?\d+"), "<BLK>"),
         (re.compile(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?"), "<IP>"),
         (re.compile(r"(?<![A-Za-z])-?\d+(?![A-Za-z])"), "<NUM>")]


def mask(text):
    for r, tok in MASKS:
        text = r.sub(tok, text)
    return text


def token_distance(a, b):
    a, b = a.split(), b.split()
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1], prev[-1] / max(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--counts", default="results/counts.csv")
    ap.add_argument("--out", default="results/drift/paraphrases_review.csv")
    ap.add_argument("--warmup-lines", type=int, default=1_500_000)
    ap.add_argument("--test-start-line", type=int, default=9_012_193,
                    help="first parsed line of the test period (from results/leakage_check.json)")
    a = ap.parse_args()
    t0 = time.time()

    # 1. warm-up parser and coverage scan in one pass over the log
    d = pd.read_csv(a.counts)
    te = d.iloc[int(len(d) * 0.8):]
    test_blocks = set(te["BlockId"])
    miner = build_miner()
    example = {}
    covered = {s["id"]: 0 for s in P.STATEMENTS}
    parsed = 0
    with open(a.log, encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = LINE_RE.match(raw.rstrip("\n"))
            if not m:
                continue
            content = m.group(6)
            if not BLOCK_RE.search(content):
                continue
            if parsed < a.warmup_lines:
                miner.add_log_message(content)
            if parsed >= a.test_start_line:
                blocks = set(BLOCK_RE.findall(content))
                if blocks & test_blocks:
                    s, _ = P.match(content)
                    if s is not None:
                        covered[s["id"]] += 1
                        example.setdefault(s["id"], content)
            parsed += 1
    print(f"scanned {parsed:,} lines in {time.time() - t0:.0f}s")

    # events of each statement's templates in test blocks
    events = {s["id"]: int(sum(te[e].sum() for e in s["templates"] if e in te)) for s in P.STATEMENTS}

    rows = []
    for s in P.STATEMENTS:
        ex = example.get(s["id"])
        if ex is None:
            continue
        orig_masked = mask(ex)
        base_id = copy.deepcopy(miner).add_log_message(ex)["cluster_id"]   # template of the original wording
        for lvl in P.LEVELS:
            new = P.reword(ex, lvl)
            dist, norm = token_distance(orig_masked, mask(new))
            m = copy.deepcopy(miner)
            r = m.add_log_message(new)
            absorbed = r["cluster_id"] == base_id
            rows.append({"statement": s["id"], "templates": "+".join(s["templates"]), "level": lvl,
                         "original_example": ex, "reworded_example": new,
                         "token_edit_distance": dist, "magnitude": round(norm, 3),
                         "drain3": ("absorbed (template generalised)" if absorbed and r["change_type"] != "none"
                                    else "absorbed" if absorbed else "new template"),
                         "drain3_template": r["template_mined"],
                         "test_block_lines_matched": covered[s["id"]],
                         "test_block_events": events[s["id"]]})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(out[["statement", "level", "magnitude", "drain3", "reworded_example"]].to_string(index=False))
    print("\nmean magnitude by level:", out.groupby("level")["magnitude"].mean().round(3).to_dict())
    print("new templates by level:", out[out.drain3 == "new template"].groupby("level").size().to_dict())
    cov = out.drop_duplicates("statement")[["statement", "templates", "test_block_lines_matched", "test_block_events"]]
    print("\ncoverage:\n" + cov.to_string(index=False))
    print(f"\nwrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
