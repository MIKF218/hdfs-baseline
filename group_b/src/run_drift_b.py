#!/usr/bin/env python3
"""
Controlled format-drift experiments, group B (semantic representation, fixed detector), HDFS.

Group B differs from group A (src/run_drift.py) ONLY in how a block is represented:
  group A : Drain3 template IDs, counted per block (unseen templates -> one UNKNOWN feature)
  group B : every log message is masked (block IDs, IPs and numbers with the same rules as Drain3 in
            group A, plus file paths, which Drain3 turns into wildcards), embedded with a frozen
            Sentence-BERT model (all-MiniLM-L6-v2, L2-normalised), and the block vector is the
            TF-IDF-weighted sum of its lines' embeddings (the same weighting group A applies to counts).
            No parser is used.

Messages not seen in training (for example reworded ones) need an IDF weight (--unseen-weight):
  nearest (default, main design): the weight of the most similar training message (cosine similarity),
           i.e. a paraphrase is weighted like the meaning it is closest to;
  max      (robustness check): the largest weight, as for group A's UNKNOWN feature.
Without drift no test block contains an unseen message, so this choice does not change the baseline.

Everything else is shared with group A, by importing group A's code rather than copying it:
  * blocks, labels and the chronological 60/20/20 split come from results/counts.csv (load_counts);
  * scenarios, onset (25%), ramp (50%), gradual seed and paraphrases are the same (scenario_list,
    parse_scenario, src/paraphrases.py), and the drift is injected by the same rule into the same
    lines; the number of reworded lines is checked against group A's results when they exist;
  * the detector is the same: TF-IDF weighting, PCA keeping 95% of the variance, squared prediction
    error, Q-statistic threshold at alpha = 0.001, fitted once on training blocks without labels and
    then frozen.
  * evaluation is the same function (evaluate): overall, by phase, by exposure and per window.
Labels are used only to compute metrics.

Usage (from the project root, so the shared data and group A paths resolve):
  python group_b/src/run_drift_b.py --log data/HDFS.log --counts results/counts.csv
  python group_b/src/run_drift_b.py ... --scenarios prevalence

The first run parses the clean log once (about 2 minutes) and downloads the model once
(set HF_HOME to keep the model cache, e.g. HF_HOME=/workspace/.hf_cache in the container).

Outputs (group_b/results/drift/, or drift_maxweight/ with --unseen-weight max): <scenario>.json,
summary.csv (same columns as group A where they apply), clean_counts_v2.npz + clean_vocab_v2.json
(cache of the clean parse), embeddings_<embedder>.npz (embedding cache).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

# Group A's code lives in <project>/src; group B imports it so both groups share one implementation.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from hdfs_baseline import BLOCK_RE, LINE_RE, build_miner  # noqa: E402
import paraphrases as PARA  # noqa: E402
from run_drift import (ONSET, RAMP, WINDOW_BLOCKS, evaluate, load_counts,  # noqa: E402
                       parse_scenario, scenario_list)

ALPHA = 0.001
VAR_KEPT = 0.95
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PATH_RE = re.compile(r"(?<!\S)/(?!<IP>)\S+")      # file paths (not the "/<IP>" of addresses)


def make_masker():
    """Drain3's masking from group A (block IDs, IPs, numbers), then file paths."""
    m = build_miner().masker
    return lambda content: PATH_RE.sub("<PATH>", m.mask(content))


# ----------------------------------------------------------------------------- embeddings
class Embedder:
    """Frozen sentence embeddings with a cache on disk. Each distinct masked message is embedded once."""

    def __init__(self, backend, model, cache_path):
        self.backend, self.model_name, self.cache_path = backend, model, cache_path
        self.vec = {}
        self._model = None
        if os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=False)
            if str(z["backend"]) == f"{backend}:{model}":
                self.vec = dict(zip(z["texts"].tolist(), z["vectors"]))

    def _encode(self, texts):
        if self.backend == "sbert":
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device="cpu")
            return self._model.encode(texts, batch_size=64, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        # "hash": a deterministic bag-of-words stand-in, ONLY for testing the pipeline without the model
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.split():
                seed = int.from_bytes(hashlib.sha1(w.encode()).digest()[:4], "little")
                out[i] += np.random.default_rng(seed).standard_normal(384).astype(np.float32)
            out[i] /= max(np.linalg.norm(out[i]), 1e-12)
        return out

    def matrix(self, vocab):
        missing = [t for t in vocab if t not in self.vec]
        if missing:
            for t, v in zip(missing, self._encode(missing)):
                self.vec[t] = v
            texts = list(self.vec)
            np.savez(self.cache_path, backend=np.array(f"{self.backend}:{self.model_name}"),
                     texts=np.array(texts), vectors=np.stack([self.vec[t] for t in texts]))
        return np.stack([self.vec[t] for t in vocab]).astype(np.float64)


# ----------------------------------------------------------------------------- clean parse (once)
def clean_parse(log_path, block_index, first_test_block, out_dir):
    """Per-block counts of distinct masked messages over the whole clean log, plus the test-period start."""
    npz, vjs = os.path.join(out_dir, "clean_counts_v2.npz"), os.path.join(out_dir, "clean_vocab_v2.json")
    if os.path.exists(npz) and os.path.exists(vjs):
        meta = json.load(open(vjs))
        print(f"loaded clean parse: {len(meta['vocab'])} distinct messages, test period = {meta['test_lines']:,} lines")
        return sparse.load_npz(npz).tocsr().astype(np.float64), meta
    print("parsing the clean log once (about 2 minutes) ...")
    t0 = time.time()
    mask = make_masker()
    vocab, mask_cache = {}, {}
    rows, cols = [], []
    offset, test_offset, lines, lines_before_test = 0, None, 0, None
    with open(log_path, "rb") as f:
        for raw in f:
            m = LINE_RE.match(raw.decode("utf-8", "replace").rstrip("\r\n"))
            if m:
                content = m.group(6)
                blocks = set(BLOCK_RE.findall(content))
                if blocks:
                    if test_offset is None and first_test_block in blocks:
                        test_offset, lines_before_test = offset, lines
                    j = vocab.setdefault(mask(content), len(vocab))
                    for b in blocks:
                        i = block_index.get(b)
                        if i is not None:
                            rows.append(i)
                            cols.append(j)
                    lines += 1
                    if lines % 1_000_000 == 0:
                        print(f"  {lines:,} lines", flush=True)
            offset += len(raw)
    if test_offset is None:
        sys.exit("first test block not found in the log")
    M = sparse.coo_matrix((np.ones(len(rows), dtype=np.float64), (np.array(rows), np.array(cols))),
                          shape=(len(block_index), len(vocab))).tocsr()
    meta = {"masking": "drain3 + file paths", "vocab": list(vocab), "test_offset": test_offset, "lines_before_test": lines_before_test,
            "test_lines": lines - lines_before_test, "lines_total": lines}
    sparse.save_npz(npz, M)
    json.dump(meta, open(vjs, "w"))
    print(f"clean parse done ({time.time() - t0:.0f}s): {lines:,} lines, {len(vocab)} distinct masked messages, "
          f"{meta['test_lines']:,} in the test period")
    return M, meta


# ----------------------------------------------------------------------------- detector
def fit_pca_q_dense(Mtr, E, alpha=ALPHA, var_kept=VAR_KEPT):
    """PCA + Q-statistic on block vectors X = Mtr @ E, without materialising X (covariance from Mtr)."""
    Mtr = Mtr.astype(np.float64)                     # float64: the covariance below subtracts large numbers
    n = Mtr.shape[0]
    mean_m = np.asarray(Mtr.mean(axis=0)).ravel()
    G = (Mtr.T @ Mtr).toarray()                        # message co-occurrence, V x V
    mu = mean_m @ E
    cov = (E.T @ G @ E - n * np.outer(mu, mu)) / (n - 1)
    lam, vecs = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, vecs = np.clip(lam[order], 0.0, None), vecs[:, order]
    ratio = lam / lam.sum()
    k = int(np.searchsorted(np.cumsum(ratio), var_kept) + 1)
    P = vecs[:, :k]
    resid = lam[k:]
    resid = resid[resid > 1e-12]
    th1, th2, th3 = (float((resid ** i).sum()) for i in (1, 2, 3))
    h0 = 1.0 - 2.0 * th1 * th3 / (3.0 * th2 ** 2)
    c = norm.ppf(1.0 - alpha)
    q = th1 * (c * np.sqrt(2.0 * th2 * h0 ** 2) / th1 + 1.0 + th2 * h0 * (h0 - 1.0) / th1 ** 2) ** (1.0 / h0)

    def spe(X, chunk=20000):
        out = np.empty(len(X))
        for s in range(0, len(X), chunk):
            Z = np.asarray(X[s:s + chunk], dtype=np.float64) - mu
            R = Z - (Z @ P) @ P.T
            out[s:s + chunk] = (R ** 2).sum(axis=1)
        return out

    info = {"n_components_kept": k, "n_features": int(E.shape[1]), "theta1": th1, "theta2": th2,
            "theta3": th3, "h0": float(h0), "alpha": alpha, "q_threshold": float(q)}
    return spe, float(q), info


# ----------------------------------------------------------------------------- TF-IDF weights
def idf_from_training(Mtr):
    """IDF per message over training blocks, the same formula group A applies to template counts."""
    df = np.asarray((Mtr > 0).sum(axis=0)).ravel()
    return np.log((Mtr.shape[0] + 1) / (df + 1)) + 1.0, df > 0


def weights_for(vocab_list, E, idf_train, seen_train, rule):
    """IDF for every message; messages not seen in training get the nearest training message's IDF or the max."""
    w = np.empty(len(vocab_list))
    n_known = len(idf_train)
    w[:n_known] = idf_train
    unseen = [j for j in range(len(vocab_list)) if j >= n_known or not seen_train[j]]
    mapping = {}
    seen_idx = np.flatnonzero(seen_train)
    wmax = float(idf_train.max())
    for j in unseen:
        if rule == "max":
            w[j] = wmax
        else:
            sims = E[seen_idx] @ E[j]
            k = int(seen_idx[int(np.argmax(sims))])
            w[j] = idf_train[k]
            mapping[vocab_list[j]] = {"nearest_training_message": vocab_list[k],
                                      "cosine": round(float(sims.max()), 3), "weight": round(float(w[j]), 3)}
    return w, mapping


# ----------------------------------------------------------------------------- one scenario
def run_scenario(sc, a, meta, test_index, vocab):
    mask = make_masker()
    L = meta["test_lines"]
    onset = int(a.onset * L)
    ramp = max(int(a.ramp * L), 1)
    stmts = set(sc["statements"])
    rng = np.random.default_rng(a.gradual_seed)
    n_test = len(test_index)
    first_pos = np.full(n_test, -1, dtype=np.int64)
    last_pos = np.full(n_test, -1, dtype=np.int64)
    touched = np.zeros(n_test, dtype=bool)
    rows, cols = [], []
    n_clean = len(vocab)
    sha = hashlib.sha1()
    stats = {"lines_parsed": 0, "eligible_lines_after_onset": 0, "lines_reworded": 0}
    pos = 0
    t0 = time.time()
    with open(a.log, "rb") as f:
        f.seek(meta["test_offset"])
        for raw in f:
            m = LINE_RE.match(raw.decode("utf-8", "replace").rstrip("\r\n"))
            if not m:
                continue
            content = m.group(6)
            blocks = set(BLOCK_RE.findall(content))
            if not blocks:
                continue
            reworded = False
            if sc["type"] != "none" and pos >= onset:          # identical rule to group A
                s, params = PARA.match(content)
                if s is not None and s["id"] in stmts:
                    stats["eligible_lines_after_onset"] += 1
                    if sc["type"] == "sudden" or rng.random() < min((pos - onset) / ramp, 1.0):
                        content = s[sc["level"]].format(**params)
                        reworded = True
                        sha.update(f"{pos}:{s['id']};".encode())
            if reworded:
                stats["lines_reworded"] += 1
            j = vocab.setdefault(mask(content), len(vocab))
            for b in blocks:
                i = test_index.get(b)
                if i is not None:
                    rows.append(i)
                    cols.append(j)
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
    stats["new_messages"] = [t for t, j in vocab.items() if j >= n_clean]
    M = sparse.coo_matrix((np.ones(len(rows), dtype=np.float64), (np.array(rows, dtype=np.int64),
                          np.array(cols, dtype=np.int64))), shape=(n_test, len(vocab))).tocsr()
    return M, first_pos, last_pos, touched, stats, onset, ramp


def check_against_a(name, stats, a_dir):
    p = os.path.join(a_dir, f"{name}.json")
    if not os.path.exists(p):
        return None
    pa = json.load(open(p))["parser"]
    same = (pa["lines_reworded"] == stats["lines_reworded"]
            and pa["eligible_lines_after_onset"] == stats["eligible_lines_after_onset"]
            and pa["lines_parsed"] == stats["lines_parsed"])
    if "reworded_lines_sha1" in pa:
        same = same and pa["reworded_lines_sha1"] == stats["reworded_lines_sha1"]
    return same


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--counts", default="results/counts.csv")
    ap.add_argument("--labels", help="not needed; labels are read from --counts (kept for symmetry)")
    ap.add_argument("--out-dir", default=None,
                    help="default group_b/results/drift, or group_b/results/drift_maxweight with --unseen-weight max")
    ap.add_argument("--unseen-weight", choices=("nearest", "max"), default="nearest",
                    help="IDF weight of messages not seen in training (nearest = main design, max = robustness check)")
    ap.add_argument("--group-a-dir", default="results/drift", help="group A results, for the consistency check")
    ap.add_argument("--scenarios", nargs="+", default=["core"])
    ap.add_argument("--embedder", choices=("sbert", "hash"), default="sbert",
                    help="'hash' is a stand-in for testing the pipeline only, never for results")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--onset", type=float, default=ONSET)
    ap.add_argument("--ramp", type=float, default=RAMP)
    ap.add_argument("--window-blocks", type=int, default=WINDOW_BLOCKS)
    ap.add_argument("--gradual-seed", type=int, default=0)
    ap.add_argument("--max-test-lines", type=int, default=0, help="debug only: stop early")
    a = ap.parse_args()
    if a.out_dir is None:
        a.out_dir = "group_b/results/drift" if a.unseen_weight == "nearest" else "group_b/results/drift_maxweight"
    os.makedirs(a.out_dir, exist_ok=True)

    block_ids, _, y_all, _, cut, ts = load_counts(a.counts)          # same blocks, labels and split as A
    block_index = {b: i for i, b in enumerate(block_ids)}
    test_ids = block_ids[ts:]
    test_index = {b: i for i, b in enumerate(test_ids)}
    y = y_all[ts:]

    M_all, meta = clean_parse(a.log, block_index, test_ids[0], a.out_dir)
    emb = Embedder(a.embedder, a.model, os.path.join(a.out_dir, f"embeddings_{a.embedder}.npz"))
    clean_vocab = {t: j for j, t in enumerate(meta["vocab"])}
    E = emb.matrix(meta["vocab"])
    idf_train, seen_train = idf_from_training(M_all[:cut])
    w_clean, _ = weights_for(meta["vocab"], E, idf_train, seen_train, a.unseen_weight)
    spe, q, info = fit_pca_q_dense(M_all[:cut] @ sparse.diags(w_clean), E)
    print(f"model ({a.embedder}, unseen weight = {a.unseen_weight}): {E.shape[1]}-dim embeddings of "
          f"{len(clean_vocab)} messages ({int(seen_train.sum())} seen in training); "
          f"PCA kept {info['n_components_kept']} components, Q = {q:.4f}")

    summary_path = os.path.join(a.out_dir, "summary.csv")
    rows_out = {r["scenario"]: r for r in pd.read_csv(summary_path).to_dict("records")} \
        if os.path.exists(summary_path) else {}
    for name in scenario_list(a.scenarios):
        sc = parse_scenario(name)
        print(f"\n== {name}: statements {sc['statements'] or '-'}")
        vocab = dict(clean_vocab)
        M, first_pos, last_pos, touched, stats, onset, ramp = run_scenario(sc, a, meta, test_index, vocab)
        vlist = list(vocab)
        Et = emb.matrix(vlist)
        w, mapping = weights_for(vlist, Et, idf_train, seen_train, a.unseen_weight)
        stats["unseen_message_weights"] = mapping if a.unseen_weight == "nearest" else \
            {t: float(w[j]) for j, t in enumerate(vlist) if j >= len(idf_train) or not seen_train[j]}
        C = np.asarray((M @ sparse.diags(w)) @ Et)
        res = evaluate(C, first_pos, last_pos, touched, y, spe, q, onset, ramp, sc["type"], a.window_blocks,
                       has_unknown=False)
        same_as_a = check_against_a(name, stats, a.group_a_dir)
        out = {"group": "B", "scenario": sc,
               "settings": {"embedder": a.embedder, "model": a.model if a.embedder == "sbert" else None,
                            "masking": meta.get("masking"),
                            "block_vector": "TF-IDF-weighted sum of L2-normalised message embeddings",
                            "unseen_weight": a.unseen_weight,
                            "onset": a.onset, "ramp": a.ramp, "onset_test_line": onset, "ramp_lines": ramp,
                            "test_period_lines": meta["test_lines"], "gradual_seed": a.gradual_seed,
                            "window_blocks": a.window_blocks, "q_threshold": q,
                            "pca_components": info["n_components_kept"]},
               "drift_matches_group_a": same_as_a, "parser": stats, **res}
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as f:
            json.dump(out, f, indent=2)

        o, ph, ex = res["overall"], res["by_phase"], res["by_exposure"]
        post = ph.get("after_ramp" if sc["type"] == "gradual" else "after_onset", {})
        hit, clean = ex.get("with_reworded_lines", {}), ex.get("without_reworded_lines", {})
        rows_out[name] = {
            "scenario": name, "group": "B", "embedder": a.embedder, "unseen_weight": a.unseen_weight,
            "type": sc["type"], "level": sc["level"],
            "prevalence": sc["prevalence"], "subset_seed": sc["subset_seed"], "statements": " ".join(sc["statements"]),
            "lines_reworded": stats["lines_reworded"], "new_messages": len(stats["new_messages"]),
            "drift_matches_group_a": same_as_a,
            **{f"all_{k}": o.get(k) for k in ("precision", "recall", "f1", "fpr", "pr_auc", "tp", "fp", "fn")},
            "before_onset_fpr": ph.get("before_onset", {}).get("fpr"),
            **{f"after_{k}": post.get(k) for k in ("blocks", "precision", "recall", "f1", "fpr", "pr_auc")},
            "blocks_with_reworded_lines": hit.get("blocks", 0),
            "with_reworded_fpr": hit.get("fpr"), "with_reworded_recall": hit.get("recall"),
            "without_reworded_fpr": clean.get("fpr"), "without_reworded_recall": clean.get("recall")}
        pd.DataFrame(list(rows_out.values())).to_csv(summary_path, index=False)

        print(f"   reworded {stats['lines_reworded']:,} lines; {len(stats['new_messages'])} new messages; "
              f"parse {stats['parse_seconds']}s; same drift as group A: "
              f"{'yes' if same_as_a else 'NO - check!' if same_as_a is False else 'no group A result to compare'}")
        print(f"   ALL   P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f} FPR={o['fpr']:.5f} "
              f"PR-AUC={o.get('pr_auc', float('nan')):.4f} tp={o['tp']} fp={o['fp']} fn={o['fn']}")
        for k, v in list(ph.items()) + list(ex.items()):
            print(f"   {k:22s} blocks={v['blocks']:>7,} P={v['precision']:.4f} R={v['recall']:.4f} "
                  f"F1={v['f1']:.4f} FPR={v['fpr']:.5f} fp={v['fp']}")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
