# Group B: Semantic Representation, Fixed Detector (HDFS)

Group B changes only the representation. Every log message is masked (block IDs, IPs and numbers with
the same rules Drain3 uses in group A, plus file paths, which Drain3 turns into wildcards), embedded with
a frozen Sentence-BERT model (`all-MiniLM-L6-v2`, L2-normalised), and each block is the TF-IDF-weighted
sum of its lines' embeddings (the same weighting group A applies to counts). No parser is used.
Everything else is shared with group A by importing its code from the project's `src/`:

| Kept identical to group A | How |
|---|---|
| Blocks, labels, chronological 60/20/20 split | read from `results/counts.csv` (`load_counts`) |
| Drift scenarios, onset, ramp, gradual seed, paraphrases | same `scenario_list`, `parse_scenario`, `paraphrases.py` and injection rule |
| Reworded lines | checked against group A: same counts and the same SHA-1 fingerprint of reworded lines |
| Detector | TF-IDF weighting + PCA (95% variance) + Q-statistic (alpha = 0.001), fitted on training blocks without labels, frozen |
| Metrics | same `evaluate` function: overall, by phase, by exposure, per window |

**Messages not seen in training** (reworded ones) need a TF-IDF weight. Main design
(`--unseen-weight nearest`, default): the weight of the most similar training message, so a paraphrase is
weighted like the meaning it is closest to. Robustness check (`--unseen-weight max`): the largest weight,
as for group A's UNKNOWN feature. Without drift no test block contains an unseen message, so this choice
does not affect the baseline; it was fixed before any valid drift results existed.

```bash
# 1. go/no-go check first: the no-drift baseline (clean parse about 2 minutes + model download)
python group_b/src/run_drift_b.py --log data/HDFS.log --counts results/counts.csv --scenarios none

# 2. core scenarios (main design)
python group_b/src/run_drift_b.py --log data/HDFS.log --counts results/counts.csv

# 3. robustness check: unseen messages get the maximum weight
python group_b/src/run_drift_b.py --log data/HDFS.log --counts results/counts.csv --unseen-weight max

# 4. prevalence scenarios
python group_b/src/run_drift_b.py --log data/HDFS.log --counts results/counts.csv --scenarios prevalence
```

Outputs go to `group_b/results/drift/` (or `drift_maxweight/`) in the same format as group A. Each scenario prints
`same drift as group A: yes` when group A's result for that scenario exists and matches.
`--embedder hash` is a stand-in for testing the pipeline only and must never be used for results.

Run every command from the project root (`hdfs-baseline/`), inside the same VS Code container as group A,
so the shared paths (`data/`, `results/counts.csv`, `results/drift/`) resolve.

## History

The first run (22 Nov 2026, `results/v1_invalid/`) failed the no-drift check (F1 0.098). Three problems
were fixed: a 32-bit numerical error in the PCA fit, unmasked file paths, and missing TF-IDF weighting.
On the validation split (approximate embeddings for the merged path messages) the corrected design reached
F1 0.829 and PR-AUC 0.826, against group A's 0.871 and 0.859.

## Folder structure

```
group_b/
├── README.md
├── src/
│   └── run_drift_b.py      drift scenarios with Sentence-BERT block vectors + PCA / Q-statistic
└── results/
    ├── drift/              main design: one <scenario>.json per run + summary.csv
    ├── drift_maxweight/    robustness check (unseen messages at maximum weight)
    └── v1_invalid/         first run (22 Nov), failed the baseline check; kept for the record, see its README
```

Shared from the project root: `data/` (dataset and model cache), `results/counts.csv` (blocks, labels,
split), `results/drift/` (group A results, for the consistency check) and `src/` (group A's code,
`paraphrases.py`).
