# Group A: Template-Based Fixed Detector (HDFS Baseline)

Part of the master's thesis **Streaming Log Anomaly Detection under Format Drift**
(M.Sc. Big Data Analytics and Management, Bahçeşehir University).

This project implements **method group A**: a template-based pipeline with a fixed (non-updating)
anomaly detector. Raw logs are parsed into event templates with Drain3, each HDFS block becomes an
event-count vector, and PCA flags blocks that deviate from normal behaviour.

**Current stage:** no-drift baseline on HDFS, completed and reproduced (November 2026).
Format-drift experiments come next and must reuse the settings below unchanged.

---

## Headline result (no drift)

Primary detector: **PCA + Q-statistic threshold (α = 0.001), no labels used for training or thresholding.**

| Precision | Recall | F1 | False-positive rate | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| 0.997 | 0.962 | 0.979 (95% CI 0.974–0.985) | 0.00004 | 0.957 | 0.978 |

Test set: final 20% of blocks (115,013 blocks, 1,679 anomalous). 1,616 anomalies caught, 63 missed,
5 false alarms. The result is deterministic and identical for every α from 0.01 to 0.0001.

Comparison detectors (reference only, not thesis methods):

| Detector | Labels used | F1 | PR-AUC |
|---|---|---|---|
| PCA, 99th percentile of training scores | None | 0.707 | 0.957 |
| Isolation Forest, 99th percentile (5 seeds) | None | 0.734 ± 0.093 | 0.754 |
| Logistic regression | Training and threshold | 0.973 | 0.996 |
| Random forest | Training and threshold | 0.998 | 1.000 |

---

## Project structure

```
hdfs-baseline/
├── .devcontainer/
│   └── devcontainer.json             VS Code Dev Container config
├── .dockerignore
├── .gitignore                        ignores data/ and Python cache only
├── Dockerfile                        Python 3.12 image; default command runs the primary pipeline
├── README.md
├── requirements.txt                  drain3, scikit-learn, pandas, numpy (pinned)
├── data/                             HDFS dataset (ignored by git, download from LogHub)
│   ├── .gitkeep
│   ├── HDFS.log
│   └── anomaly_label.csv
├── results/                          all outputs (tracked in git)
│   ├── counts.csv                    block x template count matrix, 575,061 rows (~71 MB)
│   ├── results_pca_q.json            PRIMARY result: PCA + Q-statistic
│   ├── results_v2.json               comparison detectors (Isolation Forest, PCA, supervised)
│   ├── results_v2_pca.json           PCA variants (normal-only, unsupervised percentiles)
│   ├── results.json                  first run: Isolation Forest, 80/20 split
│   ├── templates.csv                 the 46 Drain3 templates and their counts
│   ├── test_scores.csv               first run: anomaly score per test block
│   └── v2_run.log                    run log of hdfs_baseline_v2.py
└── src/
    ├── export_counts.py              step 1: HDFS.log -> results/counts.csv
    ├── hdfs_baseline_pca_q.py        step 2: PRIMARY detector, PCA + Q-statistic
    ├── hdfs_baseline_v2.py           comparison detectors and shared helpers
    └── hdfs_baseline.py              first run (Drain3 + Isolation Forest); parser used by step 1
```

---

## Setup (once)

1. Install **Docker Desktop**, **VS Code** and the VS Code extension **Dev Containers**.
2. In Docker Desktop, give Docker at least **8 GB of memory** (Settings → Resources).
3. Download HDFS_v1 from LogHub:
   https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1
4. Unzip it and copy into `data/`:
   - `HDFS.log` (main folder, about 1.5 GB)
   - `anomaly_label.csv` (inside the `preprocessed` folder)

---

## Running

### In VS Code (recommended)

Open the folder in VS Code, choose **Reopen in Container**, then run in the VS Code terminal:

```bash
# Step 1: parse the log into block count vectors (about 4-5 minutes)
python src/export_counts.py --log data/HDFS.log --labels data/anomaly_label.csv --out results/counts.csv

# Step 2: primary detector, PCA + Q-statistic (under a minute)
python src/hdfs_baseline_pca_q.py --counts results/counts.csv --labels data/anomaly_label.csv --out results/results_pca_q.json
```

Optional comparison runs:

```bash
# Isolation Forest, PCA variants and supervised reference models
python src/hdfs_baseline_v2.py --counts results/counts.csv --labels data/anomaly_label.csv --out results/results_v2.json

# First run: Drain3 + Isolation Forest directly from the raw log
python src/hdfs_baseline.py --log data/HDFS.log --labels data/anomaly_label.csv --out results
```

### With plain Docker

The image's default command runs the primary pipeline: it parses the log into `results/counts.csv`
(skipped if that file already exists), then runs PCA + Q-statistic.

```bash
docker build -t hdfs-baseline .
docker run --rm -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" hdfs-baseline
```

To run any other script, add its command after the image name, for example:

```bash
docker run --rm -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" hdfs-baseline \
  python src/hdfs_baseline_v2.py --counts results/counts.csv --labels data/anomaly_label.csv --out results/results_v2.json
```

To force a fresh parse, delete `results/counts.csv` first.

On Windows PowerShell, use `${PWD}` instead of `$PWD` and put each command on one line.

---

## Protocol and fixed settings

Keep these unchanged in the drift experiments so results stay comparable with this baseline.

| Step | Setting |
|---|---|
| Parsing | Drain3, similarity threshold 0.5, depth 4, max children 100; block IDs, IPs and numbers masked |
| Representation | one count vector per block over the 46 Drain3 templates |
| Order | blocks sorted by first appearance in the log |
| Split | chronological: train = first 60%, validation = next 20% (unused by the primary detector), test = final 20% |
| Detector | PCA on TF-IDF-weighted counts of all training blocks, components explaining 95% of variance (5 of 46 kept) |
| Score | squared prediction error (SPE) in the residual subspace |
| Threshold | Q-statistic (Jackson & Mudholkar, 1979), α = 0.001, from training data only (Q = 7.565) |
| Labels | used only to compute test metrics |
| Metrics | block-level precision, recall, F1, FPR = FP / (FP + TN), PR-AUC, ROC-AUC, bootstrap 95% CI |

---

## Outputs (`results/`)

| File | Contents |
|---|---|
| `counts.csv` | block × template count matrix (`BlockId`, `Label`, `E1`–`E46`), 575,061 rows |
| `results_pca_q.json` | primary result: settings, PCA details, Q threshold, test metrics, confidence intervals |
| `results_v2.json`, `results_v2_pca.json` | comparison detectors under the 60/20/20 split |
| `results.json`, `templates.csv`, `test_scores.csv` | first run (Isolation Forest, 80/20 split) |

---

## Known limitations

- HDFS is an easy dataset for count-based detectors; a high no-drift F1 confirms the setup, it is not a finding.
- The test set contains only 56 distinct count vectors, so confidence intervals are narrow partly because patterns repeat.
- One chronological split; the test period has a lower anomaly rate (1.46%) than training (3.23%).
- The 63 missed anomalies have count vectors close to normal blocks and cannot be separated by counts alone.
- The Q-statistic rule was adopted after percentile thresholds proved unstable, and fixed before its result was computed.

---

## Next: format-drift experiments

1. Semantics-preserving paraphrases of the 46 templates at low, medium and high rewording magnitude, checked by hand.
2. Sudden and gradual drift injection into the replayed log stream.
3. Templates first seen after rewording mapped to one shared "unknown" feature.
4. Same pipeline and settings on the drifted streams, compared with this baseline (F1 and PR-AUC).

Groups B (semantic, fixed) and C (semantic, online and unsupervised) will be separate projects.

---

## References

- He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). Drain: An online log parsing approach with fixed depth tree. *IEEE ICWS 2017*. https://doi.org/10.1109/ICWS.2017.13
- Jackson, J. E., & Mudholkar, G. S. (1979). Control procedures for residuals associated with principal component analysis. *Technometrics*, 21(3), 341–349.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *IEEE ICDM 2008*.
- Xu, W., Huang, L., Fox, A., Patterson, D., & Jordan, M. I. (2009). Detecting large-scale system problems by mining console logs. *ACM SOSP 2009*.
- Zhu, J., He, S., He, P., Liu, J., & Lyu, M. R. (2023). Loghub: A large collection of system log datasets for AI-driven log analytics. *IEEE ISSRE 2023*. https://arxiv.org/abs/2008.06448
