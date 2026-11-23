# Groups A and B: Drain3 + PCA (Q-statistic) and Sentence-BERT + PCA (Q-statistic) on HDFS
FROM python:3.12-slim

# Print progress immediately and do not write .pyc files into the mounted project.
# HF_HOME keeps the downloaded Sentence-BERT model in data/ (mounted, ignored by git), so it is
# downloaded only once.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/data/.hf_cache

WORKDIR /workspace

# CPU-only PyTorch (group B), installed first from the PyTorch CPU index so pip does not pull CUDA
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY group_b/src/ group_b/src/

# data/ and results/ are mounted from your computer at run time, not baked into the image.
# Default run = the primary pipeline:
#   1. parse data/HDFS.log into results/counts.csv (skipped if counts.csv already exists)
#   2. PCA + Q-statistic detector -> results/results_pca_q.json
CMD ["sh", "-c", "if [ ! -f results/counts.csv ]; then python src/export_counts.py --log data/HDFS.log --labels data/anomaly_label.csv --out results/counts.csv; fi && python src/hdfs_baseline_pca_q.py --counts results/counts.csv --labels data/anomaly_label.csv --out results/results_pca_q.json"]
