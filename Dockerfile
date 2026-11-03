FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Data and results are mounted from your computer at run time, not baked into the image
CMD ["python", "src/hdfs_baseline.py", "--log", "data/HDFS.log", "--labels", "data/anomaly_label.csv", "--out", "results"]
