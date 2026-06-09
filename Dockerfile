# Runtime image for the caco2-wang drug-discovery baseline.
# Build & push via Wave (recommended), then pin the digest into the
# *_container params (see README). `procps` is required: Nextflow calls `ps`
# for task metrics and Wave/Fusion runs can fail without it.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "polaris-lib==0.13.0" \
    "rdkit==2024.3.5" \
    "scikit-learn==1.5.2" \
    "pandas==2.2.3" \
    "numpy==2.1.3" \
    "scipy==1.14.1"

CMD ["python3"]
