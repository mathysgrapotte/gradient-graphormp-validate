# biomlbench-drug-polaris-adme-fang-hclint-1

Minimal, Stimulus-ready Nextflow pipeline for one BioML-bench evaluation:
**`polaris/adme-fang-hclint-1`** — ADME Fang human CLint (drug discovery, regression).

| | |
|---|---|
| Domain | drug discovery / molecular properties |
| Data | Polaris/TDC `polaris/adme-fang-hclint-1` — SMILES -> human hepatic intrinsic clearance |
| Objective | `pearsonr` (higher is better) → `result.objective` |
| Baseline | Morgan/ECFP fingerprints + scikit-learn (regression) |

Part of the [Gradient-evals](../README.md) collection. Same shape as every other
pipeline: `PREPARE_TASK_DATA → FIT_BASELINE_MODEL → SCORE_SUBMISSION`. The dataset
is prepared+pinned once off-pipeline (`scripts/prepare_and_upload_data.py`) to an
immutable, content-addressed S3 prefix; the pipeline only stages+validates it.
The model stage sees `public/{train,test_features,sample_submission}.csv` only;
the held-out answers live on a grader-only channel and are never staged into the
model task, so test labels are unreachable from `train_predict.py` by construction.

## Run it

```bash
# local, container-free (needs rdkit, scikit-learn, scipy, pandas on PATH)
nextflow run . -profile test --outdir results --dataset_s3_prefix /path/to/local/dataset
cat results/*/result.json     # -> result.objective

# cloud / Stimulus
wave -f Dockerfile --await    # pin the digest into *_container params
uv run stimulus --config stimulus/config.json
```

`result.json.result.objective` is the single number Stimulus maximizes; the search
space in `stimulus/config.json` tunes the fingerprint + estimator. Gradient drives
Stimulus over this surface and proposes model improvements as PRs.
