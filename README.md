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
pipeline: `PREPARE_TASK_DATA → FIT_BASELINE_MODEL → SCORE_SUBMISSION`, graded by
the **official Polaris grader** so the objective is leaderboard-comparable.

## Run it

```bash
# local, container-free (needs polaris-lib, rdkit, scikit-learn on PATH)
nextflow run . -profile test --outdir results
cat results/*/result.json     # -> result.objective

# cloud / Stimulus
wave -f Dockerfile --await    # pin the digest into *_container params
uv run stimulus --config stimulus/config.json
```

`result.json.result.objective` is the single number Stimulus maximizes; the search
space in `stimulus/config.json` tunes the fingerprint + estimator. Gradient drives
Stimulus over this surface and proposes model improvements as PRs.
