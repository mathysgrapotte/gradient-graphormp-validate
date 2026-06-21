# biomlbench-drug-polaris-adme-fang-hclint-1

Minimal, Stimulus-ready Nextflow pipeline for one BioML-bench evaluation:
**`polaris/adme-fang-hclint-1`** — ADME Fang human CLint (drug discovery, regression).

| | |
|---|---|
| Domain | drug discovery / molecular properties |
| Data | Polaris/TDC `polaris/adme-fang-hclint-1` — SMILES -> human hepatic intrinsic clearance |
| Objective | `pearsonr` (higher is better) → `result.objective` |
| Baseline | Morgan/ECFP fingerprints + scikit-learn (regression) |
| Orthogonal arm | `graphormer`: PCQM4Mv2-pretrained Graphormer fine-tune (`bin/finetune_graphormer.py`) |

Part of the [Gradient-evals](../README.md) collection. Same shape as every other
pipeline: `PREPARE_TASK_DATA → FIT_BASELINE_MODEL → SCORE_SUBMISSION`. The dataset
is prepared+pinned once off-pipeline (`scripts/prepare_and_upload_data.py`) to an
immutable, content-addressed S3 prefix; the pipeline only stages+validates it.
The model stage sees `public/{train,test_features,sample_submission}.csv` only;
the held-out answers live on a grader-only channel and are never staged into the
model task, so test labels are unreachable from `train_predict.py` by construction.

## Run it

```bash
# local, container-free (classical baseline; needs rdkit, scikit-learn, scipy, pandas on PATH)
nextflow run . -profile test --outdir results --dataset_s3_prefix /path/to/local/dataset
cat results/*/result.json     # -> result.objective

# build the durable trial image (frozen to the public Seqera community registry),
# with the PCQM4Mv2 checkpoint + compiled Graphormer Cython extension baked in:
wave --conda-file conda/environment.yml --platform linux/amd64 --freeze --await \
  --config-env GRAPHORMER_CKPT_DIR=/opt/graphormer-pcqm4mv2 \
  --conda-run-command 'RUN HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('"'"'clefourrier/graphormer-base-pcqm4mv2'"'"', local_dir='"'"'/opt/graphormer-pcqm4mv2'"'"')"' \
  --conda-run-command 'RUN python -c "import os,glob,numpy,subprocess,transformers; b=os.path.join(os.environ['"'"'MAMBA_ROOT_PREFIX'"'"'],'"'"'bin'"'"'); cc=sorted(glob.glob(b+'"'"'/*-linux-gnu-gcc'"'"'))[0]; os.environ['"'"'CC'"'"']=cc; os.environ['"'"'CFLAGS'"'"']='"'"'-I'"'"'+numpy.get_include(); p=os.path.dirname(transformers.__file__)+'"'"'/models/graphormer'"'"'; subprocess.check_call(['"'"'cythonize'"'"','"'"'-i'"'"',p+'"'"'/algos_graphormer.pyx'"'"'])"'
# -> community.wave.seqera.io/library/hclint-graphormp:<hash>; pin into the *_container params

# cloud / Stimulus (graphormer arm)
uv run stimulus --config stimulus/config.json
```

The trial image is a single durable `community.wave.seqera.io/library/hclint-graphormp:<hash>`
image (built from `conda/environment.yml`) used by all stages: torch + transformers + rdkit +
sklearn, with the PCQM4Mv2 Graphormer checkpoint and the compiled Graphormer spatial-encoding
extension baked in, so trials do no network I/O.

`result.json.result.objective` is the single number Stimulus maximizes. The search
space in `stimulus/config.json` tunes the Graphormer fine-tune surface (regime, head,
lr, epochs, freezing, ensemble, descriptor fusion). Gradient drives Stimulus over this
surface and proposes model improvements as PRs.
