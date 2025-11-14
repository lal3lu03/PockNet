# PockNet Fusion Pipeline

![PockNet Logo](Logo.png)

This repository contains the current production pipeline used to train and
evaluate the **PockNet** fusion model.  
The active workflow combines per–surface-point tabular descriptors with ESM2
transformer embeddings and trains a Lightning module that learns attention over
structure-aware neighbour sets stored in an optimized H5 file.

The repository has been trimmed to focus on this transformer‑centric workflow.
Legacy experiments, notebooks, and random-forest/tabnet variants live under
`deprecated/` for reference.

---

## Highlights

- **Single source of truth for data** – feature CSVs and merged chain-fix tables live in `data/`.
- **Optimized H5 generation** – `generate_h5_v2_optimized.py` writes compact
  transformer-ready datasets with neighbour indices and distance metadata.
- **Toggleable k-NN aggregation** – `configs/data/h5_knn_enabled.yaml`
  configures on-the-fly k-NN aggregation; transformer mode is the default.
- **Hydra + Lightning** – experiments are reproducible and parameterised via
  `configs/`, with launch scripts that set up the environment automatically.
- **Reproducible release** – `Dockerfile`, `REPRODUCIBILITY.md`, and `docs/src_variable_renames.csv`
  pin the CUDA/PyTorch image, dataset digests, and source renames for auditing.

---

## Prerequisites

1. **Conda** (Miniconda or Anaconda) and CUDA 12.1 capable drivers.
2. **Git LFS** if you intend to version large binaries (optional).
3. An NVIDIA GPU for training/running the transformer model.

> **Important**  
> The training scripts expect the environment variable
> `PROJECT_ROOT` to point at the repository root.  
> You can set it once per session:
> ```bash
> export PROJECT_ROOT=$(pwd)
> ```

---

## Environment Setup

Create the main environment (named `pocknet_env`) and activate it:

```bash
conda env create -f environment.yaml
conda activate pocknet_env
```

The environment installs PyTorch 2.4 with CUDA 12.1 support, Lightning 2.5,
Hydra tooling, Biopython/DSSP for structural features, and the ESM2 model via
pip (`fair-esm`).

If you require a different CUDA toolkit version, adjust the `pytorch-cuda`
entry in `environment.yaml` accordingly or install PyTorch by following the
official instructions before running `pip install -r` for the remaining
packages.

---

## Docker Image (Reproducible Release)

The repository ships a pinned Docker build that captures the CUDA 12.4.1 +
CUDNN 9 runtime, PyTorch 2.6 wheels, and all Python dependencies declared in
`requirements.txt`. Build the image directly from the repository root:

```bash
docker build -t pocknet:cuda12.4 .
```

Run training or evaluation from a clean machine by mounting your datasets,
checkpoints, and logs into the container and propagating the corresponding
environment variables:

```bash
docker run --rm --gpus all \
  -v /abs/path/to/data:/workspace/data \
  -v /abs/path/to/checkpoints:/workspace/checkpoints \
  -v /abs/path/to/logs:/workspace/logs \
  -e POCKNET_DATA_ROOT=/workspace/data \
  -e POCKNET_CHECKPOINT_ROOT=/workspace/checkpoints \
  -e POCKNET_LOG_ROOT=/workspace/logs \
  pocknet:cuda12.4 \
  python src/train.py experiment=fusion_transformer_aggressive trainer.devices=2
```

The entrypoint defaults to Bash, so you can replace the final command with any
tooling (e.g., `python src/eval.py ...` or `python -m post_processing.run_production_pipeline ...`).
The `REPRODUCIBILITY.md` ledger records the exact commit hash, dataset digests,
and seed files tied to the image to simplify long-term archiving.

You can also rely on the helper targets defined in the `Makefile`:

```bash
make docker-build                     # builds pocknet:cuda12.4
make docker-run ARGS="--help"         # shows the CLI help inside the container
make docker-run ARGS="train-model -o trainer.fast_dev_run=true"
make docker-full-run                  # fast-dev full-run using the pinned CLI
```

---

## Repository Layout

```
├── README.md                     ← the document you’re reading
├── environment.yaml              ← conda environment (pocknet_env)
├── pyproject.toml / setup.py     ← packaging metadata
├── Dockerfile                  ← CUDA 12.4.1 image with pinned PyTorch deps
├── launch_aggressive_training.sh ← main Lightning launcher
├── launch_aggressive_swa.sh      ← SWA follow-up launcher
├── run_h5_generation_optimized.sh← optimized H5 builder entrypoint
├── configs/                      ← Hydra config tree (data/experiment/callbacks…)
├── src/                          ← datagen, datamodules, models, tools, train/eval
├── src/tests/                    ← current regression test for enhanced pipeline
├── post_processing/              ← legacy pipeline (under active refresh)
├── splits/                       ← curated protein ID lists (e.g., BU48)
├── CLEANUP_SUMMARY.md            ← change log for the cleanup pass
├── MODEL_CHANGELOG_OCT2025.md    ← high-level model change notes
├── REPRODUCIBILITY.md            ← frozen commit/dataset/seed ledger
├── docs/src_variable_renames.csv ← audit of renamed P2Rank-era symbols
└── seeds/master_seeds.yaml       ← canonical RNG seeds used by Hydra configs

(ignored locally but expected when running the pipeline)
├── data/                         ← generated features, embeddings, H5 files
├── logs/                         ← Lightning/Hydra run outputs
└── deprecated/                   ← archived notebooks & experiments
```

---

## Data Preparation Pipeline

All experiments share the same manifests:

- **Training/validation:** CHEN11 plus the complete “joint” dataset released with P2Rank (directories under `data/p2rank-datasets/joined/*`). The IDs are merged into `data/all_train.ds`, which `generate_esm2_embeddings.py` and `run_h5_generation_optimized.sh` read by default.
- **Testing:** BU48 (48 apo/holo pairs) remains untouched during training and is only used for the final evaluations / seed sweeps.

1. **ESM2 embeddings** (per chain):  
   ```bash
   python src/tools/generate_esm2_embeddings.py \
       --ds-file data/all_train.ds \
       --pdb-base data/p2rank-datasets \
       --out-dir data/esm2_3B_chain
   ```
   You can pass multiple `--ds-file` arguments or restrict to specific protein
   lists (`--only-missing data/missing_esm_pdb_ids.txt`).

2. **Surface feature extraction** (optional unless you need to regenerate
   `data/output_train/`):  
   ```bash
   python src/datagen/extract_protein_features.py \
       data/all_train.ds data/output_train
   ```

3. **Chain fix merge** (generates the canonical CSV used by the H5 builder):  
   ```bash
   python src/datagen/merge_chainfix_complete.py
   ```
   The script reads `data/output_train/*.pdb_features.csv` and produces
   `data/vectorsTrain_all_chainfix.csv` (overwriting after creating a timestamped
   backup).

> **Reference**  
> The physicochemical descriptor set and SAS labelling scheme mirror those
> introduced by **P2Rank** (Krivák & Hoksza, 2018). Please cite their work if
> you reuse the feature pipeline:  
> `Krivák, R., & Hoksza, D. (2018). P2Rank: machine learning based tool for rapid`
> `and accurate prediction of ligand binding sites from protein structure.`
> `Journal of Cheminformatics, 10(1), 39.`

---

## H5 Generation

Run the optimized generator to write the transformer-ready dataset:

```bash
bash run_h5_generation_optimized.sh
```

The script validates inputs and writes
`data/h5/all_train_transformer_v2_optimized.h5`, including neighbour indices and
distance metadata for transformer aggregation. Logs appear in
`h5_generation_optimized.log`.

---

## Training

### CLI-first workflow

All training/evaluation orchestration now flows through the Click CLI:

```bash
python src/scripts/end_to_end_pipeline.py train-model \
  --summary outputs/train_summary.json \
  -o experiment=fusion_transformer_aggressive \
  -o trainer.devices=2
```

The command mirrors the methodology chapter in the thesis and writes a JSON
summary containing the resolved Hydra config, the best checkpoint path, and all
Lightning metrics. Combine it with the Docker targets for fully reproducible
runs (see the examples in `REPRODUCIBILITY.md` and
`tex/master_thesis/91-appendix.tex`).

### Legacy launchers

```bash
export PROJECT_ROOT=$(pwd)
bash launch_aggressive_training.sh
```

This starts DDP training across GPUs `1,3,4` in a tmux session named
`transformer_training`.  Logs stream to
`training_aggressive_<timestamp>.log` and metrics are pushed to W&B (project
`fusion_pocknet_thesis`) if you have credentials configured.

### SWA finetuning

To run the stochastic weight averaging schedule after base training finishes,
launch:

```bash
export PROJECT_ROOT=$(pwd)
bash launch_aggressive_swa.sh
```

### Running manually

You can also invoke Hydra directly:

```bash
export PROJECT_ROOT=$(pwd)
python src/train.py experiment=fusion_transformer_aggressive trainer.devices=2
```

The second maintained experiment, `fusion_all_train_complete`, uses mean pooled
ESM embeddings with optional on-the-fly k-NN aggregation and can be selected in
the same manner.

---

## Evaluation

Hydra uses the new transformer defaults, so evaluation is straightforward:

```bash
export PROJECT_ROOT=$(pwd)
python src/eval.py \
    experiment=fusion_transformer_aggressive \
    ckpt_path=/path/to/checkpoint.ckpt
```

CSV logs for evaluation runs are emitted under `logs/<task>/runs/...` alongside
Hydra configuration snapshots.

## End-to-End CLI

`src/scripts/end_to_end_pipeline.py` exposes a Click-powered CLI that mirrors the
standalone train/eval/post-processing entry points so you can orchestrate
complete workflows with a single command.

| Command | Purpose | Example |
| --- | --- | --- |
| `train-model` | Launch Hydra/Lightning training and store a JSON summary. | `python src/scripts/end_to_end_pipeline.py train-model --summary outputs/train_summary.json -o trainer.fast_dev_run=true` |
| `predict-dataset` | Run checkpoint + pocket aggregation over an H5 dataset. | `python src/scripts/end_to_end_pipeline.py predict-dataset --checkpoint ckpts/best.ckpt --h5 data/h5/all_train_transformer_v2_optimized.h5 --csv data/vectorsTrain_all_chainfix.csv --output outputs/pocknet_eval_cli --max-proteins 2` |
| `predict-pdb` | Produce pockets for a single protein or local PDB file. | `python src/scripts/end_to_end_pipeline.py predict-pdb 1a4j_H --checkpoint ckpts/best.ckpt --h5 data/h5/all_train_transformer_v2_optimized.h5 --csv data/vectorsTrain_all_chainfix.csv --output outputs/pocknet_single` |
| `full-run` | (Optionally) train and immediately execute production inference. | `python src/scripts/end_to_end_pipeline.py full-run --h5 data/h5/all_train_transformer_v2_optimized.h5 --csv data/vectorsTrain_all_chainfix.csv --output outputs/release_candidate -o trainer.fast_dev_run=true` |

> Tip  
> Add overrides such as `-o trainer.fast_dev_run=true` or
> `-o data.limit_train_batches=0.02` for quick smoke tests in CI/Docker.

These commands mirror the methodology released with the thesis (see
`tex/master_thesis/03-methodology.tex` for the workflow narrative and
`tex/master_thesis/91-appendix.tex` for the reproducibility ledger details).

**End-to-end recipe:**  
1. `train-model` — trains on the full CHEN11 + joint dataset (as listed in `data/all_train.ds`) and logs the checkpoint path plus metrics to JSON.  
2. `predict-dataset --split test --h5 data/h5/all_train_transformer_v2_optimized.h5 --checkpoint <best.ckpt>` — runs the release post-processing on BU48 only, writing summaries to `outputs/final_seed_sweep/`.  
3. `predict-pdb <pdb_or_id>` — sanity-check a specific protein or local PDB file (copies the file into the output dir and writes `pockets.csv` + PyMOL script).  
4. `full-run` — combines training + evaluation when you want a single command for CI/Docker smoke tests (use overrides to limit batch counts if desired).

### Release metrics (5-seed sweep)

The final BU48 evaluation aggregates **five** independently-seeded SWA runs,
captures full pocket-level post-processing, and stores all artefacts under
`outputs/final_seed_sweep/`:

| Metric | Mean | 95% CI | Source |
| --- | --- | --- | --- |
| Mean IoU (pocket-level) | 0.1276 | ±0.0124 | `final_seed_sweep/final_ensemble_summary.csv` |
| Best IoU (oracle pocket) | 0.1580 | ±0.0141 | `final_seed_sweep/final_ensemble_summary.csv` |
| GT coverage | 0.8979 | ±0.0057 | `final_seed_sweep/final_ensemble_summary.csv` |
| Avg pockets / protein | 6.37 | ±0.87 | `final_seed_sweep/final_ensemble_summary.csv` |
| Threshold sweep CSV | – | – | `final_seed_sweep/threshold_sweep_aggregated.csv` |

Per-protein means/standard deviations (for IoU, coverage, pocket counts) are
available in `final_seed_sweep/protein_aggregated_metrics.csv`, which the thesis
uses for the qualitative case studies and appendix tables.

### Current Benchmark Numbers

All metrics are reported on the BU48 test split (48 apo structures) using the
recreated solvent-accessible surface pipeline:

| Stage | Model Variant | IoU | PR–AUC | Notes |
| --- | --- | --- | --- | --- |
| I | TabNet (tabular descriptors) | **0.1498** | 0.231 | Reimplementation of the hand-crafted P2Rank descriptor baseline |
| II | TabNet + centred ESM2 | 0.1710 | 0.262 | Adds residue language-model context (ESM2-t36-3B) |
| III | Transformer + kNN (epoch 14) | 0.2780 | **0.424** | Best single checkpoint before SWA |
| III | Transformer + kNN + SWA | **0.2950** | 0.414 | Single-seed SWA checkpoint (epoch window 20–30); see release metrics above for the 5-seed pocket analysis |

Pocket-level clustering (DBSCAN, `eps=3.0`, `min_samples=5`, score threshold
`0.91`) yields the following success rates (single-best prediction = Top‑1):

- DCC success@1: 39 %
- DCA success@1: 75 %
- DCC success@3: 50 %
- DCA success@3: 89 %

Full per-protein summaries and qualitative case studies are stored under
`outputs/pocknet_eval_run_test/` with the aggregated CSV/HTML exports consumed
by `tex/master_thesis/05-results.tex` and the figure notebooks.

---

## Post-processing

The production-ready post-processing stage is implemented in
`post_processing/pocketnet_aggregation.py` (see `docs/REFERENCES_P2RANK.md` for
full provenance) and is exercised via
`post_processing/run_production_pipeline.py`—the same entry point the CLI
invokes. DBSCAN success metrics, pocket CSVs, PyMOL renderings, and case-study
panels are written under `outputs/<run>/` and referenced throughout the thesis.
The older Groovy-style scripts remain archived in `deprecated/` for historical
comparisons.

---

## Troubleshooting

- **Hydra override errors:** override syntax is strict.  Refer to
  <https://hydra.cc/docs/advanced/override_grammar/basic> if you see
  `mismatched input '=' expecting <EOF>`.
- **Missing H5 or embeddings:** ensure `data/esm2_3B_chain/`,
  `data/output_train/`, and `data/vectorsTrain_all_chainfix.csv` exist before
  running the H5 generator.
- **Shared memory errors:** the datamodule writes to `/dev/shm/pocknet` by
  default.  If `/dev/shm` is unavailable, set `POCKNET_SHM_DIR` to a writable
  tmpfs path before launching training.
- **W&B offline:** set `WANDB_MODE=offline` or edit `configs/logger/wandb.yaml`
  if you prefer not to push metrics.

---

## Reproducibility Ledger

- `REPRODUCIBILITY.md` captures the frozen commit (`7972cbe8066d`), dataset
  SHA-256 digests, the Docker/Makefile workflow, and the CLI invocations mirrored
  by Appendix~\ref{app:release-ledger}.
- `seeds/master_seeds.yaml` stores the canonical RNG seeds used across Hydra
  configs (global/train/finetune) so split regeneration stays deterministic.
- `docs/src_variable_renames.csv` enumerates every P2Rank-era symbol that was
  renamed inside `src/` and `post_processing/` (e.g., `P2RankParams →
  PocketAggregationParams`) so the thesis narrative and the cleaned codebase
  reference the same concepts.
- `docs/REFERENCES_P2RANK.md` carries the textual provenance statement cited in
  `tex/master_thesis/03-methodology.tex` and Appendix~\ref{app:release-ledger},
  ensuring the post-processing stage keeps the original credit trail intact.

---

## Contributing / Roadmap

1. Flesh out the refreshed post-processing pipeline (with deterministic tests).
2. Automate end-to-end retraining via a single CLI entrypoint.
3. Rename the model and configs to the final PockNet naming once training wraps.

Pull requests are welcome—please place experimental work under `deprecated/` or
open an issue if you plan a larger refactor.

---

© 2025 PockNet Project – maintained for the fusion transformer pipeline.
