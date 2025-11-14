---
license: apache-2.0
library_name: pytorch
language:
  - en
tags:
  - protein-pocket-detection
  - esm2
  - binding-site-prediction
---

# PockNet – Fusion Transformer (Selective SWA, multi-seed release)

## Model Summary

- **Architecture:** Fusion transformer combining tabular SAS descriptors with centred ESM2-3B residue embeddings, followed by k-NN attention over local neighbourhoods.  
- **Checkpoint:** `selective_swa_epoch09_12.ckpt` (stochastic weight averaged blend of epochs 20–30).  
- **Evaluation:** Release metrics aggregate **five** independently-seeded SWA runs; per-seed artefacts live under `outputs/final_seed_sweep/`.  
- **Input:** Optimised H5 datasets from `run_h5_generation_optimized.sh` (`tabular`, `esm`, `neighbour` tensors).  
- **Output:** Residue-wise ligandability probabilities plus P2Rank-style pocket CSVs/visualisations.

## Intended Use & Limitations

| Intended Use | Notes |
|--------------|-------|
| Structure-based binding-pocket detection for academic or non-commercial research | Designed to reproduce and extend P2Rank experiments using BU48 and related datasets |
| Evaluation via the provided `auto-run` / `predict-dataset` orchestration | Ensures calibration, clustering, and reporting match the release scripts |

**Limitations**
- Trained on BU48-style protein chains with solvent-accessible surface sampling; transfer to radically different proteins is unverified.
- Requires pretrained ESM2-3B embeddings; ensure consistent preprocessing (chain-level `.pt` files) for best results.

## Training Data & Procedure

- **Datasets:** Training/validation draw from CHEN11 plus the full set of “joint” P2Rank datasets (directories under `data/p2rank-datasets/joined/*`) aggregated in `data/all_train.ds`. BU48 (48 apo/holo pairs) is held out exclusively for evaluation/testing.  
- **Features:** `src/datagen/extract_protein_features.py` (tabular descriptors) + `src/datagen/merge_chainfix_complete.py`.  
- **Embeddings:** `src/tools/generate_esm2_embeddings.py` (ESM2_t36_3B_UR50D).  
- **H5 assembly:** `run_h5_generation_optimized.sh` → `data/h5/all_train_transformer_v2_optimized.h5` with neighbour tensors and split labels.  
- **Training:** Preferred via `python src/scripts/end_to_end_pipeline.py train-model -o experiment=fusion_transformer_aggressive ...`.  
- **Multi-seed sweep:** Seeds `{13, 21, 34, 55, 89}` plus the reference `2025` run; SWA averages checkpoints from epochs 20–30.  
- **Hardware:** 3× NVIDIA V100 (16 GB) for training, single V100 for inference/post-processing.  
- **Logging:** PyTorch Lightning 2.5 + Hydra 1.3, W&B project `fusion_pocknet_thesis`.

## Metrics

### Point-level (single-seed SWA checkpoint)

| Metric | Value | Split |
| --- | --- | --- |
| IoU | 0.2950 | BU48 (test) |
| PR-AUC | 0.414 | BU48 (test) |
| ROC-AUC | 0.944 | BU48 (test) |

### Pocket-level (5-seed aggregated release, DBSCAN post-processing)

| Metric | Mean | 95 % CI | Notes |
| --- | --- | --- | --- |
| Mean IoU | 0.1276 | ±0.0124 | Average pocket IoU across BU48 |
| Best IoU (oracle) | 0.1580 | ±0.0141 | Max IoU per protein |
| GT Coverage | 0.8979 | ±0.0057 | Fraction of GT pockets matched |
| Avg pockets / protein | 6.37 | ±0.87 | Post-threshold pockets |

Success rates (DBSCAN, `eps=3.0`, `min_samples=5`, score threshold 0.91):

- **DCA success@1:** 75 %  
- **DCC success@1:** 39 %  
- **DCA success@3:** 89 %  
- **DCC success@3:** 50 %

Refer to `outputs/final_seed_sweep/*.csv` for the exact release numbers cited by
the thesis (Chapters 5–7 and Appendix 91).

## How to Use

### 1. Download with `huggingface_hub`
```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("lal3lu03/PockNet", "selective_swa_epoch09_12.ckpt")
print(ckpt_path)  # local file path
```

### 2. Run the end-to-end pipeline (CLI / Docker)

Preferred CLI workflow:

```bash
python src/scripts/end_to_end_pipeline.py predict-dataset \
  --checkpoint /path/to/selective_swa_epoch09_12.ckpt \
  --h5 data/h5/all_train_transformer_v2_optimized.h5 \
  --csv data/vectorsTrain_all_chainfix.csv \
  --output outputs/bu48_release
```

Or inside Docker:
```bash
make docker-run ARGS="predict-dataset --checkpoint /ckpts/best.ckpt --h5 /data/h5/all_train_transformer_v2_optimized.h5 --csv /data/vectorsTrain_all_chainfix.csv --output /logs/bu48_release"
```

### 3. Single-protein inference

If you already have an H5 + vectors CSV and want to inspect a single structure:

```bash
python src/scripts/end_to_end_pipeline.py predict-pdb 1a4j_H \
  --checkpoint /path/to/selective_swa_epoch09_12.ckpt \
  --h5 data/h5/all_train_transformer_v2_optimized.h5 \
  --csv data/vectorsTrain_all_chainfix.csv \
  --output outputs/pocknet_single_1a4j
```

## Files Included in the Hugging Face Repo

- `selective_swa_epoch09_12.ckpt` – release checkpoint
- `MODEL_CARD.md` – this document

All supporting scripts (`src/scripts/end_to_end_pipeline.py`, Dockerfile,
data-generation tooling, notebooks) and artefacts (`outputs/final_seed_sweep/*`,
figures, thesis sources) remain in the public GitHub repository:
<https://github.com/hageneder/PockNet>. Refer there for full reproducibility
instructions, figures, and provenance logs.

## Citation

If you use PockNet in your work, please cite:

```
@misc{lal3lu03_pocknet_2025,
  title   = {PockNet Fusion Transformer Release},
  author  = {Hageneder, Max},
  year    = {2025},
  url     = {https://huggingface.co/lal3lu03/PockNet}
}
```

## License

Apache License 2.0. Refer to the repository `LICENSE` for full terms and ensure compliance with upstream dataset/ESM2 licenses when redistributing.
