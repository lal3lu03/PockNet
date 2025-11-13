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

# PockNet – Selective SWA Epoch09_12

## Model Summary

- **Architecture:** Fusion transformer combining tabular SAS descriptors with ESM2-3B residue embeddings  
- **Checkpoint:** `selective_swa_epoch09_12.ckpt` (SWA blend of epoch 09 baseline and epoch 12 finetune)  
- **Input:** H5 files generated via `generate_h5_v2_optimized.py` (contains tabular + ESM tensors)  
- **Output:** Residue-wise ligandability probabilities + pocket clusters (P2Rank-style CSV/visualisations)  
- **Tasks:** Protein binding-pocket detection / ligandability ranking

## Intended Use & Limitations

| Intended Use | Notes |
|--------------|-------|
| Structure-based binding-pocket detection for academic or non-commercial research | Designed to reproduce and extend P2Rank experiments using BU48 and related datasets |
| Evaluation via the provided `auto-run` / `predict-dataset` orchestration | Ensures calibration, clustering, and reporting match the release scripts |

**Limitations**
- Trained on BU48-style protein chains with solvent-accessible surface sampling; transfer to radically different proteins is unverified.
- Requires pretrained ESM2-3B embeddings; ensure consistent preprocessing (chain-level `.pt` files) for best results.

## Training Data & Procedure

- **Datasets:** BU48 plus auxiliary P2Rank-style splits encoded via `.ds` manifests.
- **Features:** Generated using `src/datagen/extract_protein_features.py` and merged with chain-fixed CSVs.
- **Embeddings:** `generate_esm2_embeddings.py` (ESM2_t36_3B_UR50D) per chain.
- **H5 assembly:** `generate_h5_v2_optimized.py` storing tabular features, embeddings, neighbour tensors, and split labels.
- **Training:** `python src/train.py experiment=fusion_transformer_aggressive_oct17 ...`
- **Checkpoint selection:** SWA blend (50/50) between epoch 09 baseline and epoch 12 finetune, validated on held-out BU48.

## Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Validation AUPRC | ~0.31 | On BU48 validation split |
| Test AUPRC | ~0.445 | Single-seed evaluation on BU48 test split |
| DCA Success@1 | 75% | From P2Rank-like DBSCAN analysis |
| DCC Success@1 | 39% | From P2Rank-like DBSCAN analysis |

Refer to `outputs/pocknet_eval_run*/summary/summary.csv` for the exact values produced by the release pipeline.

## How to Use

### 1. Download with `huggingface_hub`
```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("lal3lu03/PockNet", "selective_swa_epoch09_12.ckpt")
print(ckpt_path)  # local file path
```

### 2. Run the end-to-end pipeline (Docker / local)
```bash
python src/scripts/end_to_end_pipeline.py auto-run data/bu48.ds \
  --checkpoint /path/to/selective_swa_epoch09_12.ckpt \
  --output outputs/bu48_release
```

The command creates all intermediate artefacts (`features/`, `embeddings/`, `h5/`) and writes pockets + metrics under `<output>/predictions`.

### 3. Direct dataset inference
If you already have an H5 + vectors CSV:
```bash
python src/scripts/end_to_end_pipeline.py predict-dataset \
  --checkpoint /path/to/selective_swa_epoch09_12.ckpt \
  --h5 data/h5/all_train_transformer_v2_optimized.h5 \
  --csv data/vectorsTrain_all_chainfix.csv \
  --output outputs/pocknet_eval_cli
```

## Files Included in the Hugging Face Repo

- `selective_swa_epoch09_12.ckpt` – release checkpoint
- `MODEL_CARD.md` – this document
- (Optional) auxiliary scripts / instructions for inference

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
# PockNet – Selective SWA Epoch09_12
---
license: apache-2.0
# PockNet – Selective SWA Epoch09_12
