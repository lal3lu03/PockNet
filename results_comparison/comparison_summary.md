# Random Forest Performance Comparison: With vs Without `protrusion.distanceToCenter`

## Experiment Setup
- **Dataset**: chen11.csv (training) + bu48.csv (testing)
- **Model**: RandomForestClassifier with 200 estimators, max_depth=10
- **Sampling**: SMOTE oversampling for class imbalance
- **Features**: 42 total features (41 without `protrusion.distanceToCenter`)

## Results Summary

### WITH `protrusion.distanceToCenter` Feature (42 features)
| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 0.8587 | 0.8259 | **0.7595** |
| **AUC-ROC** | 0.9397 | 0.8257 | **0.8232** |
| **Avg Precision** | 0.9385 | 0.1012 | **0.0515** |
| **IoU Score** | 0.7584 | 0.0616 | **0.0342** |

### WITHOUT `protrusion.distanceToCenter` Feature (41 features)
| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 0.8497 | 0.7865 | **0.8621** |
| **AUC-ROC** | 0.9355 | 0.7380 | **0.7094** |
| **Avg Precision** | 0.9374 | 0.0566 | **0.0292** |
| **IoU Score** | 0.7509 | 0.0418 | **0.0267** |

## Performance Impact Analysis

### Test Set Performance Changes (Without vs With Feature)
| Metric | With Feature | Without Feature | **Change** | **% Impact** |
|--------|--------------|-----------------|------------|--------------|
| **Accuracy** | 0.7595 | 0.8621 | +0.1026 | **+13.5%** ⬆️ |
| **AUC-ROC** | 0.8232 | 0.7094 | -0.1138 | **-13.8%** ⬇️ |
| **Avg Precision** | 0.0515 | 0.0292 | -0.0223 | **-43.3%** ⬇️ |
| **IoU Score** | 0.0342 | 0.0267 | -0.0075 | **-21.9%** ⬇️ |

## Key Observations

### 1. **Feature Importance Redistribution**
- **With feature**: `protrusion.distanceToCenter` dominates (31.3% importance)
- **Without feature**: Top features become:
  - `chem.aromaticAtoms` (8.99%)
  - `volsite.vsAromatic` (8.34%)
  - `chem.atomO` (6.22%)

### 2. **Class Imbalance Effects**
- **Accuracy Paradox**: Higher accuracy without key feature due to severe class imbalance (98.8% non-binding sites)
- Model becomes more biased toward majority class (non-binding sites)
- **AUC-ROC** is more reliable metric for imbalanced data

### 3. **IoU (Intersection over Union) Insights**
- IoU measures overlap between predicted and actual positive class
- **Low IoU scores** (0.0267-0.0342) indicate poor overlap in binding site prediction
- **21.9% drop** in IoU when removing key feature confirms its importance

### 4. **Feature Criticality**
- `protrusion.distanceToCenter` is **crucial** for binding site identification
- Removing it significantly impacts:
  - Model's discrimination ability (AUC-ROC drop)
  - Precision for positive class (Avg Precision drop)
  - Spatial overlap accuracy (IoU drop)

## Conclusion

The `protrusion.distanceToCenter` feature is **essential** for effective protein binding site prediction. While accuracy improves without it (due to class imbalance bias), all meaningful performance metrics (AUC-ROC, Precision, IoU) deteriorate significantly, confirming its critical role in distinguishing binding from non-binding sites.

**Recommendation**: Always include `protrusion.distanceToCenter` for optimal binding site prediction performance.
