import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    jaccard_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split


def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "iou": jaccard_score(y_true, y_pred),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    
    # Additional metrics for imbalanced data
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics

