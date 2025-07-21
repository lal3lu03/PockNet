import argparse
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import wandb
from pathlib import Path

from model_utils import stratified_split, compute_metrics


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_feature_order(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main(cfg_path, run_dummy=False):
    cfg = load_config(cfg_path)
    base = Path(cfg_path).parent

    feature_path = (base / cfg["features_file"]).resolve()
    data_path = (base / cfg["train_csv"]).resolve()
    finetune_path = (base / cfg.get("finetune_csv", "")).resolve() if cfg.get("finetune_csv") else None
    test_path = (base / cfg.get("test_csv", "")).resolve() if cfg.get("test_csv") else None

    feature_order = load_feature_order(feature_path)
    print(f"Expected features: {len(feature_order)} features")
    print(f"First 5 features: {feature_order[:5]}")
    print(f"Last 5 features: {feature_order[-5:]}")

    data = pd.read_csv(data_path, low_memory=False)
    print(f"Training dataset shape: {data.shape}")
    
    # Handle mixed types - convert chain_id to string
    if 'chain_id' in data.columns:
        data['chain_id'] = data['chain_id'].astype(str)
    
    # Debug class column for training data
    if 'class' in data.columns:
        print(f"Training 'class' column info:")
        print(f"  - Data type: {data['class'].dtype}")
        print(f"  - Unique values: {data['class'].unique()}")
        print(f"  - NaN count: {data['class'].isna().sum()}")
        print(f"  - Value counts: {data['class'].value_counts(dropna=False)}")
        
        # Handle NaN values
        if data['class'].isna().any():
            print("WARNING: Found NaN values in training 'class' column. Dropping rows with NaN class.")
            data = data.dropna(subset=['class'])
            print(f"After dropping NaN: shape = {data.shape}")
    else:
        print("ERROR: No 'class' column found in training data!")
    
    X = data[feature_order]
    y = data["class"].astype(int)
    
    # Debug feature columns
    missing_features = [f for f in feature_order if f not in data.columns]
    if missing_features:
        print(f"WARNING: Missing features in training data: {missing_features[:10]}...")
        print(f"Total missing features: {len(missing_features)}")
    
    print(f"Successfully loaded training data: X shape = {X.shape}, y shape = {y.shape}")
    
    # Print class distribution for imbalanced data analysis
    class_counts = y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    print(f"Class 1 percentage: {class_counts[1] / len(y) * 100:.2f}%")

    # Log class distribution to wandb
    class_distribution = {
        "total_samples": len(y),
        "class_0_count": int(class_counts[0]),
        "class_1_count": int(class_counts[1]),
        "class_1_percentage": class_counts[1] / len(y) * 100
    }

    finetune_data = None
    if finetune_path and finetune_path.exists():
        print(f"Loading finetuning data from: {finetune_path}")
        finetune_df = pd.read_csv(finetune_path, low_memory=False)
        print(f"Finetuning dataset shape: {finetune_df.shape}")
        
        # Debug class column
        if 'class' in finetune_df.columns:
            print(f"Finetuning 'class' column info:")
            print(f"  - Data type: {finetune_df['class'].dtype}")
            print(f"  - Unique values: {finetune_df['class'].unique()}")
            print(f"  - NaN count: {finetune_df['class'].isna().sum()}")
            print(f"  - Value counts: {finetune_df['class'].value_counts(dropna=False)}")
            
            # Handle NaN values
            if finetune_df['class'].isna().any():
                print("WARNING: Found NaN values in finetuning 'class' column. Dropping rows with NaN class.")
                finetune_df = finetune_df.dropna(subset=['class'])
                print(f"After dropping NaN: shape = {finetune_df.shape}")
            
            finetune_data = (finetune_df[feature_order], finetune_df["class"].astype(int))
        else:
            print("ERROR: No 'class' column found in finetuning data!")

    test_data = None
    if test_path and test_path.exists():
        print(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path, low_memory=False)
        print(f"Test dataset shape: {test_df.shape}")
        
        # Debug class column
        if 'class' in test_df.columns:
            print(f"Test 'class' column info:")
            print(f"  - Data type: {test_df['class'].dtype}")
            print(f"  - Unique values: {test_df['class'].unique()}")
            print(f"  - NaN count: {test_df['class'].isna().sum()}")
            print(f"  - Value counts: {test_df['class'].value_counts(dropna=False)}")
            
            # Handle NaN values
            if test_df['class'].isna().any():
                print("WARNING: Found NaN values in test 'class' column. Dropping rows with NaN class.")
                test_df = test_df.dropna(subset=['class'])
                print(f"After dropping NaN: shape = {test_df.shape}")
            
            test_data = (test_df[feature_order], test_df["class"].astype(int))
        else:
            print("ERROR: No 'class' column found in test data!")

    X_train, X_val, y_train, y_val = stratified_split(
        X, y, test_size=cfg.get("validation_size", 0.2), random_state=cfg.get("seed", 42)
    )

    # Set up class weighting based on config
    class_weight_setting = cfg.get("class_weight", "balanced")
    if class_weight_setting == "null" or class_weight_setting is None:
        class_weight = None
    elif class_weight_setting == "balanced":
        # Compute class weights manually for warm_start compatibility
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, class_weights))
        print(f"Computed class weights: {class_weight}")
    else:
        class_weight = class_weight_setting

    rf_params = {
        "n_estimators": cfg.get("rf_trees", 200),
        "max_depth": None if cfg.get("rf_depth", 0) == 0 else cfg.get("rf_depth"),
        "oob_score": True,
        "max_features": 6 if cfg.get("rf_features", 0) == 0 else cfg.get("rf_features"),
        "n_jobs": -1 if cfg.get("rf_threads", 0) == 0 else cfg.get("rf_threads"),
        "random_state": cfg.get("seed", 42),
        "bootstrap": True,
        "class_weight": class_weight,  # Handle imbalanced data
    }
    if cfg.get("rf_bagsize", 100) != 100:
        rf_params["max_samples"] = cfg["rf_bagsize"] / 100.0

    wandb.init(project="p2rank_rf", config=rf_params, reinit=True, mode="online")
    
    # Log class distribution and dataset info
    wandb.log(class_distribution)

    # Phase 1: Initial training
    print("=== Phase 1: Initial Training ===")
    model = RandomForestClassifier(**rf_params, warm_start=True)
    model.fit(X_train, y_train)
    
    # Evaluate after initial training
    print("Evaluating after initial training...")
    y_pred_val = model.predict(X_val)
    y_prob_val = model.predict_proba(X_val)[:, 1]
    initial_metrics = compute_metrics(y_val, y_pred_val, y_prob_val)
    
    print("Initial Training - Validation Metrics:")
    for k, v in initial_metrics.items():
        if k == "confusion_matrix":
            print(f"{k}:\n{v}")
        else:
            print(f"{k}: {v:.4f}")
    
    print(f"Initial model has {model.n_estimators} trees")
    
    # Log initial training metrics and model info to wandb
    initial_wandb_metrics = {f"initial_{k}": v for k, v in initial_metrics.items() if k != "confusion_matrix"}
    initial_wandb_metrics["initial_model_trees"] = model.n_estimators
    wandb.log(initial_wandb_metrics)
    wandb.log({"initial_confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_val.tolist(),
        preds=y_pred_val.tolist(),
        class_names=["0", "1"],
    )})
    
    # Phase 2: True Incremental Finetuning (if data available)
    if finetune_data is not None:
        print("\n=== Phase 2: Incremental Finetuning ===")
        X_fine, y_fine = finetune_data
        
        # Print finetuning data stats
        fine_class_counts = y_fine.value_counts()
        print(f"Finetuning data class distribution: {fine_class_counts.to_dict()}")
        print(f"Finetuning Class 1 percentage: {fine_class_counts[1] / len(y_fine) * 100:.2f}%")
        
        # Log finetuning data distribution
        finetune_distribution = {
            "finetune_total_samples": len(y_fine),
            "finetune_class_0_count": int(fine_class_counts[0]),
            "finetune_class_1_count": int(fine_class_counts[1]),
            "finetune_class_1_percentage": fine_class_counts[1] / len(y_fine) * 100
        }
        wandb.log(finetune_distribution)
        
        # Add more trees trained on finetuning data (incremental learning)
        additional_trees = cfg.get("finetune_trees", 0)
        if additional_trees == 0:
            additional_trees = cfg.get("rf_trees", 200)  # Default: same as initial training
        
        # Compute class weights for finetuning data if needed
        if class_weight_setting == "balanced":
            fine_classes = np.unique(y_fine)
            fine_class_weights = compute_class_weight('balanced', classes=fine_classes, y=y_fine)
            fine_class_weight = dict(zip(fine_classes, fine_class_weights))
            print(f"Computed finetuning class weights: {fine_class_weight}")
            model.set_params(n_estimators=model.n_estimators + additional_trees, class_weight=fine_class_weight)
        else:
            model.set_params(n_estimators=model.n_estimators + additional_trees)
        
        print(f"Adding {additional_trees} new trees (total will be {model.n_estimators})")
        print("Training new trees on finetuning data...")
        
        # Fit additional trees on finetuning data only
        model.fit(X_fine, y_fine)
        
        # Evaluate after incremental finetuning
        print("Evaluating after incremental finetuning...")
        y_pred_val_fine = model.predict(X_val)
        y_prob_val_fine = model.predict_proba(X_val)[:, 1]
        finetuned_metrics = compute_metrics(y_val, y_pred_val_fine, y_prob_val_fine)
        
        print(f"Final model has {model.n_estimators} trees")
        print("After Incremental Finetuning - Validation Metrics:")
        for k, v in finetuned_metrics.items():
            if k == "confusion_matrix":
                print(f"{k}:\n{v}")
            else:
                print(f"{k}: {v:.4f}")
        
        # Compare improvement
        print("\nIncremental Finetuning Impact:")
        print(f"F1 Score: {initial_metrics['f1']:.4f} → {finetuned_metrics['f1']:.4f} (Δ: {finetuned_metrics['f1'] - initial_metrics['f1']:.4f})")
        print(f"Precision: {initial_metrics['precision']:.4f} → {finetuned_metrics['precision']:.4f} (Δ: {finetuned_metrics['precision'] - initial_metrics['precision']:.4f})")
        print(f"Recall: {initial_metrics['recall']:.4f} → {finetuned_metrics['recall']:.4f} (Δ: {finetuned_metrics['recall'] - initial_metrics['recall']:.4f})")
        print(f"IoU: {initial_metrics['iou']:.4f} → {finetuned_metrics['iou']:.4f} (Δ: {finetuned_metrics['iou'] - initial_metrics['iou']:.4f})")
        print(f"ROC-AUC: {initial_metrics['roc_auc']:.4f} → {finetuned_metrics['roc_auc']:.4f} (Δ: {finetuned_metrics['roc_auc'] - initial_metrics['roc_auc']:.4f})")
        print(f"PR-AUC: {initial_metrics['pr_auc']:.4f} → {finetuned_metrics['pr_auc']:.4f} (Δ: {finetuned_metrics['pr_auc'] - initial_metrics['pr_auc']:.4f})")
        
        # Log finetuning impact deltas and final model info to wandb
        finetuning_impact = {
            "delta_f1": finetuned_metrics['f1'] - initial_metrics['f1'],
            "delta_precision": finetuned_metrics['precision'] - initial_metrics['precision'],
            "delta_recall": finetuned_metrics['recall'] - initial_metrics['recall'],
            "delta_iou": finetuned_metrics['iou'] - initial_metrics['iou'],
            "delta_roc_auc": finetuned_metrics['roc_auc'] - initial_metrics['roc_auc'],
            "delta_pr_auc": finetuned_metrics['pr_auc'] - initial_metrics['pr_auc'],
            "final_model_trees": model.n_estimators,
            "added_trees": additional_trees
        }
        wandb.log(finetuning_impact)
        
        # Log finetuned metrics
        finetuned_wandb_metrics = {f"finetuned_{k}": v for k, v in finetuned_metrics.items() if k != "confusion_matrix"}
        wandb.log(finetuned_wandb_metrics)
        wandb.log({"finetuned_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_val.tolist(),
            preds=y_pred_val_fine.tolist(),
            class_names=["0", "1"],
        )})
        
        # Use finetuned metrics for logging
        metrics = finetuned_metrics
        y_pred = y_pred_val_fine
        y_prob = y_prob_val_fine
    else:
        print("\nNo finetuning data provided, using initial model.")
        metrics = initial_metrics
        y_pred = y_pred_val
        y_prob = y_prob_val
        
        # Log that no finetuning was performed
        wandb.log({"final_model_trees": model.n_estimators, "finetuning_performed": False})

    wandb.log({k: v for k, v in metrics.items() if k != "confusion_matrix"})
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_val.tolist(),
        preds=y_pred.tolist(),
        class_names=["0", "1"],
    )})
    
    print("\n=== Final Validation Results ===")
    print("Validation Metrics:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"{k}:\n{v}")
        else:
            print(f"{k}: {v:.4f}")

    print("Classification Report:\n", classification_report(y_val, y_pred))

    if run_dummy:
        dummy = DummyClassifier(strategy="stratified", random_state=cfg.get("seed", 42))
        dummy.fit(X_train, y_train)
        d_pred = dummy.predict(X_val)
        d_prob = dummy.predict_proba(X_val)[:, 1] if hasattr(dummy, "predict_proba") else None
        d_metrics = compute_metrics(y_val, d_pred, d_prob)
        wandb.log({f"dummy_{k}": v for k, v in d_metrics.items() if k != "confusion_matrix"})
        wandb.log({"dummy_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_val.tolist(),
            preds=d_pred.tolist(),
            class_names=["0", "1"],
        )})
        print("Dummy Validation Metrics:")
        for k, v in d_metrics.items():
            if k == "confusion_matrix":
                print(f"{k}:\n{v}")
            else:
                print(f"{k}: {v}")

    # Phase 3: Final Testing
    if test_data is not None:
        print("\n=== Phase 3: Final Testing ===")
        X_test, y_test = test_data
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_pred, test_prob)
        wandb.log({f"test_{k}": v for k, v in test_metrics.items() if k != "confusion_matrix"})
        wandb.log({"test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test.tolist(),
            preds=test_pred.tolist(),
            class_names=["0", "1"],
        )})
        print("Test Metrics:")
        for k, v in test_metrics.items():
            if k == "confusion_matrix":
                print(f"{k}:\n{v}")
            else:
                print(f"{k}: {v:.4f}")
        print("Test Classification Report:\n", classification_report(y_test, test_pred))

        if run_dummy:
            d_test_pred = dummy.predict(X_test)
            d_test_prob = dummy.predict_proba(X_test)[:, 1] if hasattr(dummy, "predict_proba") else None
            d_test_metrics = compute_metrics(y_test, d_test_pred, d_test_prob)
            wandb.log({f"dummy_test_{k}": v for k, v in d_test_metrics.items() if k != "confusion_matrix"})
            wandb.log({"dummy_test_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test.tolist(),
                preds=d_test_pred.tolist(),
                class_names=["0", "1"],
            )})
            print("Dummy Test Metrics:")
            for k, v in d_test_metrics.items():
                if k == "confusion_matrix":
                    print(f"{k}:\n{v}")
                else:
                    print(f"{k}: {v:.4f}")
            print("Dummy Test Classification Report:\n", classification_report(y_test, d_test_pred))

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest like P2Rank")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--run_dummy", action="store_true", help="Run DummyClassifier for random baseline")
    args = parser.parse_args()
    main(args.config, run_dummy=args.run_dummy)

