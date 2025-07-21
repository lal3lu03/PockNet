#!/usr/bin/env python3
"""
Minimal Random Forest implementation for PockNet binding site prediction.
This script runs without hydra or pytorch lightning dependencies and includes feature importance analysis.

Required packages (install with pip):
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn (optional, for better plots)
- imbalanced-learn (optional, for sampling strategies)
- joblib (usually comes with scikit-learn)


to run the script use conda activate p2rank_env
and then run:
LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" python train_rf_minimal.py
"""


import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check dependencies first - simplified for p2rank_env
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.metrics import precision_recall_curve, average_precision_score, jaccard_score
    import joblib
    print("All required packages found successfully.")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running in the p2rank_env environment:")
    print("conda activate p2rank_env")
    sys.exit(1)

# Optional imports
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Warning: seaborn not available. Using basic matplotlib styling.")
    sns = None


class PockNetDataLoader:
    """Simple dataloader for PockNet data without Lightning dependencies."""
    
    def __init__(self, data_dir="data/", normalize_features=True, sampling_strategy="none"):
        self.data_dir = data_dir
        self.normalize_features = normalize_features
        self.sampling_strategy = sampling_strategy
        self.scaler = StandardScaler() if normalize_features else None
        self.feature_names = None
        
    def load_data(self):
        """Load and prepare training and test data."""
        # Load training data (chen11)
        train_file = os.path.join(self.data_dir, "train", "chen11.csv")
        test_file = os.path.join(self.data_dir, "test", "bu48.csv")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test data not found: {test_file}")
            
        print(f"Loading training data from: {train_file}")
        print(f"Loading test data from: {test_file}")
        
        # Load datasets
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Define metadata columns to exclude from features
        metadata_columns = ['file_name', 'x', 'y', 'z', 'chain_id', 'residue_number', 'residue_name', 'class']
        
        # Exclude features for comparison (empty list = include all features)
        excluded_features = []  # Include all features to demonstrate overfitting
        
        # Get feature columns (all except metadata and excluded features)
        feature_columns = [col for col in train_df.columns 
                          if col not in metadata_columns and col not in excluded_features]
        self.feature_names = feature_columns
        
        print(f"Number of features: {len(feature_columns)}")
        print(f"Excluded features: {excluded_features}")
        print(f"Feature columns: {feature_columns[:10]}...")  # Show first 10 features
        
        # Extract features and targets
        X_train_full = train_df[feature_columns].values.astype(np.float32)
        y_train_full = train_df['class'].values.astype(int)
        
        X_test = test_df[feature_columns].values.astype(np.float32)
        y_test = test_df['class'].values.astype(int)
        
        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,  # 80% train, 20% validation
            random_state=42,
            stratify=y_train_full
        )
        
        print(f"\nData splits:")
        print(f"Training: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples") 
        print(f"Test: {X_test.shape[0]} samples")
        
        # Print class distributions
        print(f"\nClass distributions:")
        print(f"Training - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
        print(f"Validation - Class 0: {np.sum(y_val == 0)}, Class 1: {np.sum(y_val == 1)}")
        print(f"Test - Class 0: {np.sum(y_test == 0)}, Class 1: {np.sum(y_test == 1)}")
        
        # Apply normalization if requested
        if self.normalize_features:
            print("Applying feature normalization...")
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
        
        # Apply sampling strategy for class imbalance if requested
        if self.sampling_strategy != "none":
            X_train, y_train = self._apply_sampling(X_train, y_train)
            print(f"After sampling - Training: Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _apply_sampling(self, X, y):
        """Apply sampling strategy to handle class imbalance."""
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError:
            print("Warning: imbalanced-learn not available. Skipping sampling.")
            return X, y
        
        print(f"Original class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
        
        if self.sampling_strategy == "oversample":
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print("Applied SMOTE oversampling")
            
        elif self.sampling_strategy == "undersample":
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            print("Applied random undersampling")
            
        elif self.sampling_strategy == "combined":
            # First oversample, then undersample
            smote = SMOTE(random_state=42)
            X_temp, y_temp = smote.fit_resample(X, y)
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_temp, y_temp)
            print("Applied combined sampling (SMOTE + undersampling)")
            
        else:
            X_resampled, y_resampled = X, y
        
        return X_resampled, y_resampled


class RandomForestTrainer:
    """Random Forest trainer with feature importance analysis."""
    
    def __init__(self, 
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features="sqrt",
                 random_state=42,
                 n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced"  # Handle class imbalance
        )
        self.feature_names = None
        
    def train(self, X_train, y_train, feature_names=None):
        """Train the random forest model."""
        print("Training Random Forest...")
        print(f"Model parameters: {self.model.get_params()}")
        
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        
        # Print training accuracy
        train_score = self.model.score(X_train, y_train)
        print(f"Training accuracy: {train_score:.4f}")
        
        return self.model
    
    def evaluate(self, X, y, dataset_name=""):
        """Evaluate the model and return metrics."""
        print(f"\nEvaluating on {dataset_name} set...")
        
        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Basic metrics
        accuracy = self.model.score(X, y)
        auc_score = roc_auc_score(y, y_pred_proba)
        avg_precision = average_precision_score(y, y_pred_proba)
        iou_score = jaccard_score(y, y_pred, average='binary')  # IoU for binary classification
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"IoU (Jaccard Score): {iou_score:.4f}")
        
        # Classification report
        print(f"\nClassification Report ({dataset_name}):")
        print(classification_report(y, y_pred))
        
        return {
            'accuracy': accuracy,
            'auc_roc': auc_score,
            'avg_precision': avg_precision,
            'iou_score': iou_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance analysis."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        importance = self.model.feature_importances_
        
        if self.feature_names is not None:
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            feature_importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(feature_importance_df.head(top_n))
        
        return feature_importance_df
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance."""
        feature_importance_df = self.get_feature_importance(top_n=len(self.feature_names or []))
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
        
        return feature_importance_df
    
    def plot_roc_curves(self, results_dict, save_path=None):
        """Plot ROC curves for different datasets."""
        plt.figure(figsize=(10, 8))
        
        for dataset_name, (y_true, y_pred_proba) in results_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, path):
        """Save the trained model."""
        joblib.dump(self.model, path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path):
        """Load a trained model."""
        self.model = joblib.load(path)
        print(f"Model loaded from: {path}")


def main():
    """Main training and evaluation pipeline."""
    print("=" * 60)
    print("PockNet Random Forest - Minimal Implementation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("rf_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize dataloader
    dataloader = PockNetDataLoader(
        data_dir="data/",
        normalize_features=True,
        sampling_strategy="oversample"  # Options: "none", "oversample", "undersample", "combined"
    )
    
    # Load data
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataloader.load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the data files exist in the data/ directory.")
        return
    
    # Initialize trainer with good default parameters
    trainer = RandomForestTrainer(
        n_estimators=200,        # More trees for better performance
        max_depth=10,            # Limit depth to prevent overfitting
        min_samples_split=5,     # Minimum samples to split a node
        min_samples_leaf=2,      # Minimum samples in leaf
        max_features="sqrt",     # Number of features for best split
        random_state=42,
        n_jobs=-1               # Use all available cores
    )
    
    # Train the model
    trainer.train(X_train, y_train, feature_names=dataloader.feature_names)
    
    # Evaluate on all datasets
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    train_results = trainer.evaluate(X_train, y_train, "Training")
    val_results = trainer.evaluate(X_val, y_val, "Validation") 
    test_results = trainer.evaluate(X_test, y_test, "Test")
    
    # Feature importance analysis
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Get and save feature importance
    feature_importance_df = trainer.plot_feature_importance(
        top_n=20, 
        save_path=output_dir / "feature_importance.png"
    )
    
    # Save feature importance as CSV
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"Feature importance saved to: {output_dir / 'feature_importance.csv'}")
    
    # Plot ROC curves
    roc_data = {
        "Training": (y_train, train_results['y_pred_proba']),
        "Validation": (y_val, val_results['y_pred_proba']),
        "Test": (y_test, test_results['y_pred_proba'])
    }
    trainer.plot_roc_curves(roc_data, save_path=output_dir / "roc_curves.png")
    
    # Save model
    trainer.save_model(output_dir / "random_forest_model.pkl")
    
    # Save summary results
    summary = {
        'model_params': trainer.model.get_params(),
        'train_accuracy': train_results['accuracy'],
        'train_auc': train_results['auc_roc'],
        'train_iou': train_results['iou_score'],
        'val_accuracy': val_results['accuracy'],
        'val_auc': val_results['auc_roc'],
        'val_iou': val_results['iou_score'],
        'test_accuracy': test_results['accuracy'],
        'test_auc': test_results['auc_roc'],
        'test_iou': test_results['iou_score'],
        'num_features': len(dataloader.feature_names or []),
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'test_samples': len(y_test)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "training_summary.csv", index=False)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {output_dir}")
    print(f"Key files created:")
    print(f"  - random_forest_model.pkl: Trained model")
    print(f"  - feature_importance.csv: Feature importance rankings")
    print(f"  - feature_importance.png: Feature importance plot")
    print(f"  - roc_curves.png: ROC curve comparison")
    print(f"  - training_summary.csv: Summary metrics")
    
    print(f"\nFinal Test Results:")
    print(f"  - Accuracy: {test_results['accuracy']:.4f}")
    print(f"  - AUC-ROC: {test_results['auc_roc']:.4f}")
    print(f"  - Avg Precision: {test_results['avg_precision']:.4f}")
    print(f"  - IoU Score: {test_results['iou_score']:.4f}")
    
    # Generalization analysis - detect overfitting patterns
    print("\n" + "=" * 50)
    print("GENERALIZATION ANALYSIS")
    print("=" * 50)
    
    # Calculate generalization gaps
    train_test_auc_gap = train_results['auc_roc'] - test_results['auc_roc']
    train_test_accuracy_gap = train_results['accuracy'] - test_results['accuracy']
    
    print(f"Training-Test Performance Gaps:")
    print(f"  AUC-ROC gap: {train_test_auc_gap:.4f} ({train_test_auc_gap/train_results['auc_roc']*100:.1f}%)")
    print(f"  Accuracy gap: {train_test_accuracy_gap:.4f} ({train_test_accuracy_gap/train_results['accuracy']*100:.1f}%)")
    
    # Feature dominance analysis
    feature_importance_df = trainer.get_feature_importance(top_n=len(dataloader.feature_names or []))
    top_feature_importance = feature_importance_df.iloc[0]['importance']
    importance_concentration = feature_importance_df.head(5)['importance'].sum()
    
    print(f"\nFeature Dominance Analysis:")
    print(f"  Top feature importance: {top_feature_importance:.3f} ({top_feature_importance*100:.1f}%)")
    print(f"  Top 5 features concentration: {importance_concentration:.3f} ({importance_concentration*100:.1f}%)")
    
    if top_feature_importance > 0.25:
        print(f"  ⚠️  WARNING: High feature dominance detected (>{top_feature_importance*100:.1f}%)")
        print(f"      This may indicate overfitting to training data patterns")
        print(f"      Consider feature normalization or ensemble methods")
    
    # Save generalization metrics
    generalization_metrics = {
        'train_test_auc_gap': train_test_auc_gap,
        'train_test_accuracy_gap': train_test_accuracy_gap,
        'top_feature_importance': top_feature_importance,
        'top5_importance_concentration': importance_concentration,
        'generalization_score': 1 - (train_test_auc_gap / train_results['auc_roc'])  # Higher is better
    }
    
    generalization_df = pd.DataFrame([generalization_metrics])
    generalization_df.to_csv(output_dir / "generalization_analysis.csv", index=False)
    print(f"\nGeneralization metrics saved to: {output_dir / 'generalization_analysis.csv'}")


if __name__ == "__main__":
    main()
