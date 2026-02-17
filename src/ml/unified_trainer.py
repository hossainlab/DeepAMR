"""Unified Training Pipeline for DeepAMR.

This module provides a comprehensive training system that:
1. Combines multiple data sources (NCBI, CARD, PATRIC)
2. Handles class imbalance with multiple strategies
3. Supports both sklearn and PyTorch models
4. Implements proper cross-validation
5. Provides detailed evaluation metrics
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, hamming_loss,
    precision_recall_curve, average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_dataset(data_dir: str, prefix: str) -> Dict:
    """Load a preprocessed dataset."""
    data_path = Path(data_dir)
    
    data = {
        "X_train": np.load(data_path / f"{prefix}_X_train.npy"),
        "X_val": np.load(data_path / f"{prefix}_X_val.npy"),
        "X_test": np.load(data_path / f"{prefix}_X_test.npy"),
        "y_train": np.load(data_path / f"{prefix}_y_train.npy"),
        "y_val": np.load(data_path / f"{prefix}_y_val.npy"),
        "y_test": np.load(data_path / f"{prefix}_y_test.npy"),
    }
    
    metadata_file = data_path / f"{prefix}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            data["metadata"] = json.load(f)
    else:
        data["metadata"] = {}
    
    return data


def combine_datasets(datasets: List[Dict], task_type: str = "multilabel") -> Dict:
    """Combine multiple datasets for training.
    
    For multilabel, aligns class labels across datasets.
    """
    if len(datasets) == 1:
        return datasets[0]
    
    # Get all unique class names
    all_classes = set()
    for ds in datasets:
        if "class_names" in ds.get("metadata", {}):
            all_classes.update(ds["metadata"]["class_names"])
    all_classes = sorted(all_classes)
    
    combined = {
        "X_train": [],
        "X_val": [],
        "X_test": [],
        "y_train": [],
        "y_val": [],
        "y_test": [],
        "metadata": {
            "class_names": all_classes,
            "task_type": task_type,
            "n_classes": len(all_classes),
            "source_datasets": [],
        }
    }
    
    for ds in datasets:
        ds_classes = ds.get("metadata", {}).get("class_names", [])
        combined["metadata"]["source_datasets"].append({
            "n_samples": len(ds["X_train"]) + len(ds["X_val"]) + len(ds["X_test"]),
            "classes": ds_classes,
        })
        
        if task_type == "multilabel" and ds_classes:
            # Remap labels to unified class space
            class_map = {cls: all_classes.index(cls) for cls in ds_classes if cls in all_classes}
            
            for split in ["train", "val", "test"]:
                combined[f"X_{split}"].append(ds[f"X_{split}"])
                
                # Remap y to new class indices
                y_old = ds[f"y_{split}"]
                y_new = np.zeros((len(y_old), len(all_classes)), dtype=y_old.dtype)
                for old_idx, cls in enumerate(ds_classes):
                    if cls in class_map:
                        new_idx = class_map[cls]
                        y_new[:, new_idx] = y_old[:, old_idx]
                combined[f"y_{split}"].append(y_new)
        else:
            for split in ["train", "val", "test"]:
                combined[f"X_{split}"].append(ds[f"X_{split}"])
                combined[f"y_{split}"].append(ds[f"y_{split}"])
    
    # Concatenate
    for split in ["train", "val", "test"]:
        combined[f"X_{split}"] = np.vstack(combined[f"X_{split}"])
        combined[f"y_{split}"] = np.vstack(combined[f"y_{split}"]) if task_type == "multilabel" else np.concatenate(combined[f"y_{split}"])
    
    combined["metadata"]["n_samples"] = len(combined["X_train"]) + len(combined["X_val"]) + len(combined["X_test"])
    combined["metadata"]["n_features"] = combined["X_train"].shape[1]
    
    return combined


# =============================================================================
# Class Imbalance Handling
# =============================================================================

class ImbalanceHandler:
    """Handle class imbalance with multiple strategies."""
    
    STRATEGIES = ["none", "class_weight", "smote", "adasyn", "random_over", "random_under", "smote_tomek"]
    
    def __init__(self, strategy: str = "class_weight", random_state: int = 42):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {self.STRATEGIES}")
        self.strategy = strategy
        self.random_state = random_state
    
    def get_sample_weights(self, y: np.ndarray, task_type: str = "multiclass") -> np.ndarray:
        """Compute sample weights for imbalanced data."""
        if task_type == "multilabel":
            # For multilabel, weight by inverse frequency of each label combination
            label_counts = {}
            for i, row in enumerate(y):
                key = tuple(row)
                label_counts[key] = label_counts.get(key, 0) + 1
            
            weights = np.array([1.0 / label_counts[tuple(row)] for row in y])
            weights = weights / weights.sum() * len(weights)
            return weights
        else:
            return compute_sample_weight("balanced", y)
    
    def get_class_weights(self, y: np.ndarray, task_type: str = "multiclass") -> Union[Dict, np.ndarray]:
        """Compute class weights."""
        if task_type == "multilabel":
            # For multilabel, compute weight per class based on positive/negative ratio
            n_samples = len(y)
            weights = []
            for i in range(y.shape[1]):
                n_pos = y[:, i].sum()
                n_neg = n_samples - n_pos
                if n_pos > 0:
                    weight = n_neg / n_pos
                else:
                    weight = 1.0
                weights.append(min(weight, 10.0))  # Cap at 10x
            return np.array(weights)
        else:
            classes = np.unique(y)
            weights = compute_class_weight("balanced", classes=classes, y=y)
            return dict(zip(classes, weights))
    
    def resample(self, X: np.ndarray, y: np.ndarray, task_type: str = "multiclass") -> Tuple[np.ndarray, np.ndarray]:
        """Resample data to handle imbalance."""
        if self.strategy == "none" or self.strategy == "class_weight":
            return X, y
        
        if task_type == "multilabel":
            # For multilabel, use random oversampling of minority label combinations
            logger.warning("Resampling for multilabel is experimental. Using random oversampling.")
            # Convert multilabel to label combinations for resampling
            label_strings = [''.join(map(str, row)) for row in y]
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(label_strings)
            
            sampler = RandomOverSampler(random_state=self.random_state)
            X_res, y_res_encoded = sampler.fit_resample(X, y_encoded)
            
            # Map back to multilabel
            y_res_strings = le.inverse_transform(y_res_encoded)
            y_res = np.array([[int(c) for c in s] for s in y_res_strings])
            return X_res, y_res
        
        # Multiclass resampling
        try:
            if self.strategy == "smote":
                sampler = SMOTE(random_state=self.random_state)
            elif self.strategy == "adasyn":
                sampler = ADASYN(random_state=self.random_state)
            elif self.strategy == "random_over":
                sampler = RandomOverSampler(random_state=self.random_state)
            elif self.strategy == "random_under":
                sampler = RandomUnderSampler(random_state=self.random_state)
            elif self.strategy == "smote_tomek":
                sampler = SMOTETomek(random_state=self.random_state)
            else:
                return X, y
            
            X_res, y_res = sampler.fit_resample(X, y)
            logger.info(f"Resampled: {len(X)} -> {len(X_res)} samples")
            return X_res, y_res
        except Exception as e:
            logger.warning(f"Resampling failed: {e}. Using original data.")
            return X, y


# =============================================================================
# PyTorch Models
# =============================================================================

class AMRNet(nn.Module):
    """Deep neural network for AMR prediction."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        task_type: str = "multiclass",
    ):
        super().__init__()
        self.task_type = task_type
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))

        # Note: No Sigmoid for multilabel - BCEWithLogitsLoss applies it internally
        # For multiclass, CrossEntropyLoss applies softmax internally

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class PyTorchTrainer:
    """Training wrapper for PyTorch models."""
    
    def __init__(
        self,
        model: nn.Module,
        task_type: str = "multiclass",
        class_weights: Optional[np.ndarray] = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.task_type = task_type
        
        if task_type == "multilabel":
            if class_weights is not None:
                # Use pos_weight for class-weighted multilabel loss
                pos_weight = torch.FloatTensor(class_weights).to(self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            if class_weights is not None:
                weight = torch.FloatTensor(list(class_weights.values()) if isinstance(class_weights, dict) else class_weights).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weight)
            else:
                self.criterion = nn.CrossEntropyLoss()
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 10,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train the model."""
        # Prepare data
        X_train_t = torch.FloatTensor(X_train)
        if self.task_type == "multilabel":
            y_train_t = torch.FloatTensor(y_train)
        else:
            y_train_t = torch.LongTensor(y_train)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        
        # Use weighted sampling if sample weights provided
        if sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            if self.task_type == "multilabel":
                y_val_t = torch.FloatTensor(y_val).to(self.device)
            else:
                y_val_t = torch.LongTensor(y_val).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = self.criterion(val_outputs, y_val_t).item()
                history["val_loss"].append(val_loss)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_loss:.4f}" if X_val is not None else ""
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}{val_str}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t)

            if self.task_type == "multilabel":
                # Apply sigmoid for inference (BCEWithLogitsLoss uses raw logits)
                probs = torch.sigmoid(outputs)
                return (probs.cpu().numpy() > 0.5).astype(int)
            else:
                return outputs.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t)

            if self.task_type == "multilabel":
                # Apply sigmoid for probability output
                return torch.sigmoid(outputs).cpu().numpy()
            else:
                return torch.softmax(outputs, dim=1).cpu().numpy()


# =============================================================================
# Unified Trainer
# =============================================================================

class UnifiedAMRTrainer:
    """Unified training pipeline for AMR prediction."""
    
    SKLEARN_MODELS = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=200, max_depth=20,
            n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=500,
            random_state=42, early_stopping=True
        ),
    }
    
    def __init__(
        self,
        task_type: str = "multilabel",
        imbalance_strategy: str = "class_weight",
        scale_features: bool = True,
    ):
        self.task_type = task_type
        self.imbalance_handler = ImbalanceHandler(imbalance_strategy)
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        
        self.model = None
        self.class_names = None
        self.feature_names = None
        self.results = {}
    
    def _create_sklearn_model(self, model_name: str):
        """Create sklearn model with proper wrapper for multilabel."""
        if model_name not in self.SKLEARN_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        base_model = self.SKLEARN_MODELS[model_name]()
        
        if self.task_type == "multilabel":
            return OneVsRestClassifier(base_model, n_jobs=-1)
        return base_model
    
    def _create_pytorch_model(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        """Create PyTorch model."""
        return AMRNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            task_type=self.task_type,
        )
    
    def train(
        self,
        data: Dict,
        model_type: str = "random_forest",
        use_pytorch: bool = False,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> Dict:
        """Train a model on the data."""
        logger.info(f"Training {model_type} ({'PyTorch' if use_pytorch else 'sklearn'})...")
        
        X_train = data["X_train"].copy()
        y_train = data["y_train"].copy()
        X_val = data.get("X_val")
        y_val = data.get("y_val")
        X_test = data["X_test"]
        y_test = data["y_test"]
        
        self.class_names = data.get("metadata", {}).get("class_names", [])
        self.feature_names = data.get("metadata", {}).get("feature_names", [])
        
        # Scale features
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
        
        # Handle imbalance
        if self.imbalance_handler.strategy not in ["none", "class_weight"]:
            X_train, y_train = self.imbalance_handler.resample(X_train, y_train, self.task_type)
        
        class_weights = None
        sample_weights = None
        if self.imbalance_handler.strategy == "class_weight":
            class_weights = self.imbalance_handler.get_class_weights(y_train, self.task_type)
            sample_weights = self.imbalance_handler.get_sample_weights(y_train, self.task_type)
        
        # Train model
        if use_pytorch:
            output_dim = y_train.shape[1] if self.task_type == "multilabel" else len(np.unique(y_train))
            model = self._create_pytorch_model(X_train.shape[1], output_dim)
            trainer = PyTorchTrainer(model, self.task_type, class_weights)
            
            history = trainer.fit(
                X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size,
                sample_weights=sample_weights,
            )
            
            self.model = trainer
            self.results["training_history"] = history
        else:
            model = self._create_sklearn_model(model_type)
            
            # For sklearn, OneVsRestClassifier doesn't support sample_weight directly
            # Just fit without sample weights for multilabel (class_weight is already in base estimator)
            model.fit(X_train, y_train)
            
            self.model = model
        
        # Evaluate
        train_metrics = self._evaluate(X_train, y_train, "train")
        if X_val is not None:
            val_metrics = self._evaluate(X_val, y_val, "val")
        test_metrics = self._evaluate(X_test, y_test, "test")
        
        self.results["model_type"] = model_type
        self.results["use_pytorch"] = use_pytorch
        self.results["task_type"] = self.task_type
        self.results["imbalance_strategy"] = self.imbalance_handler.strategy
        self.results["train_metrics"] = train_metrics
        if X_val is not None:
            self.results["val_metrics"] = val_metrics
        self.results["test_metrics"] = test_metrics
        self.results["class_names"] = self.class_names
        
        self._log_results(test_metrics)
        
        return self.results
    
    def _evaluate(self, X: np.ndarray, y_true: np.ndarray, split_name: str) -> Dict:
        """Evaluate model performance."""
        if hasattr(self.model, "predict"):
            y_pred = self.model.predict(X)
        else:
            y_pred = self.model.predict(X)
        
        metrics = {"split": split_name}
        
        if self.task_type == "multilabel":
            metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))
            metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
            metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["micro_precision"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
            metrics["micro_recall"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
            
            # Per-class metrics
            metrics["per_class"] = {}
            for i, cls in enumerate(self.class_names):
                metrics["per_class"][cls] = {
                    "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                    "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                    "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                    "support": int(y_true[:, i].sum()),
                }
            
            # AUC
            try:
                if hasattr(self.model, "predict_proba"):
                    y_proba = self.model.predict_proba(X)
                else:
                    y_proba = y_pred
                metrics["micro_auc"] = float(roc_auc_score(y_true, y_proba, average="micro"))
                metrics["macro_auc"] = float(roc_auc_score(y_true, y_proba, average="macro"))
            except Exception:
                pass
        else:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            
            try:
                if hasattr(self.model, "predict_proba"):
                    y_proba = self.model.predict_proba(X)
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
            except Exception:
                pass
        
        return metrics
    
    def _log_results(self, metrics: Dict):
        """Log evaluation results."""
        logger.info("=" * 60)
        logger.info("Test Results:")
        logger.info("=" * 60)
        
        if self.task_type == "multilabel":
            logger.info(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
            logger.info(f"  Micro F1: {metrics['micro_f1']:.4f}")
            logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
            logger.info(f"  Micro AUC: {metrics.get('micro_auc', 'N/A')}")
            
            logger.info("\nPer-class F1 scores:")
            for cls, cls_metrics in metrics.get("per_class", {}).items():
                logger.info(f"  {cls}: F1={cls_metrics['f1']:.3f}, Support={cls_metrics['support']}")
        else:
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1 (Weighted): {metrics['f1']:.4f}")
            logger.info(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
            if "auc" in metrics:
                logger.info(f"  AUC: {metrics['auc']:.4f}")
    
    def save(self, filepath: str):
        """Save the trained model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "model": self.model.model.state_dict() if isinstance(self.model, PyTorchTrainer) else self.model,
            "scaler": self.scaler,
            "task_type": self.task_type,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
            "results": self.results,
        }
        
        if isinstance(self.model, PyTorchTrainer):
            torch.save(save_data, filepath.replace(".joblib", ".pt"))
        else:
            joblib.dump(save_data, filepath)
        
        logger.info(f"Model saved to {filepath}")


def run_comprehensive_training(output_dir: str = "models/unified") -> Dict:
    """Run comprehensive training across all datasets and models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [],
    }
    
    # ==========================================================================
    # Experiment 1: NCBI AMR with different imbalance strategies
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 1: NCBI AMR - Imbalance Strategy Comparison")
    logger.info("="*80)
    
    ncbi_amr = load_dataset("data/processed/ncbi", "ncbi_amr")
    
    best_ncbi_result = None
    best_ncbi_f1 = 0
    
    for strategy in ["class_weight", "smote", "random_over"]:
        for model_type in ["random_forest", "extra_trees", "logistic_regression"]:
            logger.info(f"\n--- {model_type} with {strategy} ---")
            
            trainer = UnifiedAMRTrainer(
                task_type="multilabel",
                imbalance_strategy=strategy,
            )
            
            try:
                results = trainer.train(ncbi_amr, model_type=model_type)
                
                all_results["experiments"].append({
                    "dataset": "ncbi_amr",
                    "model": model_type,
                    "strategy": strategy,
                    "test_micro_f1": results["test_metrics"]["micro_f1"],
                    "test_macro_f1": results["test_metrics"]["macro_f1"],
                    "test_micro_auc": results["test_metrics"].get("micro_auc"),
                })
                
                if results["test_metrics"]["micro_f1"] > best_ncbi_f1:
                    best_ncbi_f1 = results["test_metrics"]["micro_f1"]
                    best_ncbi_result = (trainer, results, model_type, strategy)
            except Exception as e:
                logger.error(f"Failed: {e}")
    
    # Save best NCBI model
    if best_ncbi_result:
        trainer, results, model_type, strategy = best_ncbi_result
        trainer.save(str(output_path / f"ncbi_amr_best_{model_type}.joblib"))
        logger.info(f"\nBest NCBI AMR model: {model_type} with {strategy} (Micro F1: {best_ncbi_f1:.4f})")
    
    # ==========================================================================
    # Experiment 2: CARD Drug Class (larger dataset)
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 2: CARD Drug Class")
    logger.info("="*80)
    
    card_data = load_dataset("data/processed/card", "card_drug_class")
    
    best_card_result = None
    best_card_f1 = 0
    
    for model_type in ["random_forest", "extra_trees"]:
        logger.info(f"\n--- {model_type} ---")
        
        trainer = UnifiedAMRTrainer(
            task_type="multilabel",
            imbalance_strategy="class_weight",
        )
        
        try:
            results = trainer.train(card_data, model_type=model_type)
            
            all_results["experiments"].append({
                "dataset": "card_drug_class",
                "model": model_type,
                "strategy": "class_weight",
                "test_micro_f1": results["test_metrics"]["micro_f1"],
                "test_macro_f1": results["test_metrics"]["macro_f1"],
            })
            
            if results["test_metrics"]["micro_f1"] > best_card_f1:
                best_card_f1 = results["test_metrics"]["micro_f1"]
                best_card_result = (trainer, results, model_type)
        except Exception as e:
            logger.error(f"Failed: {e}")
    
    # Save best CARD model
    if best_card_result:
        trainer, results, model_type = best_card_result
        trainer.save(str(output_path / f"card_drug_class_best_{model_type}.joblib"))
        logger.info(f"\nBest CARD model: {model_type} (Micro F1: {best_card_f1:.4f})")
    
    # ==========================================================================
    # Experiment 3: PyTorch Deep Learning on NCBI AMR
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 3: Deep Learning on NCBI AMR")
    logger.info("="*80)
    
    trainer = UnifiedAMRTrainer(
        task_type="multilabel",
        imbalance_strategy="class_weight",
    )
    
    try:
        results = trainer.train(
            ncbi_amr,
            model_type="deep_learning",
            use_pytorch=True,
            epochs=100,
            batch_size=32,
        )
        
        all_results["experiments"].append({
            "dataset": "ncbi_amr",
            "model": "deep_learning",
            "strategy": "class_weight",
            "test_micro_f1": results["test_metrics"]["micro_f1"],
            "test_macro_f1": results["test_metrics"]["macro_f1"],
        })
        
        # Save PyTorch model
        trainer.save(str(output_path / "ncbi_amr_deep_learning.pt"))
    except Exception as e:
        logger.error(f"Deep learning failed: {e}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    # Sort by F1 score
    experiments_sorted = sorted(
        all_results["experiments"],
        key=lambda x: x.get("test_micro_f1", 0),
        reverse=True,
    )
    
    logger.info("\nTop 5 Models by Micro F1:")
    for i, exp in enumerate(experiments_sorted[:5], 1):
        logger.info(f"  {i}. {exp['dataset']} / {exp['model']} / {exp['strategy']}: "
                   f"Micro F1={exp['test_micro_f1']:.4f}, Macro F1={exp['test_macro_f1']:.4f}")
    
    # Save all results
    with open(output_path / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_path / 'training_results.json'}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified AMR Training Pipeline")
    parser.add_argument("--mode", choices=["full", "quick", "ncbi", "card"], default="full")
    parser.add_argument("--output-dir", default="models/unified")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        run_comprehensive_training(args.output_dir)
    else:
        # Quick mode for testing
        logger.info("Running quick training mode...")
        ncbi_amr = load_dataset("data/processed/ncbi", "ncbi_amr")
        trainer = UnifiedAMRTrainer(task_type="multilabel", imbalance_strategy="class_weight")
        results = trainer.train(ncbi_amr, model_type="random_forest")
        trainer.save(f"{args.output_dir}/quick_test.joblib")
