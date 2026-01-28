"""Comprehensive Model Training Pipeline for DeepAMR.

This module provides a complete training pipeline for AMR prediction models,
supporting both traditional ML (sklearn) and deep learning (PyTorch) approaches.
Works with preprocessed data from any source (NCBI, PATRIC, CARD, ResFinder).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import joblib

# Sklearn imports
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    hamming_loss,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_processed_data(data_dir: str, prefix: str) -> Dict:
    """Load preprocessed data from directory.

    Args:
        data_dir: Directory containing processed files
        prefix: File prefix (e.g., 'ncbi_organism', 'ncbi_amr')

    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test, metadata
    """
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


# ============================================================================
# Model Factory
# ============================================================================

class ModelFactory:
    """Factory for creating ML models with default configurations."""

    MODELS = {
        # Tree-based models
        "random_forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "n_jobs": -1,
                "random_state": 42,
                "class_weight": "balanced",
            },
        },
        "extra_trees": {
            "class": ExtraTreesClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 20,
                "n_jobs": -1,
                "random_state": 42,
                "class_weight": "balanced",
            },
        },
        "gradient_boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": 42,
            },
        },
        "adaboost": {
            "class": AdaBoostClassifier,
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": 42,
            },
        },
        # Linear models
        "logistic_regression": {
            "class": LogisticRegression,
            "params": {
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
                "n_jobs": -1,
            },
        },
        "sgd": {
            "class": SGDClassifier,
            "params": {
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
                "n_jobs": -1,
            },
        },
        # SVM models
        "svm": {
            "class": SVC,
            "params": {
                "kernel": "rbf",
                "probability": True,
                "random_state": 42,
                "class_weight": "balanced",
            },
        },
        "linear_svm": {
            "class": LinearSVC,
            "params": {
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
            },
        },
        # Other models
        "knn": {
            "class": KNeighborsClassifier,
            "params": {
                "n_neighbors": 5,
                "n_jobs": -1,
            },
        },
        "mlp": {
            "class": MLPClassifier,
            "params": {
                "hidden_layer_sizes": (256, 128, 64),
                "max_iter": 500,
                "random_state": 42,
                "early_stopping": True,
            },
        },
    }

    @classmethod
    def create(
        cls,
        model_name: str,
        task_type: str = "multiclass",
        custom_params: Optional[Dict] = None,
    ):
        """Create a model instance.

        Args:
            model_name: Name of the model
            task_type: 'binary', 'multiclass', or 'multilabel'
            custom_params: Custom parameters to override defaults

        Returns:
            Model instance
        """
        if model_name not in cls.MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(cls.MODELS.keys())}"
            )

        config = cls.MODELS[model_name]
        params = config["params"].copy()
        if custom_params:
            params.update(custom_params)

        model = config["class"](**params)

        # Wrap for multi-label
        if task_type == "multilabel":
            model = OneVsRestClassifier(model, n_jobs=-1)

        return model

    @classmethod
    def list_models(cls) -> List[str]:
        """List available models."""
        return list(cls.MODELS.keys())


# ============================================================================
# AMR Model Trainer
# ============================================================================

class AMRModelTrainer:
    """Training pipeline for AMR prediction models."""

    def __init__(
        self,
        model_name: str = "random_forest",
        task_type: str = "multiclass",
        scale_features: bool = True,
        model_params: Optional[Dict] = None,
    ):
        """Initialize trainer.

        Args:
            model_name: Name of the model to use
            task_type: 'binary', 'multiclass', or 'multilabel'
            scale_features: Whether to standardize features
            model_params: Custom model parameters
        """
        self.model_name = model_name
        self.task_type = task_type
        self.scale_features = scale_features

        self.model = ModelFactory.create(model_name, task_type, model_params)
        self.scaler = StandardScaler() if scale_features else None

        self.feature_names: Optional[List[str]] = None
        self.class_names: Optional[List[str]] = None
        self.is_fitted = False
        self.training_history: Dict = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ) -> "AMRModelTrainer":
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names
            class_names: Class names

        Returns:
            self
        """
        logger.info(f"Training {self.model_name} ({self.task_type})...")
        logger.info(f"  Training samples: {X_train.shape[0]}")
        logger.info(f"  Features: {X_train.shape[1]}")

        self.feature_names = feature_names
        self.class_names = class_names

        # Scale features
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)

        # Train
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate on validation if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_history["validation"] = val_metrics
            self._log_metrics("Validation", val_metrics)

        logger.info("Training complete!")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        if self.scaler:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        if self.scaler:
            X = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        else:
            raise NotImplementedError("Model does not support probability prediction")

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """Evaluate model performance."""
        y_pred = self.predict(X)

        metrics = {}

        if self.task_type == "multilabel":
            metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))
            metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
            metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["micro_precision"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
            metrics["micro_recall"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

            # Per-class metrics
            if self.class_names:
                metrics["per_class"] = {}
                for i, name in enumerate(self.class_names):
                    metrics["per_class"][name] = {
                        "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                        "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                        "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                        "support": int(y_true[:, i].sum()),
                    }

            # AUC
            try:
                y_proba = self.predict_proba(X)
                metrics["micro_auc"] = float(roc_auc_score(y_true, y_proba, average="micro"))
                metrics["macro_auc"] = float(roc_auc_score(y_true, y_proba, average="macro"))
            except Exception:
                pass

        else:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

            # Per-class report
            if self.class_names:
                report = classification_report(
                    y_true, y_pred,
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0,
                )
                metrics["classification_report"] = report

            # AUC
            try:
                y_proba = self.predict_proba(X)
                if self.task_type == "binary":
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
            except Exception:
                pass

        return metrics

    def _log_metrics(self, prefix: str, metrics: Dict) -> None:
        """Log metrics."""
        if self.task_type == "multilabel":
            logger.info(f"  {prefix} - Hamming Loss: {metrics.get('hamming_loss', 0):.4f}")
            logger.info(f"  {prefix} - Micro F1: {metrics.get('micro_f1', 0):.4f}")
            logger.info(f"  {prefix} - Macro F1: {metrics.get('macro_f1', 0):.4f}")
        else:
            logger.info(f"  {prefix} - Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"  {prefix} - F1: {metrics.get('f1', 0):.4f}")

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "estimators_"):
            importances = np.mean([
                est.feature_importances_ for est in self.model.estimators_
                if hasattr(est, "feature_importances_")
            ], axis=0)
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).mean(axis=0)
        else:
            return []

        if self.feature_names:
            importance_list = list(zip(self.feature_names, importances))
        else:
            importance_list = [(f"feature_{i}", imp) for i, imp in enumerate(importances)]

        importance_list.sort(key=lambda x: x[1], reverse=True)
        return importance_list[:top_n]

    def save(self, filepath: str) -> None:
        """Save model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "is_fitted": self.is_fitted,
            "training_history": self.training_history,
        }, filepath)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AMRModelTrainer":
        """Load model."""
        data = joblib.load(filepath)

        trainer = cls(
            model_name=data["model_name"],
            task_type=data["task_type"],
            scale_features=data["scaler"] is not None,
        )

        trainer.model = data["model"]
        trainer.scaler = data["scaler"]
        trainer.feature_names = data["feature_names"]
        trainer.class_names = data["class_names"]
        trainer.is_fitted = data["is_fitted"]
        trainer.training_history = data["training_history"]

        return trainer


# ============================================================================
# Training Pipeline
# ============================================================================

def train_single_model(
    data_dir: str,
    prefix: str,
    model_name: str = "random_forest",
    output_dir: str = "models",
) -> Dict:
    """Train a single model on preprocessed data.

    Args:
        data_dir: Directory with preprocessed data
        prefix: Data file prefix
        model_name: Model to train
        output_dir: Output directory for model and results

    Returns:
        Dictionary with metrics
    """
    logger.info("=" * 60)
    logger.info(f"Training {model_name} on {prefix}")
    logger.info("=" * 60)

    # Load data
    data = load_processed_data(data_dir, prefix)
    metadata = data.get("metadata", {})

    task_type = metadata.get("task_type", "multiclass")
    feature_names = metadata.get("feature_names", [])
    class_names = metadata.get("class_names", [])

    logger.info(f"Task: {task_type}")
    logger.info(f"Train: {data['X_train'].shape[0]}, Val: {data['X_val'].shape[0]}, Test: {data['X_test'].shape[0]}")
    logger.info(f"Features: {data['X_train'].shape[1]}, Classes: {len(class_names)}")

    # Train
    trainer = AMRModelTrainer(model_name=model_name, task_type=task_type)
    trainer.fit(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        feature_names, class_names,
    )

    # Evaluate on test set
    logger.info("\nTest Set Results:")
    test_metrics = trainer.evaluate(data["X_test"], data["y_test"])
    trainer._log_metrics("Test", test_metrics)

    # Feature importance
    logger.info("\nTop 10 Important Features:")
    for feat, imp in trainer.get_feature_importance(10):
        logger.info(f"  {feat}: {imp:.4f}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / f"{prefix}_{model_name}.joblib"
    trainer.save(str(model_file))

    results = {
        "model_name": model_name,
        "task_type": task_type,
        "class_names": class_names,
        "test_metrics": test_metrics,
        "feature_importance": trainer.get_feature_importance(20),
    }

    results_file = output_path / f"{prefix}_{model_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_file}")
    return results


def train_multiple_models(
    data_dir: str,
    prefix: str,
    models: Optional[List[str]] = None,
    output_dir: str = "models",
) -> Dict[str, Dict]:
    """Train multiple models and compare results.

    Args:
        data_dir: Directory with preprocessed data
        prefix: Data file prefix
        models: List of models to train (default: all available)
        output_dir: Output directory

    Returns:
        Dictionary mapping model names to results
    """
    if models is None:
        models = ["random_forest", "extra_trees", "gradient_boosting", "logistic_regression", "mlp"]

    all_results = {}

    for model_name in models:
        try:
            results = train_single_model(data_dir, prefix, model_name, output_dir)
            all_results[model_name] = results
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    # Summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison Summary")
    logger.info("=" * 60)

    # Load task type to determine which metrics to show
    data = load_processed_data(data_dir, prefix)
    task_type = data.get("metadata", {}).get("task_type", "multiclass")

    if task_type == "multilabel":
        metric_key = "micro_f1"
        metric_name = "Micro F1"
    else:
        metric_key = "f1"
        metric_name = "F1 Score"

    comparison = []
    for model_name, results in all_results.items():
        if "error" in results:
            comparison.append((model_name, 0.0))
        else:
            score = results.get("test_metrics", {}).get(metric_key, 0.0)
            comparison.append((model_name, score))

    comparison.sort(key=lambda x: x[1], reverse=True)

    for model_name, score in comparison:
        logger.info(f"  {model_name}: {metric_name} = {score:.4f}")

    # Save comparison
    comparison_file = Path(output_dir) / f"{prefix}_model_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train AMR prediction models")
    parser.add_argument("--data-dir", default="data/processed/ncbi", help="Data directory")
    parser.add_argument("--prefix", default="ncbi_organism", help="Data prefix")
    parser.add_argument("--model", default="random_forest", help="Model name (or 'all' for comparison)")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--list-models", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in ModelFactory.list_models():
            print(f"  - {model}")
        return

    if args.model == "all":
        train_multiple_models(args.data_dir, args.prefix, output_dir=args.output_dir)
    else:
        train_single_model(args.data_dir, args.prefix, args.model, args.output_dir)


if __name__ == "__main__":
    main()
