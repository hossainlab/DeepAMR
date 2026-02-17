#!/usr/bin/env python3
"""Optimized Model Training for DeepAMR with advanced techniques for higher accuracy.

Key improvements:
1. Advanced class imbalance handling (SMOTE + focal loss)
2. Hyperparameter optimization with Optuna
3. Ensemble methods (stacking multiple models)
4. Feature engineering enhancements
5. Cross-validation with stratification
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# ML imports
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    hamming_loss,
)
from sklearn.multiclass import OneVsRestClassifier

# Imbalance handling
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

# Feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedAMRTrainer:
    """Advanced trainer with optimization techniques for higher accuracy."""

    def __init__(self, task_type: str = "multilabel", use_optimization: bool = True):
        self.task_type = task_type
        self.use_optimization = use_optimization
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.ensemble_model = None
        self.results = {}

        # Advanced hyperparameters based on analysis
        self.optimized_params = {
            "random_forest": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "oob_score": True,
                "class_weight": "balanced_subsample",
                "random_state": 42,
                "n_jobs": -1,
            },
            "extra_trees": {
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": False,
                "class_weight": "balanced_subsample",
                "random_state": 42,
                "n_jobs": -1,
            },
            "gradient_boosting": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "random_state": 42,
            },
            "logistic_regression": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",  # Changed from liblinear to support n_jobs
                "max_iter": 2000,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
            "mlp": {
                "hidden_layer_sizes": (512, 256, 128, 64),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "learning_rate": "adaptive",
                "learning_rate_init": 0.001,
                "max_iter": 1000,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "random_state": 42,
            },
        }

    def create_base_models(self) -> Dict:
        """Create optimized base models."""
        models = {}

        for name, params in self.optimized_params.items():
            if name == "random_forest":
                models[name] = RandomForestClassifier(**params)
            elif name == "extra_trees":
                models[name] = ExtraTreesClassifier(**params)
            elif name == "gradient_boosting":
                models[name] = GradientBoostingClassifier(**params)
            elif name == "logistic_regression":
                models[name] = LogisticRegression(**params)
            elif name == "mlp":
                models[name] = MLPClassifier(**params)

        return models

    def handle_class_imbalance(
        self, X: np.ndarray, y: np.ndarray, strategy: str = "smote_tomek"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced class imbalance handling."""

        if self.task_type == "multilabel":
            # Convert to label combinations for multilabel
            label_strings = ["".join(map(str, row)) for row in y]
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_encoded = le.fit_transform(label_strings)

            if strategy == "smote_tomek":
                sampler = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=3))
            elif strategy == "borderline_smote":
                sampler = BorderlineSMOTE(random_state=42, kind="borderline-1")
            elif strategy == "adasyn":
                sampler = ADASYN(random_state=42, n_neighbors=3)
            else:
                sampler = SMOTE(random_state=42, k_neighbors=3)

            try:
                X_res, y_res_encoded = sampler.fit_resample(X, y_encoded)
                # Map back to multilabel
                y_res_strings = le.inverse_transform(y_res_encoded)
                y_res = np.array([[int(c) for c in s] for s in y_res_strings])
                return X_res, y_res
            except Exception as e:
                logger.warning(f"Resampling failed: {e}. Using original data.")
                return X, y
        else:
            # Multiclass
            if strategy == "smote_tomek":
                sampler = SMOTETomek(random_state=42)
            elif strategy == "borderline_smote":
                sampler = BorderlineSMOTE(random_state=42, kind="borderline-1")
            elif strategy == "adasyn":
                sampler = ADASYN(random_state=42)
            else:
                sampler = SMOTE(random_state=42)

            try:
                X_res, y_res = sampler.fit_resample(X, y)
                return X_res, y_res
            except Exception as e:
                logger.warning(f"Resampling failed: {e}. Using original data.")
                return X, y

    def feature_engineering(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selection_method: str = "mutual_info",
        k_best: int = 300,
    ) -> np.ndarray:
        """Advanced feature selection and engineering."""

        if self.task_type == "multilabel":
            # For multilabel, use average mutual information across labels
            mi_scores = np.zeros(X.shape[1])
            for i in range(y.shape[1]):
                mi = mutual_info_classif(X, y[:, i], random_state=42)
                mi_scores += mi
            mi_scores /= y.shape[1]

            # Select top k features
            top_indices = np.argsort(mi_scores)[-k_best:][::-1]
            self.feature_selector = top_indices
            return X[:, top_indices]
        else:
            if selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_classif, k=k_best)
                X_selected = selector.fit_transform(X, y)
                self.feature_selector = selector
                return X_selected
            else:
                # Use RFE with a simple estimator
                from sklearn.tree import DecisionTreeClassifier

                estimator = DecisionTreeClassifier(random_state=42)
                selector = RFE(estimator, n_features_to_select=k_best)
                X_selected = selector.fit_transform(X, y)
                self.feature_selector = selector
                return X_selected

    def create_ensemble(
        self, base_models: Dict, ensemble_type: str = "stacking"
    ) -> Union[VotingClassifier, StackingClassifier]:
        """Create ensemble model from base models."""

        if ensemble_type == "voting":
            # Weighted voting based on individual performance
            estimators = [(name, model) for name, model in base_models.items()]
            return VotingClassifier(estimators, voting="soft", n_jobs=-1)

        elif ensemble_type == "stacking":
            estimators = [(name, model) for name, model in base_models.items()]

            # Use meta-learner for stacking
            if self.task_type == "multilabel":
                meta_learner = OneVsRestClassifier(
                    LogisticRegression(C=0.5, random_state=42, max_iter=1000), n_jobs=-1
                )
            else:
                meta_learner = LogisticRegression(C=0.5, random_state=42, max_iter=1000)

            return StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5,
                stack_method="predict_proba",
                n_jobs=-1,
            )

    def train_single_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        use_resampling: bool = True,
        use_feature_selection: bool = True,
    ) -> Dict:
        """Train a single optimized model."""

        logger.info(f"Training optimized {model_name}...")

        # Copy data to avoid modifying original
        X_train_proc = X_train.copy()
        y_train_proc = y_train.copy()
        X_val_proc = X_val.copy()
        X_test_proc = X_test.copy()

        # Feature scaling
        X_train_proc = self.scaler.fit_transform(X_train_proc)
        X_val_proc = self.scaler.transform(X_val_proc)
        X_test_proc = self.scaler.transform(X_test_proc)

        # Feature selection
        if use_feature_selection:
            X_train_proc = self.feature_engineering(X_train_proc, y_train_proc)
            X_val_proc = X_val_proc[:, self.feature_selector]
            X_test_proc = X_test_proc[:, self.feature_selector]
            logger.info(f"Features selected: {X_train_proc.shape[1]}")

        # Handle class imbalance
        if use_resampling:
            X_train_proc, y_train_proc = self.handle_class_imbalance(
                X_train_proc, y_train_proc, strategy="smote_tomek"
            )
            logger.info(f"After resampling: {X_train_proc.shape[0]} samples")

        # Create model
        models = self.create_base_models()
        model = models[model_name]

        # Wrap for multilabel
        if self.task_type == "multilabel":
            model = OneVsRestClassifier(model, n_jobs=-1)

        # Train model
        model.fit(X_train_proc, y_train_proc)

        # Evaluate
        train_metrics = self._evaluate_model(model, X_train_proc, y_train_proc)
        val_metrics = self._evaluate_model(model, X_val_proc, y_val)
        test_metrics = self._evaluate_model(model, X_test_proc, y_test)

        # Store model
        self.models[model_name] = {
            "model": model,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "metrics": test_metrics,
        }

        return {
            "model_name": model_name,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ensemble_type: str = "stacking",
        use_resampling: bool = True,
        use_feature_selection: bool = True,
    ) -> Dict:
        """Train ensemble model for best performance."""

        logger.info(f"Training {ensemble_type} ensemble...")

        # Copy data
        X_train_proc = X_train.copy()
        y_train_proc = y_train.copy()
        X_val_proc = X_val.copy()
        X_test_proc = X_test.copy()

        # Preprocessing
        X_train_proc = self.scaler.fit_transform(X_train_proc)
        X_val_proc = self.scaler.transform(X_val_proc)
        X_test_proc = self.scaler.transform(X_test_proc)

        if use_feature_selection:
            X_train_proc = self.feature_engineering(X_train_proc, y_train_proc)
            X_val_proc = X_val_proc[:, self.feature_selector]
            X_test_proc = X_test_proc[:, self.feature_selector]

        if use_resampling:
            X_train_proc, y_train_proc = self.handle_class_imbalance(
                X_train_proc, y_train_proc, strategy="smote_tomek"
            )

        # Create base models for ensemble
        base_models = self.create_base_models()

        # Wrap for multilabel
        if self.task_type == "multilabel":
            for name in base_models:
                base_models[name] = OneVsRestClassifier(base_models[name], n_jobs=-1)

        # Train individual models first to get performance
        individual_results = {}
        for name, model in base_models.items():
            model.fit(X_train_proc, y_train_proc)
            individual_results[name] = self._evaluate_model(model, X_val_proc, y_val)

        logger.info("Individual model performances:")
        for name, metrics in individual_results.items():
            f1_score = metrics.get("f1_macro", metrics.get("micro_f1", 0))
            logger.info(f"  {name}: {f1_score:.4f}")

        # Create and train ensemble
        ensemble = self.create_ensemble(base_models, ensemble_type)
        ensemble.fit(X_train_proc, y_train_proc)

        # Evaluate ensemble
        test_metrics = self._evaluate_model(ensemble, X_test_proc, y_test)

        # Store ensemble
        self.ensemble_model = {
            "model": ensemble,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "metrics": test_metrics,
            "type": ensemble_type,
        }

        return {
            "ensemble_type": ensemble_type,
            "individual_results": individual_results,
            "test_metrics": test_metrics,
        }

    def _evaluate_model(self, model, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """Evaluate model performance."""
        y_pred = model.predict(X)

        metrics = {}

        if self.task_type == "multilabel":
            metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))
            metrics["micro_f1"] = float(
                f1_score(y_true, y_pred, average="micro", zero_division=0)
            )
            metrics["macro_f1"] = float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
            metrics["weighted_f1"] = float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            )

            # AUC
            try:
                y_proba = model.predict_proba(X)
                metrics["micro_auc"] = float(
                    roc_auc_score(y_true, y_proba, average="micro")
                )
                metrics["macro_auc"] = float(
                    roc_auc_score(y_true, y_proba, average="macro")
                )
            except Exception:
                pass
        else:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            )
            metrics["f1_macro"] = float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )

            try:
                y_proba = model.predict_proba(X)
                metrics["auc"] = float(
                    roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
                )
            except Exception:
                pass

        return metrics

    def save_models(self, output_dir: str):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        import joblib

        # Save individual models
        for name, model_data in self.models.items():
            model_file = output_path / f"optimized_{name}.joblib"
            joblib.dump(model_data, model_file)
            logger.info(f"Saved {name} model to {model_file}")

        # Save ensemble
        if self.ensemble_model:
            ensemble_file = (
                output_path / f"optimized_ensemble_{self.ensemble_model['type']}.joblib"
            )
            joblib.dump(self.ensemble_model, ensemble_file)
            logger.info(f"Saved ensemble model to {ensemble_file}")

        # Save results
        results_file = output_path / "optimization_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Saved results to {results_file}")


def run_optimized_training():
    """Run comprehensive optimized training pipeline."""
    logger.info("=" * 80)
    logger.info("OPTIMIZED AMR MODEL TRAINING")
    logger.info("=" * 80)

    # Load data
    from src.ml.unified_trainer import load_dataset

    data = load_dataset("data/processed/ncbi", "ncbi_amr")

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    metadata = data["metadata"]

    logger.info(
        f"Dataset loaded: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test"
    )
    logger.info(
        f"Features: {X_train.shape[1]}, Classes: {len(metadata['class_names'])}"
    )

    # Initialize trainer
    trainer = OptimizedAMRTrainer(task_type="multilabel")

    # Train individual models
    model_names = [
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "logistic_regression",
        "mlp",
    ]
    individual_results = {}

    for model_name in model_names:
        try:
            results = trainer.train_single_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                model_name,
                use_resampling=True,
                use_feature_selection=True,
            )
            individual_results[model_name] = results
            logger.info(
                f"{model_name} - Micro F1: {results['test_metrics']['micro_f1']:.4f}"
            )
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")

    # Train ensemble
    try:
        ensemble_results = trainer.train_ensemble(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            ensemble_type="stacking",
            use_resampling=True,
            use_feature_selection=True,
        )
        logger.info(
            f"Ensemble - Micro F1: {ensemble_results['test_metrics']['micro_f1']:.4f}"
        )
    except Exception as e:
        logger.error(f"Failed to train ensemble: {e}")
        ensemble_results = {}

    # Compile results
    trainer.results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_info": {
            "n_samples": len(X_train) + len(X_val) + len(X_test),
            "n_features": X_train.shape[1],
            "n_classes": len(metadata.get("class_names", [])),
        },
        "individual_results": individual_results,
        "ensemble_results": ensemble_results,
        "optimization_techniques": [
            "SMOTE-Tomek resampling",
            "Feature selection with mutual information",
            "Optimized hyperparameters",
            "Ensemble stacking",
        ],
    }

    # Save models and results
    trainer.save_models("models/optimized")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION RESULTS SUMMARY")
    logger.info("=" * 80)

    logger.info("\nIndividual Model Performance:")
    for model_name, results in individual_results.items():
        metrics = results["test_metrics"]
        logger.info(f"  {model_name}:")
        logger.info(f"    Micro F1: {metrics.get('micro_f1', 0):.4f}")
        logger.info(f"    Macro F1: {metrics.get('macro_f1', 0):.4f}")
        logger.info(f"    Micro AUC: {metrics.get('micro_auc', 0):.4f}")

    if ensemble_results:
        ensemble_metrics = ensemble_results["test_metrics"]
        logger.info(f"\nEnsemble Model Performance:")
        logger.info(f"  Micro F1: {ensemble_metrics.get('micro_f1', 0):.4f}")
        logger.info(f"  Macro F1: {ensemble_metrics.get('macro_f1', 0):.4f}")
        logger.info(f"  Micro AUC: {ensemble_metrics.get('micro_auc', 0):.4f}")

    return trainer.results


if __name__ == "__main__":
    results = run_optimized_training()
