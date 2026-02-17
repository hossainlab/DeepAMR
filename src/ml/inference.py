#!/usr/bin/env python3
"""Production-Ready Inference Module for DeepAMR.

This module provides a clean API for making predictions with trained models,
designed for integration with web frontends and APIs.

Usage:
    from src.ml.inference import DeepAMRPredictor

    predictor = DeepAMRPredictor()
    results = predictor.predict(features)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Neural Network Architectures (must match training)
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention for feature importance weighting."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        batch_size = x.size(0)

        q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.n_heads * self.d_head)
        return self.out_linear(context).squeeze(1)


class ResidualBlock(nn.Module):
    """Residual block with GELU activation."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x + self.net(x))


class AdvancedDeepAMR(nn.Module):
    """Advanced Deep Learning Model for AMR Prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 4,
    ):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.attention = MultiHeadAttention(hidden_dim, n_heads=8)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x + self.attention(x)
        for block in self.res_blocks:
            x = block(x)
        return self.classifier(x)


# =============================================================================
# Production Predictor Class
# =============================================================================

class DeepAMRPredictor:
    """Production-ready predictor for AMR resistance.

    This class provides a clean interface for making predictions with
    trained DeepAMR models, suitable for frontend/API integration.

    Attributes:
        model: The loaded PyTorch model
        scaler: Feature scaler for preprocessing
        drug_classes: List of drug class names
        device: Computation device (cuda/mps/cpu)

    Example:
        >>> predictor = DeepAMRPredictor()
        >>> results = predictor.predict(kmer_features)
        >>> print(results['predictions'])
        {'aminoglycoside': True, 'beta-lactam': False, ...}
    """

    # Default drug classes
    DEFAULT_DRUG_CLASSES = [
        "aminoglycoside",
        "beta-lactam",
        "fosfomycin",
        "glycopeptide",
        "macrolide",
        "phenicol",
        "quinolone",
        "rifampicin",
        "sulfonamide",
        "tetracycline",
        "trimethoprim",
    ]

    def __init__(
        self,
        model_path: str = "models/advanced_deepamr_system.pt",
        device: str = "auto",
    ):
        """Initialize the predictor.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
        """
        self.model_path = Path(model_path)

        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the trained model and preprocessing components."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Extract components
        self.scaler = checkpoint.get("scaler")
        self.drug_classes = checkpoint.get("classes", self.DEFAULT_DRUG_CLASSES)

        # Determine model dimensions from checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Infer dimensions from state dict
        input_dim = state_dict["embedding.0.weight"].shape[1]
        output_dim = state_dict["classifier.3.weight"].shape[0]
        hidden_dim = state_dict["embedding.0.weight"].shape[0]

        # Create model architecture
        self.model = AdvancedDeepAMR(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.eval()

        logger.info(f"Model loaded successfully. Drug classes: {len(self.drug_classes)}")

    def predict(
        self,
        features: Union[np.ndarray, List[List[float]]],
        threshold: float = 0.5,
        return_probabilities: bool = True,
    ) -> Dict:
        """Make AMR resistance predictions.

        Args:
            features: Input features (k-mer frequencies). Shape: (n_samples, n_features)
                     or (n_features,) for single sample
            threshold: Probability threshold for positive prediction (default: 0.5)
            return_probabilities: Whether to include probability scores

        Returns:
            Dictionary containing:
                - predictions: Dict mapping drug class to resistance status
                - probabilities: Dict mapping drug class to probability (if requested)
                - resistant_count: Number of drug classes with predicted resistance
                - susceptible_count: Number of drug classes predicted susceptible
        """
        # Convert to numpy if needed
        if isinstance(features, list):
            features = np.array(features)

        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Convert to tensor
        X = torch.FloatTensor(features).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(X)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        # Process results
        results = []
        for i in range(len(probabilities)):
            probs = probabilities[i]
            preds = (probs > threshold).astype(int)

            result = {
                "predictions": {
                    drug: bool(preds[j]) for j, drug in enumerate(self.drug_classes)
                },
                "resistant_count": int(preds.sum()),
                "susceptible_count": int(len(self.drug_classes) - preds.sum()),
            }

            if return_probabilities:
                result["probabilities"] = {
                    drug: float(probs[j]) for j, drug in enumerate(self.drug_classes)
                }

            results.append(result)

        # Return single result if single input
        return results[0] if len(results) == 1 else results

    def predict_batch(
        self,
        features_list: List[np.ndarray],
        threshold: float = 0.5,
        batch_size: int = 32,
    ) -> List[Dict]:
        """Make predictions on a batch of samples efficiently.

        Args:
            features_list: List of feature arrays
            threshold: Probability threshold for positive prediction
            batch_size: Processing batch size

        Returns:
            List of prediction dictionaries
        """
        all_results = []

        # Process in batches
        for i in range(0, len(features_list), batch_size):
            batch = np.array(features_list[i:i + batch_size])
            batch_results = self.predict(batch, threshold=threshold)

            if isinstance(batch_results, dict):
                all_results.append(batch_results)
            else:
                all_results.extend(batch_results)

        return all_results

    def get_resistance_summary(self, predictions: Dict) -> Dict:
        """Generate a human-readable summary of predictions.

        Args:
            predictions: Output from predict()

        Returns:
            Summary dictionary with formatted results
        """
        resistant_drugs = [
            drug for drug, status in predictions["predictions"].items() if status
        ]
        susceptible_drugs = [
            drug for drug, status in predictions["predictions"].items() if not status
        ]

        # Risk level assessment
        if predictions["resistant_count"] >= 5:
            risk_level = "HIGH"
            risk_description = "Multi-drug resistant (MDR) - Requires specialist consultation"
        elif predictions["resistant_count"] >= 3:
            risk_level = "MODERATE"
            risk_description = "Multiple resistance detected - Consider alternative treatments"
        elif predictions["resistant_count"] >= 1:
            risk_level = "LOW"
            risk_description = "Limited resistance - Standard alternatives available"
        else:
            risk_level = "MINIMAL"
            risk_description = "No predicted resistance - Standard treatment likely effective"

        summary = {
            "risk_level": risk_level,
            "risk_description": risk_description,
            "resistant_drugs": resistant_drugs,
            "susceptible_drugs": susceptible_drugs,
            "total_tested": len(self.drug_classes),
        }

        # Add probability-based confidence if available
        if "probabilities" in predictions:
            probs = predictions["probabilities"]
            high_confidence = [
                drug for drug, prob in probs.items()
                if prob > 0.8 or prob < 0.2
            ]
            summary["high_confidence_predictions"] = len(high_confidence)
            summary["average_confidence"] = np.mean([
                max(p, 1-p) for p in probs.values()
            ])

        return summary

    @property
    def model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "drug_classes": self.drug_classes,
            "n_classes": len(self.drug_classes),
            "has_scaler": self.scaler is not None,
        }


# =============================================================================
# Sklearn Model Predictor (for ensemble/traditional ML models)
# =============================================================================

class SklearnAMRPredictor:
    """Predictor for sklearn-based AMR models."""

    def __init__(
        self,
        model_path: str = "models/optimized/optimized_ensemble_stacking.joblib",
    ):
        """Initialize sklearn predictor.

        Args:
            model_path: Path to joblib model file
        """
        self.model_path = Path(model_path)
        self._load_model()

    def _load_model(self):
        """Load the sklearn model and components."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        data = joblib.load(self.model_path)

        self.model = data["model"]
        self.scaler = data.get("scaler")
        self.feature_selector = data.get("feature_selector")
        self.drug_classes = data.get("classes", DeepAMRPredictor.DEFAULT_DRUG_CLASSES)

        logger.info(f"Sklearn model loaded from {self.model_path}")

    def predict(
        self,
        features: np.ndarray,
        return_probabilities: bool = True,
    ) -> Dict:
        """Make predictions with sklearn model."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Preprocess
        if self.scaler is not None:
            features = self.scaler.transform(features)

        if self.feature_selector is not None:
            if isinstance(self.feature_selector, np.ndarray):
                features = features[:, self.feature_selector]
            else:
                features = self.feature_selector.transform(features)

        # Predict
        predictions = self.model.predict(features)

        results = []
        for i in range(len(predictions)):
            preds = predictions[i]

            result = {
                "predictions": {
                    drug: bool(preds[j]) for j, drug in enumerate(self.drug_classes)
                },
                "resistant_count": int(preds.sum()),
                "susceptible_count": int(len(self.drug_classes) - preds.sum()),
            }

            if return_probabilities and hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(features[i:i+1])[0]
                    result["probabilities"] = {
                        drug: float(probs[j]) for j, drug in enumerate(self.drug_classes)
                    }
                except Exception:
                    pass

            results.append(result)

        return results[0] if len(results) == 1 else results

    @property
    def model_info(self) -> Dict:
        return {
            "model_path": str(self.model_path),
            "device": "cpu",
            "drug_classes": self.drug_classes,
            "n_classes": len(self.drug_classes),
            "has_scaler": self.scaler is not None,
        }


# =============================================================================
# Unified Predictor Factory
# =============================================================================

def get_predictor(
    model_type: str = "deep_learning",
    model_path: Optional[str] = None,
) -> Union[DeepAMRPredictor, SklearnAMRPredictor]:
    """Factory function to get the appropriate predictor.

    Args:
        model_type: Type of model ('deep_learning', 'sklearn', 'ensemble')
        model_path: Optional custom model path

    Returns:
        Configured predictor instance
    """
    if model_type == "deep_learning":
        path = model_path or "models/advanced_deepamr_system.pt"
        return DeepAMRPredictor(path)
    elif model_type in ["sklearn", "ensemble"]:
        path = model_path or "models/optimized/optimized_ensemble_stacking.joblib"
        return SklearnAMRPredictor(path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepAMR Inference")
    parser.add_argument("--model", default="deep_learning", choices=["deep_learning", "sklearn"])
    parser.add_argument("--test", action="store_true", help="Run test prediction")

    args = parser.parse_args()

    predictor = get_predictor(args.model)
    print(f"Model info: {predictor.model_info}")

    if args.test:
        # Load test data for demo
        import numpy as np
        X_test = np.load("data/processed/ncbi/ncbi_amr_X_test.npy")

        # Predict on first sample
        result = predictor.predict(X_test[0])
        print("\nTest prediction:")
        print(f"Predictions: {result['predictions']}")
        print(f"Resistant: {result['resistant_count']}, Susceptible: {result['susceptible_count']}")

        if "probabilities" in result:
            print("\nProbabilities:")
            for drug, prob in result["probabilities"].items():
                status = "R" if prob > 0.5 else "S"
                print(f"  {drug}: {prob:.3f} ({status})")
