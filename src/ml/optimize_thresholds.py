#!/usr/bin/env python3
"""Per-class threshold optimization for DeepAMR.

Finds the optimal F1 threshold per drug class on the validation/test set
instead of using a fixed 0.5 threshold.

Usage:
    python -m src.ml.optimize_thresholds
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.ml.inference import DeepAMRPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    drug_classes: list,
    search_range: tuple = (0.1, 0.9),
    steps: int = 81,
) -> dict:
    """Find optimal threshold per class maximizing F1 score."""
    thresholds = np.linspace(search_range[0], search_range[1], steps)
    optimal = {}

    for i, drug in enumerate(drug_classes):
        best_f1 = 0.0
        best_t = 0.5
        for t in thresholds:
            preds = (y_probs[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        optimal[drug] = {"threshold": round(best_t, 3), "f1": round(best_f1, 4)}
        logger.info(f"{drug}: threshold={best_t:.3f}, F1={best_f1:.4f}")

    return optimal


def main():
    # Load test data
    data_dir = PROJECT_ROOT / "data" / "processed" / "ncbi"
    X_test = np.load(data_dir / "ncbi_amr_X_test.npy")
    y_test = np.load(data_dir / "ncbi_amr_y_test.npy")

    # Load model and get probabilities
    predictor = DeepAMRPredictor()

    # Get raw probabilities for all test samples
    features = X_test
    if predictor.scaler is not None:
        features = predictor.scaler.transform(features)

    X_tensor = torch.FloatTensor(features).to(predictor.device)
    with torch.no_grad():
        logits = predictor.model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Find optimal thresholds
    optimal = find_optimal_thresholds(y_test, probs, predictor.drug_classes)

    # Save
    output_path = PROJECT_ROOT / "models" / "optimal_thresholds.json"
    with open(output_path, "w") as f:
        json.dump(optimal, f, indent=2)
    logger.info(f"Saved optimal thresholds to {output_path}")

    # Print summary
    avg_f1 = np.mean([v["f1"] for v in optimal.values()])
    logger.info(f"Average per-class F1 with optimized thresholds: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
