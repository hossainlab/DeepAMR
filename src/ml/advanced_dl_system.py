#!/usr/bin/env python3
"""Advanced Deep Learning System for Multi-Source AMR Prediction.

This system integrates data from NCBI, CARD, PATRIC, and ResFinder to build
a robust, high-accuracy deep learning model for AMR prediction.

Architecture features:
1. Multi-Head Attention for k-mer importance weighting
2. Deep Residual Blocks for feature extraction
3. Focal Loss to handle extreme class imbalance
4. Multi-dataset alignment and normalization
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    hamming_loss,
    precision_recall_curve,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# =============================================================================
# 1. Data Integration Engine
# =============================================================================


class MultiSourceDataLoader:
    """Aligns and merges datasets from different sources."""

    def __init__(self, data_root: str = "data/processed"):
        self.data_root = Path(data_root)
        self.sources = ["ncbi", "card", "patric", "resfinder"]
        self.unified_drug_classes = [
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
        self.scaler = StandardScaler()

    def load_and_align(self) -> Dict:
        """Loads all available datasets and aligns them to a unified label space."""
        logger.info("Aligning multi-source datasets...")

        merged_data = {split: {"X": [], "y": []} for split in ["train", "val", "test"]}

        for source in self.sources:
            source_dir = self.data_root / source
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue

            # Find all metadata files in the source directory
            for meta_path in source_dir.glob("*_metadata.json"):
                prefix = meta_path.name.replace("_metadata.json", "")

                with open(meta_path) as f:
                    meta = json.load(f)

                # Skip if not AMR drug class target
                if meta.get("target") not in [
                    "amr_drug_class",
                    "drug_class",
                    "multilabel",
                ]:
                    # Some datasets might not have 'target' key but still be valid
                    if "drug_classes" not in meta and "class_names" not in meta:
                        continue

                source_classes = meta.get("class_names", []) or meta.get(
                    "drug_classes", []
                )
                class_map = {
                    cls: self.unified_drug_classes.index(cls)
                    for cls in source_classes
                    if cls in self.unified_drug_classes
                }

                if not class_map:
                    continue

                logger.info(
                    f"Merging {source} ({prefix}) with {len(class_map)} matching classes"
                )

                for split in ["train", "val", "test"]:
                    x_path = source_dir / f"{prefix}_X_{split}.npy"
                    y_path = source_dir / f"{prefix}_y_{split}.npy"

                    if not (x_path.exists() and y_path.exists()):
                        continue

                    X = np.load(x_path)
                    y_orig = np.load(y_path)

                    # Align Y to unified label space
                    y_aligned = np.zeros((len(y_orig), len(self.unified_drug_classes)))

                    if y_orig.ndim == 1:
                        # Multiclass (one label per sample) - convert to multi-label format
                        for i, label_idx in enumerate(y_orig):
                            if label_idx < len(source_classes):
                                cls = source_classes[label_idx]
                                if cls in class_map:
                                    y_aligned[i, class_map[cls]] = 1
                    else:
                        # Already multi-label
                        for old_idx, cls in enumerate(source_classes):
                            if cls in class_map:
                                new_idx = class_map[cls]
                                y_aligned[:, new_idx] = y_orig[:, old_idx]

                    merged_data[split]["X"].append(X)
                    merged_data[split]["y"].append(y_aligned)

        # Concatenate all sources
        final_data = {}
        for split in ["train", "val", "test"]:
            if not merged_data[split]["X"]:
                raise RuntimeError(f"No data found for {split} split")

            final_data[f"X_{split}"] = np.vstack(merged_data[split]["X"])
            final_data[f"y_{split}"] = np.vstack(merged_data[split]["y"])

        # Scale features
        final_data["X_train"] = self.scaler.fit_transform(final_data["X_train"])
        final_data["X_val"] = self.scaler.transform(final_data["X_val"])
        final_data["X_test"] = self.scaler.transform(final_data["X_test"])

        logger.info(
            f"Unified dataset created: {len(final_data['X_train'])} training samples"
        )
        return final_data


# =============================================================================
# 2. Advanced Neural Architecture
# =============================================================================


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model) - here seq_len=1 for k-mer vector
        # Treat as (batch, d_model) -> (batch, 1, d_model)
        x = x.unsqueeze(1)
        batch_size = x.size(0)

        q = (
            self.q_linear(x)
            .view(batch_size, -1, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(x)
            .view(batch_size, -1, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(x)
            .view(batch_size, -1, self.n_heads, self.d_head)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)

        context = (
            torch.matmul(attn, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_head)
        )
        return self.out_linear(context).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
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

    def forward(self, x):
        return self.gelu(x + self.net(x))


class AdvancedDeepAMR(nn.Module):
    """Advanced Deep Learning Model for AMR Prediction."""

    def __init__(self, input_dim, output_dim, hidden_dim=512, n_blocks=4):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()
        )

        self.attention = MultiHeadAttention(hidden_dim, n_heads=8)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.attention(x)
        for block in self.res_blocks:
            x = block(x)
        return self.classifier(x)


# =============================================================================
# 3. Training Logic (Focal Loss + Weighted Sampling)
# =============================================================================


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def train_advanced_system():
    logger.info("Starting Advanced DL System Training...")

    # 1. Load and Merge Data
    loader = MultiSourceDataLoader()
    data = loader.load_and_align()

    X_train, y_train = (
        torch.FloatTensor(data["X_train"]),
        torch.FloatTensor(data["y_train"]),
    )
    X_val, y_val = torch.FloatTensor(data["X_val"]), torch.FloatTensor(data["y_val"])
    X_test, y_test = (
        torch.FloatTensor(data["X_test"]),
        torch.FloatTensor(data["y_test"]),
    )

    train_ds = TensorDataset(X_train, y_train)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    # Handle multi-label imbalance via batch sampling
    # (Simplified: using random shuffle, focusing on Focal Loss for imbalance)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # 2. Build Model
    model = AdvancedDeepAMR(input_dim=X_train.shape[1], output_dim=y_train.shape[1]).to(
        DEVICE
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = FocalLoss(gamma=2.5)  # High gamma for extreme imbalance

    best_f1 = 0
    epochs = 100
    patience = 15
    counter = 0

    logger.info(f"Model Training on {DEVICE}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                logits = model(batch_X)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(batch_y.numpy())

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        val_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch + 1:03d} | Train Loss: {total_loss / len(train_loader):.4f} | Val F1: {val_f1:.4f}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler": loader.scaler,
                    "classes": loader.unified_drug_classes,
                    "input_dim": X_train.shape[1],
                    "output_dim": y_train.shape[1],
                    "hidden_dim": 512,
                    "n_blocks": 4,
                    "best_val_f1": best_f1,
                    "epoch": epoch + 1,
                },
                "models/advanced_deepamr_system.pt",
            )
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Final Test Evaluation
    checkpoint = torch.load("models/advanced_deepamr_system.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            logits = model(batch_X)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.append(probs)
            test_targets.append(batch_y.numpy())

    y_prob = np.vstack(test_preds)
    y_pred = (y_prob > 0.5).astype(int)
    y_true = np.vstack(test_targets)

    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "micro_auc": roc_auc_score(y_true, y_prob, average="micro"),
    }

    logger.info("=" * 60)
    logger.info("FINAL SYSTEM PERFORMANCE (Unified Multi-Source)")
    logger.info(f"Micro F1: {metrics['micro_f1']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Micro AUC: {metrics['micro_auc']:.4f}")
    logger.info(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    logger.info("=" * 60)

    # Save detailed results
    with open("models/advanced_system_results.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    train_advanced_system()
