#!/usr/bin/env python3
"""
DeepAMR: Deep Learning Models for Antimicrobial Resistance Prediction

This script trains deep learning models for:
1. Organism Classification (multiclass)
2. AMR Drug Resistance Prediction (multilabel)

Designed for high-impact deployment in Bangladesh healthcare systems.

Usage:
    python src/ml/deep_learning_trainer.py --task organism
    python src/ml/deep_learning_trainer.py --task amr
    python src/ml/deep_learning_trainer.py --task both
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


# =============================================================================
# Neural Network Architectures
# =============================================================================

class OrganismClassifier(nn.Module):
    """Deep neural network for organism classification from k-mer features."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [256, 128, 64],
        num_classes: int = 8,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AMRPredictor(nn.Module):
    """Deep neural network for multi-label AMR prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [512, 256, 128],
        num_classes: int = 11,
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        # Shared feature extractor
        shared_layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes[:-1]):
            shared_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size

        self.shared = nn.Sequential(*shared_layers)

        # Drug-class specific heads for better performance
        self.drug_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_size, hidden_sizes[-1]),
                nn.BatchNorm1d(hidden_sizes[-1]),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(hidden_sizes[-1], 1),
            )
            for _ in range(num_classes)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared(x)
        outputs = [head(shared_features) for head in self.drug_heads]
        return torch.cat(outputs, dim=1)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, size: int, dropout_rate: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class DeepAMRNet(nn.Module):
    """Advanced deep network with residual connections for AMR prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_residual_blocks: int = 3,
        num_classes: int = 11,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, dropout_rate) for _ in range(num_residual_blocks)]
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        return self.output_layer(x)


# =============================================================================
# Training Utilities
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in multilabel classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def compute_class_weights(y: np.ndarray, task: str = 'multiclass') -> torch.Tensor:
    """Compute class weights to handle imbalanced data."""
    if task == 'multiclass':
        class_counts = np.bincount(y)
        total = len(y)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)
    else:  # multilabel
        pos_counts = y.sum(axis=0)
        neg_counts = len(y) - pos_counts
        weights = neg_counts / (pos_counts + 1e-6)
        weights = np.clip(weights, 1.0, 10.0)  # Clip extreme weights
        return torch.FloatTensor(weights)


# =============================================================================
# Trainer Classes
# =============================================================================

class BaseTrainer:
    """Base trainer class with common functionality."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: torch.device = DEVICE,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['metrics']


class OrganismTrainer(BaseTrainer):
    """Trainer for organism classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(model, train_loader, val_loader, test_loader, **kwargs)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def train_epoch(self) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        }

        return avg_loss, metrics

    def validate(self, loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_probs = np.array(all_probs)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'precision_macro': precision_score(all_labels, all_preds, average='macro'),
            'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        }

        # ROC-AUC for multiclass
        try:
            metrics['roc_auc'] = roc_auc_score(
                all_labels, all_probs, multi_class='ovr', average='macro'
            )
        except ValueError:
            metrics['roc_auc'] = 0.0

        return avg_loss, metrics, all_preds, all_labels, all_probs

    def train(
        self,
        epochs: int = 100,
        patience: int = 15,
        save_path: str = 'models/organism_classifier.pt',
    ) -> Dict:
        """Full training loop."""
        early_stopping = EarlyStopping(patience=patience, mode='max')
        best_f1 = 0

        logger.info("Starting organism classification training...")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(epochs):
            # Train
            train_loss, train_metrics = self.train_epoch()

            # Validate
            val_loss, val_metrics, _, _, _ = self.validate(self.val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )

            # Save best model
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(save_path, epoch, val_metrics)
                logger.info(f"New best model saved! F1: {best_f1:.4f}")

            # Early stopping
            if early_stopping(val_metrics['f1_macro']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Final evaluation on test set
        logger.info("\nEvaluating on test set...")
        test_loss, test_metrics, test_preds, test_labels, test_probs = self.validate(
            self.test_loader
        )

        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")

        return {
            'history': self.history,
            'test_metrics': test_metrics,
            'test_predictions': test_preds,
            'test_labels': test_labels,
            'test_probabilities': test_probs,
        }


class AMRTrainer(BaseTrainer):
    """Trainer for multilabel AMR prediction."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        pos_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        **kwargs,
    ):
        super().__init__(model, train_loader, val_loader, test_loader, **kwargs)

        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            if pos_weights is not None:
                pos_weights = pos_weights.to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def train_epoch(self) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.float().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            'f1_micro': f1_score(all_labels, all_preds, average='micro'),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_samples': f1_score(all_labels, all_preds, average='samples'),
        }

        return avg_loss, metrics

    def validate(self, loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = {
            'f1_micro': f1_score(all_labels, all_preds, average='micro'),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_samples': f1_score(all_labels, all_preds, average='samples'),
            'precision_micro': precision_score(all_labels, all_preds, average='micro'),
            'recall_micro': recall_score(all_labels, all_preds, average='micro'),
        }

        # Per-class metrics
        per_class_f1 = f1_score(all_labels, all_preds, average=None)
        metrics['per_class_f1'] = per_class_f1.tolist()

        # ROC-AUC
        try:
            metrics['roc_auc_micro'] = roc_auc_score(all_labels, all_probs, average='micro')
            metrics['roc_auc_macro'] = roc_auc_score(all_labels, all_probs, average='macro')
        except ValueError:
            metrics['roc_auc_micro'] = 0.0
            metrics['roc_auc_macro'] = 0.0

        return avg_loss, metrics, all_preds, all_labels, all_probs

    def train(
        self,
        epochs: int = 100,
        patience: int = 15,
        save_path: str = 'models/amr_predictor.pt',
    ) -> Dict:
        """Full training loop."""
        early_stopping = EarlyStopping(patience=patience, mode='max')
        best_f1 = 0

        logger.info("Starting AMR prediction training...")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(epochs):
            # Train
            train_loss, train_metrics = self.train_epoch()

            # Validate
            val_loss, val_metrics, _, _, _ = self.validate(self.val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train F1: {train_metrics['f1_macro']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f} | "
                f"Val AUC: {val_metrics.get('roc_auc_macro', 0):.4f}"
            )

            # Save best model
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(save_path, epoch, val_metrics)
                logger.info(f"New best model saved! F1: {best_f1:.4f}")

            # Early stopping
            if early_stopping(val_metrics['f1_macro']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Final evaluation on test set
        logger.info("\nEvaluating on test set...")
        test_loss, test_metrics, test_preds, test_labels, test_probs = self.validate(
            self.test_loader
        )

        logger.info(f"\nTest Results:")
        logger.info(f"  F1 (micro): {test_metrics['f1_micro']:.4f}")
        logger.info(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (samples): {test_metrics['f1_samples']:.4f}")
        logger.info(f"  ROC-AUC (macro): {test_metrics['roc_auc_macro']:.4f}")

        return {
            'history': self.history,
            'test_metrics': test_metrics,
            'test_predictions': test_preds,
            'test_labels': test_labels,
            'test_probabilities': test_probs,
        }


# =============================================================================
# Data Loading
# =============================================================================

def load_data(task: str = 'organism') -> Tuple:
    """Load preprocessed data for training."""
    data_dir = Path('data/processed/ncbi')

    if task == 'organism':
        prefix = 'ncbi_organism'
    else:
        prefix = 'ncbi_amr'

    X_train = np.load(data_dir / f'{prefix}_X_train.npy')
    X_val = np.load(data_dir / f'{prefix}_X_val.npy')
    X_test = np.load(data_dir / f'{prefix}_X_test.npy')
    y_train = np.load(data_dir / f'{prefix}_y_train.npy')
    y_val = np.load(data_dir / f'{prefix}_y_val.npy')
    y_test = np.load(data_dir / f'{prefix}_y_test.npy')

    with open(data_dir / f'{prefix}_metadata.json') as f:
        metadata = json.load(f)

    logger.info(f"Loaded {task} data:")
    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, metadata


def create_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[StandardScaler]]:
    """Create PyTorch DataLoaders."""

    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train) if y_train.ndim == 1 else torch.FloatTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val) if y_val.ndim == 1 else torch.FloatTensor(y_val),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test) if y_test.ndim == 1 else torch.FloatTensor(y_test),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler


# =============================================================================
# Main Training Functions
# =============================================================================

def train_organism_classifier(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_sizes: List[int] = [256, 128, 64],
    dropout_rate: float = 0.3,
    save_dir: str = 'models',
) -> Dict:
    """Train organism classification model."""

    logger.info("=" * 60)
    logger.info("ORGANISM CLASSIFICATION TRAINING")
    logger.info("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_data('organism')

    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )

    # Compute class weights
    class_weights = compute_class_weights(y_train, 'multiclass')

    # Create model
    model = OrganismClassifier(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        num_classes=len(metadata['class_names']),
        dropout_rate=dropout_rate,
    )

    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = OrganismTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        learning_rate=learning_rate,
    )

    # Train
    save_path = Path(save_dir) / 'organism_classifier.pt'
    results = trainer.train(epochs=epochs, save_path=str(save_path))

    # Save scaler
    if scaler is not None:
        import joblib
        scaler_path = Path(save_dir) / 'organism_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    # Save metadata and results
    results_path = Path(save_dir) / 'organism_results.json'
    save_results = {
        'metadata': metadata,
        'test_metrics': results['test_metrics'],
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_sizes': hidden_sizes,
            'dropout_rate': dropout_rate,
        },
    }
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    # Save training history
    history_path = Path(save_dir) / 'organism_history.json'
    history_save = {
        'train_loss': results['history']['train_loss'],
        'val_loss': results['history']['val_loss'],
        'train_metrics': results['history']['train_metrics'],
        'val_metrics': results['history']['val_metrics'],
        'learning_rates': results['history']['learning_rates'],
    }
    with open(history_path, 'w') as f:
        json.dump(history_save, f, indent=2)

    logger.info(f"\nResults saved to {save_dir}")

    return results


def train_amr_predictor(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_sizes: List[int] = [512, 256, 128],
    dropout_rate: float = 0.4,
    use_focal_loss: bool = True,
    save_dir: str = 'models',
) -> Dict:
    """Train AMR prediction model."""

    logger.info("=" * 60)
    logger.info("AMR PREDICTION TRAINING")
    logger.info("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_data('amr')

    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )

    # Compute positive weights for class imbalance
    pos_weights = compute_class_weights(y_train, 'multilabel')

    # Create model
    model = AMRPredictor(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        num_classes=len(metadata['class_names']),
        dropout_rate=dropout_rate,
    )

    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = AMRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        pos_weights=pos_weights,
        use_focal_loss=use_focal_loss,
        learning_rate=learning_rate,
    )

    # Train
    save_path = Path(save_dir) / 'amr_predictor.pt'
    results = trainer.train(epochs=epochs, save_path=str(save_path))

    # Save scaler
    if scaler is not None:
        import joblib
        scaler_path = Path(save_dir) / 'amr_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    # Save metadata and results
    results_path = Path(save_dir) / 'amr_results.json'
    save_results = {
        'metadata': metadata,
        'test_metrics': {k: v if not isinstance(v, np.ndarray) else v.tolist()
                        for k, v in results['test_metrics'].items()},
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_sizes': hidden_sizes,
            'dropout_rate': dropout_rate,
            'use_focal_loss': use_focal_loss,
        },
    }
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    # Save training history
    history_path = Path(save_dir) / 'amr_history.json'
    history_save = {
        'train_loss': results['history']['train_loss'],
        'val_loss': results['history']['val_loss'],
        'train_metrics': results['history']['train_metrics'],
        'val_metrics': [{k: v if not isinstance(v, list) else v
                        for k, v in m.items()} for m in results['history']['val_metrics']],
        'learning_rates': results['history']['learning_rates'],
    }
    with open(history_path, 'w') as f:
        json.dump(history_save, f, indent=2)

    logger.info(f"\nResults saved to {save_dir}")

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train deep learning models for AMR prediction'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['organism', 'amr', 'both'],
        default='both',
        help='Task to train: organism, amr, or both',
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models', help='Save directory')

    args = parser.parse_args()

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Training timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"Training started at {timestamp}")

    results = {}

    if args.task in ['organism', 'both']:
        results['organism'] = train_organism_classifier(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_dir=args.save_dir,
        )

    if args.task in ['amr', 'both']:
        results['amr'] = train_amr_predictor(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_dir=args.save_dir,
        )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    if 'organism' in results:
        logger.info(f"\nOrganism Classification Test Accuracy: "
                   f"{results['organism']['test_metrics']['accuracy']:.4f}")

    if 'amr' in results:
        logger.info(f"\nAMR Prediction Test F1 (macro): "
                   f"{results['amr']['test_metrics']['f1_macro']:.4f}")

    logger.info(f"\nModels saved to: {args.save_dir}/")


if __name__ == '__main__':
    main()
