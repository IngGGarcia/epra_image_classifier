"""
Violence classification trainer.

This module provides specialized training logic for violence classification
tasks with support for advanced techniques like MixUp, CutMix, and
class balancing.
"""

import time
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from ...data.augmentation import AugmentationMixer
from ...utils.config import Config
from ...utils.helpers import calculate_metrics
from .base_trainer import BaseTrainer


class ViolenceTrainer(BaseTrainer):
    """
    Specialized trainer for violence classification tasks.

    This trainer includes violence-specific training techniques and metrics.
    """

    def __init__(self, model, config: Config):
        """
        Initialize violence trainer.

        Args:
            model: Model to train
            config: Configuration object
        """
        super().__init__(model, config)

        # Initialize augmentation mixer for advanced training
        self.augmentation_mixer = AugmentationMixer(
            mixup_alpha=0.2, cutmix_alpha=1.0, mixup_prob=0.3, cutmix_prob=0.3
        )

        # Class weights for handling imbalance
        self.class_weights = None

        # Training components
        self.device = next(model.parameters()).device
        self.criterion = self._build_criterion()
        self.scaler = self._build_scaler()

        self._setup_class_weights()

    def _build_criterion(self):
        """Build loss criterion."""
        import torch.nn as nn

        return nn.CrossEntropyLoss()

    def _build_scaler(self):
        """Build gradient scaler for mixed precision training."""
        import torch.cuda.amp as amp

        return amp.GradScaler() if self.device.type == "cuda" else None

    def to_device(self, tensor):
        """Move tensor to device."""
        return tensor.to(self.device)

    def set_mode(self, mode: str):
        """Set model mode."""
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()

    def _setup_class_weights(self) -> None:
        """Setup class weights for handling class imbalance."""
        # Class weights will be calculated from dataset if needed
        if hasattr(self.criterion, "weight") and self.criterion.weight is None:
            self.logger.info("Class weights will be calculated from training data")

    def _is_loss_metric(self) -> bool:
        """Loss is the primary metric (lower is better)."""
        return True

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """
        Train for one epoch with violence-specific techniques.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.set_mode("train")

        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        # Calculate class weights from this epoch's data if needed
        if self.class_weights is None:
            self.class_weights = self._calculate_class_weights(train_loader)

        for batch_idx, (images, targets) in enumerate(train_loader):
            batch_start = time.time()

            # Move to device
            images = self.to_device(images)
            targets = self.to_device(targets)

            # Log progress every 50 batches
            if batch_idx % 50 == 0:
                progress = (batch_idx / len(train_loader)) * 100
                self.logger.info(
                    f"Epoch Progress: {progress:.1f}% - Batch {batch_idx}/{len(train_loader)}"
                )

            # Apply advanced augmentations (MixUp/CutMix)
            mixed_images, targets_a, targets_b, lam, method = self.augmentation_mixer(
                images, targets
            )

            # Forward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(mixed_images)
                    logits = outputs["logits"]

                    # Calculate loss with augmentation support
                    loss = self.augmentation_mixer.loss_fn(
                        self.criterion,
                        logits,
                        targets_a,
                        targets_b,
                        lam,
                        method,
                    )

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping if specified
                if self.config.training.gradient_clip_value is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_value,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard training
                outputs = self.model(mixed_images)
                logits = outputs["logits"]

                # Calculate loss with augmentation support
                loss = self.augmentation_mixer.loss_fn(
                    self.criterion, logits, targets_a, targets_b, lam, method
                )

                # Backward pass
                loss.backward()

                # Gradient clipping if specified
                if self.config.training.gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_value,
                    )

                self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # For accuracy calculation (use original targets)
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            # Log batch progress
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                batch_time = time.time() - batch_start
                current_avg_loss = total_loss / max(total_samples, 1)
                self.logger.info(
                    f"Batch {batch_idx:4d}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} (Avg: {current_avg_loss:.4f}) | "
                    f"Aug: {method} | Time: {batch_time:.3f}s"
                )

        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_targets, all_predictions)

        # Calculate additional metrics
        if self.config.model.num_classes == 2:
            # Binary metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average="binary"
            )
        else:
            # Multiclass metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average="weighted"
            )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def validate_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.set_mode("eval")

        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for images, targets in val_loader:
                # Move to device
                images = self.to_device(images)
                targets = self.to_device(targets)

                # Forward pass
                outputs = self.model(images)
                logits = outputs["logits"]

                # Calculate loss
                loss = self.criterion(logits, targets)

                # Accumulate metrics
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # Get predictions and probabilities
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_targets, all_predictions)

        # Calculate additional metrics
        if self.config.model.num_classes == 2:
            # Binary metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average="binary"
            )

            # Violence-specific metrics
            violence_predictions = [p for p, t in zip(all_predictions, all_targets)]
            violence_targets = all_targets

            # Violence detection rate (recall for violence class)
            violence_recall = recall if len(set(all_targets)) > 1 else 0.0

        else:
            # Multiclass metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average="weighted"
            )
            violence_recall = recall  # Overall recall for multiclass

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "violence_recall": violence_recall,
        }

        # Add class-specific metrics for multiclass
        if self.config.model.num_classes > 2:
            class_precision, class_recall, class_f1, _ = (
                precision_recall_fscore_support(
                    all_targets, all_predictions, average=None
                )
            )

            for i in range(self.config.model.num_classes):
                metrics[f"class_{i}_precision"] = (
                    class_precision[i] if i < len(class_precision) else 0.0
                )
                metrics[f"class_{i}_recall"] = (
                    class_recall[i] if i < len(class_recall) else 0.0
                )
                metrics[f"class_{i}_f1"] = class_f1[i] if i < len(class_f1) else 0.0

        return metrics

    def _calculate_class_weights(self, train_loader: DataLoader) -> torch.Tensor | None:
        """Calculate class weights from training data."""
        if not hasattr(train_loader.dataset, "get_class_counts"):
            return None

        try:
            class_counts = train_loader.dataset.get_class_counts()
            total_samples = sum(class_counts.values())
            num_classes = len(class_counts)

            weights = []
            for i in range(num_classes):
                class_name = (
                    list(class_counts.keys())[i]
                    if i < len(class_counts)
                    else f"class_{i}"
                )
                count = class_counts.get(class_name, 1)
                weight = total_samples / (num_classes * count)
                weights.append(weight)

            class_weights = torch.tensor(
                weights, dtype=torch.float32, device=self.device
            )

            # Update criterion with class weights
            if hasattr(self.criterion, "weight"):
                self.criterion.weight = class_weights
                self.logger.info(f"Applied class weights: {class_weights.tolist()}")

            return class_weights

        except Exception as e:
            self.logger.warning(f"Could not calculate class weights: {e}")
            return None

    def evaluate_violence_detection(self, test_loader: DataLoader) -> dict[str, Any]:
        """
        Comprehensive evaluation for violence detection.

        Args:
            test_loader: Test data loader

        Returns:
            Detailed evaluation results
        """
        self.set_mode("eval")

        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_features = []

        # Enable feature extraction
        self.model.return_features = True

        try:
            with torch.no_grad():
                for images, targets in test_loader:
                    images = self.to_device(images)
                    targets = self.to_device(targets)

                    outputs = self.model(images)
                    logits = outputs["logits"]

                    probabilities = F.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)

                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                    if "features" in outputs:
                        all_features.extend(outputs["features"].cpu().numpy())

        finally:
            self.model.return_features = False

        # Calculate comprehensive metrics
        results = calculate_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets),
            self.config.model.num_classes,
        )

        # Add violence-specific analysis
        if self.config.model.num_classes == 2:
            # Binary violence detection analysis
            violence_indices = [
                i for i, target in enumerate(all_targets) if target == 1
            ]
            non_violence_indices = [
                i for i, target in enumerate(all_targets) if target == 0
            ]

            if violence_indices and non_violence_indices:
                violence_probs = [all_probabilities[i][1] for i in violence_indices]
                non_violence_probs = [
                    all_probabilities[i][1] for i in non_violence_indices
                ]

                results.update(
                    {
                        "violence_detection_rate": results["recall"],
                        "false_positive_rate": 1 - results["precision"],
                        "avg_violence_confidence": sum(violence_probs)
                        / len(violence_probs),
                        "avg_non_violence_confidence": 1
                        - (sum(non_violence_probs) / len(non_violence_probs)),
                    }
                )

        # Add feature analysis if available
        if all_features:
            import numpy as np

            features_array = np.array(all_features)
            results["feature_statistics"] = {
                "mean_activation": float(np.mean(features_array)),
                "std_activation": float(np.std(features_array)),
                "feature_dimensionality": features_array.shape[1],
            }

        return results
