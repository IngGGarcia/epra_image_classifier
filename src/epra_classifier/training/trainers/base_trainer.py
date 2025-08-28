"""
Base trainer class for model training.

This module provides a unified training interface with support for
modern training techniques including mixed precision, gradient accumulation,
and advanced optimization strategies.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader

from ...utils.config import Config
from ...utils.helpers import format_time
from ...utils.logging import get_logger


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "min"):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for accuracy/metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        self.is_better = (
            lambda score, best: score < (best - min_delta)
            if mode == "min"
            else lambda score, best: score > (best + min_delta)
        )

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class BaseTrainer(ABC):
    """
    Base trainer class for model training.

    This abstract base class provides common training functionality
    including optimization, scheduling, checkpointing, and monitoring.
    """

    def __init__(self, model, config: Config):
        """
        Initialize base trainer.

        Args:
            model: Model to train
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.logger = get_logger()

        # Training state
        self.current_epoch = 0
        self.best_metric = float("inf") if self._is_loss_metric() else float("-inf")
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

        # Early stopping
        mode = "min" if self._is_loss_metric() else "max"
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience, mode=mode
        )

        # Initialize training components
        self._setup_training()

    def _setup_training(self) -> None:
        """Setup training components."""
        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Log hyperparameters
        self._log_hyperparameters()

    def _build_optimizer(self):
        """Build optimizer for training."""
        import torch.optim as optim

        if self.config.training.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        import torch.optim.lr_scheduler as lr_scheduler

        if self.config.training.scheduler == "reduce_lr_on_plateau":
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min" if self._is_loss_metric() else "max",
                factor=0.5,
                patience=5,
                verbose=True,
            )
        elif self.config.training.scheduler == "cosine_annealing":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler == "step_lr":
            return lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.num_epochs // 3,
                gamma=0.1,
            )
        else:
            return None  # No scheduler

    def _log_hyperparameters(self) -> None:
        """Log training hyperparameters."""
        hparams = {
            "learning_rate": self.config.training.learning_rate,
            "batch_size": self.config.data.batch_size,
            "num_epochs": self.config.training.num_epochs,
            "optimizer": self.config.training.optimizer,
            "scheduler": self.config.training.scheduler,
            "weight_decay": self.config.training.weight_decay,
            "dropout_rate": self.config.model.dropout_rate,
            "model_type": self.config.model.model_type,
            "num_classes": self.config.model.num_classes,
        }

        self.logger.log_hyperparameters(hparams)

    @abstractmethod
    def _is_loss_metric(self) -> bool:
        """Whether the main metric is a loss (lower is better)."""
        pass

    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def validate_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int | None = None,
    ) -> dict[str, Any]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.training.num_epochs

        self.logger.info(f"Starting training for {num_epochs} epochs")
        # Start training setup (moved to trainer)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "learning_rates": [],
            "epochs": [],
        }

        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                # Training phase
                train_metrics = self.train_epoch(train_loader)
                history["train_loss"].append(train_metrics["loss"])

                # Validation phase
                val_metrics = {}
                if val_loader is not None:
                    val_metrics = self.validate_epoch(val_loader)
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_metrics"].append(val_metrics)

                # Learning rate scheduling
                self._update_scheduler(val_metrics.get("loss"))
                current_lr = self.optimizer.param_groups[0]["lr"]
                history["learning_rates"].append(current_lr)
                history["epochs"].append(epoch)

                # Logging
                epoch_time = time.time() - epoch_start
                self._log_epoch_results(
                    epoch, train_metrics, val_metrics, epoch_time, current_lr
                )

                # Checkpointing
                is_best = self._is_best_model(val_metrics)
                if is_best:
                    self.best_metric = val_metrics.get("loss", train_metrics["loss"])

                self._save_checkpoint(epoch, val_metrics, is_best)

                # Early stopping check
                metric_for_stopping = val_metrics.get("loss", train_metrics["loss"])
                if self.early_stopping(metric_for_stopping):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        finally:
            # Training finished
            pass

        self.logger.info("Training completed")
        return history

    def _update_scheduler(self, val_loss: float | None) -> None:
        """Update learning rate scheduler."""
        if self.scheduler is None:
            return

        scheduler_name = self.config.training.scheduler.lower()

        if scheduler_name == "reduce_on_plateau" and val_loss is not None:
            self.scheduler.step(val_loss)
        elif scheduler_name in ["cosine", "step"]:
            self.scheduler.step()

    def _is_best_model(self, val_metrics: dict[str, float]) -> bool:
        """Check if current model is the best so far."""
        if not val_metrics:
            return False

        current_metric = val_metrics.get("loss")
        if current_metric is None:
            return False

        if self._is_loss_metric():
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric

    def _save_checkpoint(
        self, epoch: int, metrics: dict[str, float], is_best: bool
    ) -> None:
        """Save model checkpoint."""
        # Save checkpoint every N epochs or if it's the best
        should_save = (
            is_best
            or epoch % self.config.training.save_every_n_epochs == 0
            or epoch == self.config.training.num_epochs - 1
        )

        if should_save:
            self.save_checkpoint(epoch, metrics, is_best)

    def _log_epoch_results(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        epoch_time: float,
        learning_rate: float,
    ) -> None:
        """Log results for one epoch."""
        # Prepare metrics for logging
        log_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "learning_rate": learning_rate,
            "epoch_time": epoch_time,
        }

        # Add validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                log_metrics[f"val_{key}"] = value

        # Add other training metrics
        for key, value in train_metrics.items():
            if key != "loss":
                log_metrics[f"train_{key}"] = value

        # Log to console and tracking systems
        self.logger.log_metrics(log_metrics, step=epoch)

        # Console output
        train_loss_str = f"Train Loss: {train_metrics['loss']:.4f}"
        val_loss_str = (
            f"Val Loss: {val_metrics.get('loss', 'N/A'):.4f}" if val_metrics else ""
        )
        lr_str = f"LR: {learning_rate:.2e}"
        time_str = f"Time: {format_time(epoch_time)}"

        log_line = f"Epoch {epoch:3d} | {train_loss_str}"
        if val_loss_str:
            log_line += f" | {val_loss_str}"
        log_line += f" | {lr_str} | {time_str}"

        self.logger.info(log_line)

    def save_model(self, path: str) -> None:
        """Save model to specified path."""
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model from specified path."""
        device = next(self.model.parameters()).device
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.logger.info(f"Model loaded from {path}")

    def save_checkpoint(
        self, epoch: int, metrics: dict[str, float], is_best: bool = False
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = (
            f"{self.config.logging.checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model if this is the best
        if is_best:
            best_path = f"{self.config.logging.checkpoint_dir}/../best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved: {best_path}")

    def load_checkpoint(self, path: str) -> dict:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.logger.info(
            f"Checkpoint loaded from {path}, resuming from epoch {self.current_epoch}"
        )

        return checkpoint

    def get_training_summary(self) -> dict[str, Any]:
        """Get summary of training session."""
        return {
            "total_epochs": self.current_epoch + 1,
            "best_metric": self.best_metric,
            "final_lr": self.optimizer.param_groups[0]["lr"],
            "training_time": 0,  # TODO: Implement training time tracking
            "early_stopped": self.early_stopping.early_stop,
            "config": self.config.to_dict(),
        }
