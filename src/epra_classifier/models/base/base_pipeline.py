"""
Base pipeline class for image classification tasks.

This module provides a unified base pipeline that implements common functionality
for both binary and multiclass classification tasks, including model management,
data loading, checkpoint handling, and device management.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ...utils.config import Config
from ...utils.helpers import (
    count_parameters,
    create_directory,
    format_time,
    get_device,
    set_seed,
)
from ...utils.logging import get_logger


class PipelineError(Exception):
    """Base exception class for pipeline-related errors."""

    pass


class CheckpointError(PipelineError):
    """Exception raised for checkpoint-related errors."""

    pass


class DatasetError(PipelineError):
    """Exception raised for dataset-related errors."""

    pass


class BasePipeline(ABC):
    """
    Base pipeline class for image classification tasks.

    This abstract base class provides common functionality for all classification
    pipelines, including model management, data loading, checkpoint handling,
    and training utilities. It implements best practices from PyTorch and ML research.

    The pipeline follows a standard structure for:
    - Model initialization and management
    - Checkpoint loading and saving
    - DataLoader configuration with optimized settings
    - Device management and mixed precision
    - Logging and metrics tracking
    - Reproducibility control
    """

    def __init__(self, config: Config):
        """
        Initialize the base pipeline.

        Args:
            config: Configuration object containing all pipeline settings
        """
        self.config = config
        self.logger = get_logger()

        # Set up reproducibility
        set_seed(config.system.random_seed)

        # Set up device
        self.device = get_device(config.system.device)
        self.logger.info(f"Using device: {self.device}")

        # Create necessary directories
        self.checkpoint_dir = create_directory(config.logging.checkpoint_dir)
        self.log_dir = create_directory(config.logging.log_dir)

        # Initialize model (to be implemented by subclasses)
        self.model: nn.Module | None = None
        self.criterion: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

        # Training state
        self.current_epoch = 0
        self.best_metric = float("-inf")
        self.training_start_time = None

        # Mixed precision training
        self.scaler = None
        if config.system.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision training enabled")

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Build the model architecture.

        Returns:
            Initialized model
        """
        pass

    @abstractmethod
    def _build_criterion(self) -> nn.Module:
        """
        Build the loss function.

        Returns:
            Loss function
        """
        pass

    def initialize(self) -> None:
        """Initialize the pipeline components."""
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)

        # Build criterion
        self.criterion = self._build_criterion()

        # Log model information
        total_params, trainable_params = count_parameters(self.model)
        self.logger.info(f"Model initialized with {total_params:,} total parameters")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        # Log model summary if available
        try:
            self.logger.log_model_summary(
                self.model, tuple(self.config.data.image_size)
            )
        except Exception as e:
            self.logger.warning(f"Could not log model summary: {e}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build optimizer based on configuration.

        Returns:
            Configured optimizer
        """
        if self.model is None:
            raise RuntimeError("Model must be initialized before building optimizer")

        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.training.momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """
        Build learning rate scheduler based on configuration.

        Returns:
            Configured scheduler or None
        """
        if self.optimizer is None:
            return None

        scheduler_name = self.config.training.scheduler.lower()

        if scheduler_name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
                verbose=True,
            )
        elif scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs
            )
        elif scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler_patience,
                gamma=self.config.training.scheduler_factor,
            )
        elif scheduler_name == "none":
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        batch_size: int | None = None,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader with optimized settings.

        Args:
            dataset: Dataset to load
            shuffle: Whether to shuffle the data
            batch_size: Batch size (uses config default if None)
            num_workers: Number of worker processes (uses config default if None)
            pin_memory: Whether to pin memory (uses config default if None)
            drop_last: Whether to drop the last incomplete batch

        Returns:
            Configured DataLoader

        Raises:
            DatasetError: If dataset is invalid
        """
        if dataset is None:
            raise DatasetError("Dataset cannot be None")

        if not hasattr(dataset, "__len__"):
            raise DatasetError("Dataset must implement __len__ method")

        batch_size = batch_size or self.config.data.batch_size
        num_workers = num_workers or self.config.data.num_workers
        pin_memory = (
            pin_memory if pin_memory is not None else self.config.data.pin_memory
        )

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and self.device.type == "cuda",
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else 2,
        )

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
        extra_data: dict[str, Any] | None = None,
    ) -> str:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Current metrics
            is_best: Whether this is the best model so far
            extra_data: Additional data to save in checkpoint

        Returns:
            Path to saved checkpoint
        """
        if self.model is None or self.optimizer is None:
            raise CheckpointError("Model and optimizer must be initialized")

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
            "random_seed": self.config.system.random_seed,
            "timestamp": datetime.now().isoformat(),
        }

        if self.scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint_data["scaler_state_dict"] = self.scaler.state_dict()

        if extra_data:
            checkpoint_data.update(extra_data)

        # Determine checkpoint filename
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = (
                self.checkpoint_dir / f"checkpoint_epoch_{epoch}_{timestamp}.pth"
            )

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            self.logger.info(f"New best model saved with metrics: {metrics}")

        # Log checkpoint info
        self.logger.save_checkpoint_info(str(checkpoint_path), metrics)

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str | None = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (loads best if None)
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            strict: Whether to strictly enforce state dict matching

        Returns:
            Checkpoint data dictionary

        Raises:
            CheckpointError: If checkpoint loading fails
        """
        if self.model is None:
            raise CheckpointError("Model must be initialized before loading checkpoint")

        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise CheckpointError(f"Checkpoint not found: {checkpoint_path}")

        try:
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(
                    checkpoint["model_state_dict"], strict=strict
                )

                # Load optimizer state
                if (
                    load_optimizer
                    and self.optimizer is not None
                    and "optimizer_state_dict" in checkpoint
                ):
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # Load scheduler state
                if (
                    load_scheduler
                    and self.scheduler is not None
                    and "scheduler_state_dict" in checkpoint
                ):
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                # Load scaler state
                if self.scaler is not None and "scaler_state_dict" in checkpoint:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

                # Update training state
                if "epoch" in checkpoint:
                    self.current_epoch = checkpoint["epoch"]

                self.logger.info("Checkpoint loaded successfully")
                return checkpoint
            else:
                # Legacy checkpoint format (just model state dict)
                self.model.load_state_dict(checkpoint, strict=strict)
                self.logger.info("Legacy checkpoint loaded successfully")
                return {"model_state_dict": checkpoint}

        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e

    def to_device(
        self, data: torch.Tensor | dict | list | tuple
    ) -> torch.Tensor | dict | list | tuple:
        """
        Move data to the specified device.

        Args:
            data: Data to move to device

        Returns:
            Data moved to device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.to_device(item) for item in data)
        else:
            return data

    def set_mode(self, mode: str) -> None:
        """
        Set model mode (train/eval).

        Args:
            mode: Mode to set ('train' or 'eval')
        """
        if self.model is None:
            return

        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'eval'")

    def get_lr(self) -> float:
        """
        Get current learning rate.

        Returns:
            Current learning rate
        """
        if self.optimizer is None:
            return 0.0

        return self.optimizer.param_groups[0]["lr"]

    def get_training_time(self) -> str:
        """
        Get formatted training time.

        Returns:
            Formatted training time string
        """
        if self.training_start_time is None:
            return "0s"

        elapsed = time.time() - self.training_start_time
        return format_time(elapsed)

    def start_training(self) -> None:
        """Mark the start of training."""
        self.training_start_time = time.time()
        self.logger.info("Training started")

    def finish_training(self) -> None:
        """Mark the end of training."""
        if self.training_start_time is not None:
            total_time = self.get_training_time()
            self.logger.info(f"Training completed in {total_time}")
        else:
            self.logger.info("Training completed")

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self.logger, "close"):
            self.logger.close()

        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
