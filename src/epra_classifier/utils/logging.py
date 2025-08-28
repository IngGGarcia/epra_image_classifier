"""
Centralized logging system for EPRA Image Classifier.

This module provides a unified logging interface that supports multiple
backends including console, file, TensorBoard, and Weights & Biases.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """
    Centralized logger that supports multiple backends.

    This logger provides a unified interface for logging to console,
    files, TensorBoard, and Weights & Biases simultaneously.
    """

    def __init__(
        self,
        name: str = "epra_classifier",
        log_level: str = "INFO",
        log_dir: str | Path | None = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "epra-classifier",
        experiment_name: str | None = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            use_tensorboard: Whether to use TensorBoard logging
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            experiment_name: Name for this experiment
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.experiment_name = (
            experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize console logger
        self._setup_console_logger(log_level)

        # Initialize file logger
        self._setup_file_logger()

        # Initialize TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            self._setup_tensorboard()

        # Initialize Weights & Biases
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            self._setup_wandb(wandb_project)
        elif use_wandb and not WANDB_AVAILABLE:
            self.warning(
                "Weights & Biases not available. Install with: pip install wandb"
            )

    def _setup_console_logger(self, log_level: str) -> None:
        """Setup console logging."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(console_handler)

    def _setup_file_logger(self) -> None:
        """Setup file logging."""
        log_file = self.log_dir / f"{self.experiment_name}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        self.info(f"Logging to file: {log_file}")

    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        tb_dir = self.log_dir / "tensorboard" / self.experiment_name
        self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
        self.info(f"TensorBoard logging to: {tb_dir}")

    def _setup_wandb(self, project: str) -> None:
        """Setup Weights & Biases logging."""
        try:
            self.wandb_run = wandb.init(
                project=project, name=self.experiment_name, reinit=True
            )
            self.info(f"Weights & Biases initialized for project: {project}")
        except Exception as e:
            self.warning(f"Failed to initialize Weights & Biases: {e}")
            self.wandb_run = None

    # Standard logging methods
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

    # Metric logging methods
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """
        Log metrics to all configured backends.

        Args:
            metrics: Dictionary of metric names and values
            step: Step number (epoch, iteration, etc.)
        """
        # Log to console/file
        metrics_str = ", ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
        )
        step_str = f" (step {step})" if step is not None else ""
        self.info(f"Metrics{step_str}: {metrics_str}")

        # Log to TensorBoard
        if self.tb_writer and step is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(name, value, step)

        # Log to Weights & Biases
        if self.wandb_run:
            wandb_metrics = {
                k: v for k, v in metrics.items() if isinstance(v, (int, float, str))
            }
            if step is not None:
                wandb_metrics["step"] = step
            wandb.log(wandb_metrics)

    def log_hyperparameters(self, hparams: dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            hparams: Dictionary of hyperparameter names and values
        """
        self.info("Hyperparameters:")
        for name, value in hparams.items():
            self.info(f"  {name}: {value}")

        # Log to TensorBoard
        if self.tb_writer:
            # Convert values to scalar types for TensorBoard
            tb_hparams = {}
            for k, v in hparams.items():
                if isinstance(v, (int, float, str, bool)):
                    tb_hparams[k] = v
                else:
                    tb_hparams[k] = str(v)

            self.tb_writer.add_hparams(tb_hparams, {})

        # Log to Weights & Biases
        if self.wandb_run:
            wandb.config.update(hparams)

    def log_model_summary(self, model: torch.nn.Module, input_size: tuple) -> None:
        """
        Log model summary information.

        Args:
            model: PyTorch model
            input_size: Input tensor size (without batch dimension)
        """
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            self.info("Model Summary:")
            self.info(f"  Total parameters: {total_params:,}")
            self.info(f"  Trainable parameters: {trainable_params:,}")
            self.info(
                f"  Non-trainable parameters: {total_params - trainable_params:,}"
            )

            # Log model architecture
            self.debug(f"Model architecture:\n{model}")

            # Log to Weights & Biases
            if self.wandb_run:
                wandb.config.update(
                    {
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                        "model_architecture": str(model.__class__.__name__),
                    }
                )
        except Exception as e:
            self.warning(f"Failed to log model summary: {e}")

    def log_image(self, tag: str, image: torch.Tensor, step: int | None = None) -> None:
        """
        Log an image.

        Args:
            tag: Image tag/name
            image: Image tensor (C, H, W)
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_image(tag, image, step)

        if self.wandb_run:
            import numpy as np

            # Convert tensor to numpy and transpose for wandb
            img_np = image.cpu().numpy()
            if img_np.shape[0] == 3:  # RGB
                img_np = np.transpose(img_np, (1, 2, 0))
            wandb.log({tag: wandb.Image(img_np)}, step=step)

    def log_histogram(
        self, tag: str, values: torch.Tensor, step: int | None = None
    ) -> None:
        """
        Log histogram of values.

        Args:
            tag: Histogram tag/name
            values: Tensor values to create histogram from
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)

        if self.wandb_run:
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)

    def save_checkpoint_info(
        self, checkpoint_path: str, metrics: dict[str, Any]
    ) -> None:
        """
        Log checkpoint save information.

        Args:
            checkpoint_path: Path where checkpoint was saved
            metrics: Metrics at the time of saving
        """
        self.info(f"Checkpoint saved: {checkpoint_path}")
        self.log_metrics(metrics)

        if self.wandb_run:
            # Save checkpoint as artifact
            artifact = wandb.Artifact(
                name=f"model_checkpoint_{self.experiment_name}", type="model"
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()

        if self.wandb_run:
            wandb.finish()

        self.info("Logger closed")


# Global logger instance
_global_logger: Logger | None = None


def get_logger(name: str = "epra_classifier", **kwargs) -> Logger:
    """
    Get the global logger instance.

    Args:
        name: Logger name
        **kwargs: Additional arguments for Logger initialization

    Returns:
        Logger instance
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = Logger(name=name, **kwargs)

    return _global_logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | None = None,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "epra-classifier",
    experiment_name: str | None = None,
) -> Logger:
    """
    Setup global logging configuration.

    Args:
        log_level: Logging level
        log_dir: Directory for log files
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        experiment_name: Experiment name

    Returns:
        Configured logger instance
    """
    global _global_logger

    _global_logger = Logger(
        log_level=log_level,
        log_dir=log_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        experiment_name=experiment_name,
    )

    return _global_logger
