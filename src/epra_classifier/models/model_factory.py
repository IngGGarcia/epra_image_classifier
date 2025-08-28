"""
Model factory for creating different types of violence classification models.

This module provides factory functions for creating models with consistent
configurations and proper initialization.
"""

from typing import Any

from ..utils.config import Config, ModelConfig
from ..utils.logging import get_logger
from .base.base_model import BaseModel
from .binary.binary_model import BinaryViolenceModel, create_binary_model
from .multiclass.multiclass_model import (
    MulticlassViolenceModel,
    create_multiclass_model,
)


def create_model(
    config: Config | ModelConfig, model_type: str | None = None
) -> BaseModel:
    """
    Create a model based on configuration.

    Args:
        config: Configuration object
        model_type: Override model type ("binary", "multiclass", or None for auto-detect)

    Returns:
        Initialized model
    """
    logger = get_logger()

    # Extract model config if needed
    if isinstance(config, Config):
        model_config = config.model
    else:
        model_config = config

    # Determine model type
    if model_type is None:
        if model_config.num_classes == 2:
            model_type = "binary"
        elif model_config.num_classes > 2:
            model_type = "multiclass"
        else:
            raise ValueError(f"Invalid number of classes: {model_config.num_classes}")

    model_type = model_type.lower()

    # Create model based on type
    if model_type == "binary":
        model = create_binary_model(model_config)
        logger.info(
            f"Created binary violence model with {model_config.model_type} backbone"
        )

    elif model_type == "multiclass":
        model = create_multiclass_model(model_config)
        logger.info(
            f"Created multiclass violence model with {model_config.model_type} backbone"
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Log model summary
    summary = model.summary()
    logger.info(
        f"Model summary: {summary['total_parameters']:,} parameters "
        f"({summary['trainable_parameters']:,} trainable)"
    )

    return model


def get_available_models() -> dict[str, dict[str, Any]]:
    """
    Get information about available model architectures.

    Returns:
        Dictionary with model information
    """
    return {
        "binary": {
            "class": BinaryViolenceModel,
            "description": "Binary violence classification using ResNet18",
            "backbone_options": ["resnet18"],
            "num_classes": 2,
            "features": ["attention", "transfer_learning"],
            "use_cases": ["violence_detection", "binary_classification"],
        },
        "multiclass": {
            "class": MulticlassViolenceModel,
            "description": "Multiclass violence level classification using EfficientNet",
            "backbone_options": [
                "efficientnet_b0",
                "efficientnet_b1",
                "efficientnet_b2",
            ],
            "num_classes": "variable (3-10)",
            "features": ["spatial_attention", "context_analysis", "transfer_learning"],
            "use_cases": ["violence_level_detection", "multiclass_classification"],
        },
    }


def create_model_from_checkpoint(
    checkpoint_path: str, config: Config | ModelConfig | None = None
) -> BaseModel:
    """
    Create a model and load weights from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration (will be loaded from checkpoint if None)

    Returns:
        Model with loaded weights
    """
    from pathlib import Path

    import torch

    logger = get_logger()

    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get config from checkpoint if not provided
    if config is None:
        if "config" in checkpoint:
            from ..utils.config import Config

            config_dict = checkpoint["config"]
            config = Config.from_dict(config_dict)
        else:
            raise ValueError("Config not found in checkpoint and not provided")

    # Create model
    model = create_model(config)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume checkpoint is just state dict
        model.load_state_dict(checkpoint)

    logger.info(f"Loaded model from checkpoint: {checkpoint_path}")

    return model


def get_model_requirements(model_type: str) -> dict[str, Any]:
    """
    Get hardware and software requirements for a model type.

    Args:
        model_type: Type of model ("binary" or "multiclass")

    Returns:
        Dictionary with requirements
    """
    requirements = {
        "binary": {
            "min_gpu_memory": "2GB",
            "recommended_gpu_memory": "4GB",
            "min_cpu_cores": 2,
            "recommended_cpu_cores": 4,
            "pytorch_version": ">=1.12.0",
            "python_version": ">=3.8",
            "approximate_size": "45MB",
            "inference_speed": "~50ms per image (GPU), ~200ms (CPU)",
        },
        "multiclass": {
            "min_gpu_memory": "4GB",
            "recommended_gpu_memory": "8GB",
            "min_cpu_cores": 4,
            "recommended_cpu_cores": 8,
            "pytorch_version": ">=1.12.0",
            "python_version": ">=3.8",
            "approximate_size": "20MB",
            "inference_speed": "~30ms per image (GPU), ~150ms (CPU)",
        },
    }

    return requirements.get(model_type.lower(), {})


def validate_model_config(config: ModelConfig, model_type: str) -> bool:
    """
    Validate model configuration for a specific model type.

    Args:
        config: Model configuration
        model_type: Type of model to validate for

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    model_type = model_type.lower()

    if model_type == "binary":
        if config.num_classes != 2:
            raise ValueError(
                f"Binary model requires num_classes=2, got {config.num_classes}"
            )

        valid_backbones = ["resnet18", "resnet34", "resnet50"]
        if config.model_type not in valid_backbones:
            raise ValueError(
                f"Binary model supports {valid_backbones}, got {config.model_type}"
            )

    elif model_type == "multiclass":
        if config.num_classes < 3:
            raise ValueError(
                f"Multiclass model requires num_classes>=3, got {config.num_classes}"
            )

        valid_backbones = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
        if config.model_type not in valid_backbones:
            raise ValueError(
                f"Multiclass model supports {valid_backbones}, got {config.model_type}"
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Common validations
    if config.dropout_rate < 0 or config.dropout_rate >= 1:
        raise ValueError(f"Dropout rate must be in [0, 1), got {config.dropout_rate}")

    if config.feature_dim <= 0:
        raise ValueError(
            f"Feature dimension must be positive, got {config.feature_dim}"
        )

    return True


def print_model_summary(model: BaseModel) -> None:
    """
    Print a detailed model summary.

    Args:
        model: Model to summarize
    """
    summary = model.summary()

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Model Type: {summary['model_type']}")
    print(f"Backbone: {summary['backbone_type']}")
    print(f"Number of Classes: {summary['num_classes']}")
    print(f"Feature Dimension: {summary['feature_dim']}")
    print(f"Use Attention: {summary['use_attention']}")
    print(f"Dropout Rate: {summary['dropout_rate']}")
    print("-" * 60)
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Non-trainable Parameters: {summary['non_trainable_parameters']:,}")
    print("=" * 60)


# Convenience functions for backward compatibility
def create_violence_binary_model(config: ModelConfig) -> BinaryViolenceModel:
    """Create binary violence model (legacy function)."""
    return create_binary_model(config)


def create_violence_multiclass_model(config: ModelConfig) -> MulticlassViolenceModel:
    """Create multiclass violence model (legacy function)."""
    return create_multiclass_model(config)
