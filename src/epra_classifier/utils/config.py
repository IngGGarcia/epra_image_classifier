"""
Configuration management for EPRA Image Classifier.

This module provides centralized configuration management for all aspects
of the image classification system, including data processing, model parameters,
training settings, and evaluation metrics.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


@dataclass
class DataConfig:
    """Configuration for data processing and loading."""

    # Data paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"

    # Image processing
    image_size: tuple[int, int] = (224, 224)
    channels: int = 3

    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Data splitting
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 10
    color_jitter: bool = True

    # Dataset limits
    max_images_per_class: int | None = None

    # Normalization (ImageNet standards)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Model type
    model_type: str = "resnet18"  # resnet18, efficientnet_b0

    # Common parameters
    num_classes: int = 2  # Binary: 2, Multiclass: 5
    dropout_rate: float = 0.3
    pretrained: bool = True

    # Feature extraction
    feature_dim: int = 512
    use_attention: bool = True

    # Model-specific parameters
    hidden_size: int = 512
    use_spatial_attention: bool = False
    use_context_analysis: bool = False


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Optimization
    optimizer: str = "adam"  # adam, sgd, adamw
    momentum: float = 0.9  # For SGD
    scheduler: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, step
    scheduler_patience: int = 7
    scheduler_factor: float = 0.5

    # Loss function
    loss_function: str = "cross_entropy"  # cross_entropy, focal_loss, label_smoothing
    label_smoothing: float = 0.1
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0

    # Regularization
    early_stopping_patience: int = 15
    gradient_clip_value: float | None = None

    # Checkpointing
    save_best_only: bool = True
    save_every_n_epochs: int = 10

    # Validation
    val_check_interval: int = 1  # epochs


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Metrics to calculate
    metrics: list[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "confusion_matrix",
        ]
    )

    # Evaluation settings
    batch_size: int = 64
    threshold: float = 0.5  # For binary classification

    # Visualization
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_precision_recall: bool = True
    save_plots: bool = True

    # Class names for visualization
    class_names: list[str] = field(default_factory=lambda: ["Non-Violence", "Violence"])


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""

    # Logging level
    log_level: str = "INFO"

    # Experiment tracking
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "epra-classifier"
    experiment_name: str | None = None

    # Logging directories
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    # Logging frequency
    log_every_n_steps: int = 10
    save_model_every_n_epochs: int = 5


@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    # Model loading
    checkpoint_path: str | None = None
    model_type: str = "binary"  # binary, multiclass

    # Inference settings
    batch_size: int = 1
    device: str = "auto"  # auto, cpu, cuda

    # Output format
    return_probabilities: bool = True
    return_features: bool = False
    output_format: str = "dict"  # dict, json, csv

    # Postprocessing
    confidence_threshold: float = 0.5
    apply_smoothing: bool = False


@dataclass
class SystemConfig:
    """System-wide configuration."""

    # Device management
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True

    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True

    # Performance
    compile_model: bool = False  # torch.compile (requires PyTorch 2.0+)

    # Paths
    project_root: str = ""
    data_root: str = "data"
    output_root: str = "outputs"


@dataclass
class Config:
    """Main configuration class that combines all sub-configurations."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        """Post-initialization processing."""
        # Set automatic device detection
        if self.system.device == "auto":
            self.system.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure paths are absolute
        self._resolve_paths()

        # Set class names based on model type
        if self.model.num_classes == 2:
            self.evaluation.class_names = ["Non-Violence", "Violence"]
        elif self.model.num_classes == 5:
            self.evaluation.class_names = [
                "Level_0",
                "Level_1",
                "Level_2",
                "Level_3",
                "Level_4",
            ]

    def _resolve_paths(self):
        """Convert relative paths to absolute paths."""
        if self.system.project_root:
            root = Path(self.system.project_root)
        else:
            root = Path.cwd()

        # Resolve data paths
        self.data.raw_data_path = str(root / self.data.raw_data_path)
        self.data.processed_data_path = str(root / self.data.processed_data_path)

        # Resolve logging paths
        self.logging.log_dir = str(root / self.logging.log_dir)
        self.logging.checkpoint_dir = str(root / self.logging.checkpoint_dir)

        # Resolve system paths
        self.system.data_root = str(root / self.system.data_root)
        self.system.output_root = str(root / self.system.output_root)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Config instance loaded from YAML
        """
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """
        Create Config instance from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            Config instance
        """
        # Initialize empty config
        config = cls()

        # Update each section
        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_config = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

        return config

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "logging": self.logging.__dict__,
            "inference": self.inference.__dict__,
            "system": self.system.__dict__,
        }

    def save_yaml(self, config_path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path where to save the YAML configuration
        """
        config_dict = self.to_dict()

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def update_from_args(self, args: dict) -> None:
        """
        Update configuration from command-line arguments.

        Args:
            args: Dictionary of command-line arguments
        """
        for key, value in args.items():
            if "." in key:
                # Handle nested keys like "training.learning_rate"
                section, param = key.split(".", 1)
                if hasattr(self, section):
                    section_config = getattr(self, section)
                    if hasattr(section_config, param):
                        setattr(section_config, param, value)
            else:
                # Handle top-level keys
                if hasattr(self, key):
                    setattr(self, key, value)


def get_default_config(model_type: str = "binary") -> Config:
    """
    Get default configuration for a specific model type.

    Args:
        model_type: Type of model ("binary" or "multiclass")

    Returns:
        Default configuration for the specified model type
    """
    config = Config()

    if model_type == "binary":
        config.model.num_classes = 2
        config.model.model_type = "resnet18"
        config.evaluation.class_names = ["Non-Violence", "Violence"]
        config.logging.wandb_project = "epra-binary-classifier"
    elif model_type == "multiclass":
        config.model.num_classes = 5
        config.model.model_type = "efficientnet_b0"
        config.evaluation.class_names = [
            "Level_0",
            "Level_1",
            "Level_2",
            "Level_3",
            "Level_4",
        ]
        config.logging.wandb_project = "epra-multiclass-classifier"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return config


def create_config_files():
    """Create default configuration files for binary and multiclass models."""
    # Create configs directory
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    # Create binary config
    binary_config = get_default_config("binary")
    binary_config.save_yaml(config_dir / "binary_config.yaml")

    # Create multiclass config
    multiclass_config = get_default_config("multiclass")
    multiclass_config.save_yaml(config_dir / "multiclass_config.yaml")

    print(f"Default configuration files created in {config_dir}")


if __name__ == "__main__":
    # Create default configuration files
    create_config_files()
