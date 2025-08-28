#!/usr/bin/env python3
"""
Training script for EPRA Image Classifier.

This script provides a unified interface for training both binary and
multiclass violence classification models with comprehensive configuration
support and experiment tracking.
"""

import argparse
import sys
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epra_classifier.data.loaders import get_dataloaders
from epra_classifier.models import create_model
from epra_classifier.utils.config import Config
from epra_classifier.utils.helpers import set_seed
from epra_classifier.utils.logging import get_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EPRA Violence Classification Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/binary_config.yaml",
        help="Path to configuration file",
    )

    # Data
    parser.add_argument(
        "--data-dir", "-d", type=str, required=True, help="Path to dataset directory"
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["violence", "custom"],
        default="violence",
        help="Type of dataset to load",
    )

    # Model
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["binary", "multiclass", "auto"],
        default="auto",
        help="Type of model to train",
    )

    parser.add_argument(
        "--num-classes", type=int, help="Number of classes (overrides config)"
    )

    # Training
    parser.add_argument(
        "--epochs", "-e", type=int, help="Number of training epochs (overrides config)"
    )

    parser.add_argument(
        "--batch-size", "-b", type=int, help="Batch size (overrides config)"
    )

    parser.add_argument(
        "--learning-rate", "-lr", type=float, help="Learning rate (overrides config)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )

    parser.add_argument(
        "--experiment-name", type=str, help="Experiment name for tracking"
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for training",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of data loading workers (overrides config)",
    )

    # Flags
    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run setup without training (for testing)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def setup_config(args: argparse.Namespace) -> Config:
    """Setup configuration from file and command line arguments."""
    # Load base config
    config = Config.from_yaml(args.config)

    # Override with command line arguments
    if args.num_classes:
        config.model.num_classes = args.num_classes

    if args.epochs:
        config.training.num_epochs = args.epochs

    if args.batch_size:
        config.data.batch_size = args.batch_size

    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    if args.num_workers:
        config.data.num_workers = args.num_workers

    # System settings
    config.system.device = args.device
    config.system.random_seed = args.seed

    # Output settings
    config.logging.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
    config.logging.log_dir = str(Path(args.output_dir) / "logs")

    if args.experiment_name:
        config.logging.experiment_name = args.experiment_name

    if args.verbose:
        config.logging.log_level = "DEBUG"

    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Setup configuration
    config = setup_config(args)

    # Set random seed
    set_seed(config.system.random_seed)

    # Setup logger
    logger = get_logger()
    logger.info("Starting EPRA Violence Classification Training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Create output directories
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)

        # Create dataloaders
        logger.info("Creating dataloaders...")
        dataloaders = get_dataloaders(
            data_dir=args.data_dir, config=config, dataset_type=args.dataset_type
        )

        train_loader = dataloaders["train"]
        val_loader = dataloaders.get("val")

        logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"Validation samples: {len(val_loader.dataset)}")

        # Auto-detect model type if needed
        if args.model_type == "auto":
            # Get number of classes from dataset
            if hasattr(train_loader.dataset, "get_class_names"):
                num_classes = len(train_loader.dataset.get_class_names())
                model_type = "binary" if num_classes == 2 else "multiclass"
            else:
                model_type = "binary"  # Default
            logger.info(f"Auto-detected model type: {model_type}")
        else:
            model_type = args.model_type

        # Update config with detected model type
        if model_type == "binary":
            config.model.num_classes = 2
        elif model_type == "multiclass" and not args.num_classes:
            # Use dataset classes if not specified
            if hasattr(train_loader.dataset, "get_class_names"):
                config.model.num_classes = len(train_loader.dataset.get_class_names())

        # Create model
        logger.info(f"Creating {model_type} model...")
        model = create_model(config, model_type)

        # Create trainer
        from epra_classifier.training.trainers import ViolenceTrainer

        trainer = ViolenceTrainer(model, config)

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Log model summary
        summary = model.summary()
        logger.info(f"Model: {summary['model_type']}")
        logger.info(
            f"Parameters: {summary['total_parameters']:,} ({summary['trainable_parameters']:,} trainable)"
        )

        if args.dry_run:
            logger.info("Dry run completed successfully")
            return

        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.num_epochs,
        )

        # Save final results
        results_path = Path(args.output_dir) / "training_results.json"

        final_results = {
            "config": config.to_dict(),
            "model_summary": summary,
            "training_history": history,
            "performance": trainer.get_training_summary(),
        }

        import json

        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Best metric: {trainer.best_metric:.4f}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
