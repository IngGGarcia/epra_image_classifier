#!/usr/bin/env python3
"""
Evaluation script for EPRA Image Classifier.

This script provides comprehensive model evaluation including metrics calculation,
visualization generation, and detailed performance analysis.
"""

import argparse
import json
import sys
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epra_classifier.data.loaders import get_dataloaders
from epra_classifier.evaluation.metrics import (
    calculate_all_metrics,
    format_metrics_report,
)
from epra_classifier.evaluation.visualization import create_evaluation_report
from epra_classifier.inference.pipelines import ViolenceInferencePipeline
from epra_classifier.models import create_model_from_checkpoint
from epra_classifier.utils.config import Config
from epra_classifier.utils.logging import get_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate EPRA Violence Classification Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")

    # Model
    parser.add_argument(
        "--checkpoint", "-m", type=str, required=True, help="Path to model checkpoint"
    )

    # Data
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Path to test dataset directory",
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["violence", "custom"],
        default="violence",
        help="Type of dataset to evaluate",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val", "train"],
        default="test",
        help="Dataset split to evaluate on",
    )

    # Evaluation options
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for evaluation"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for evaluation",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="evaluation_results",
        help="Output directory for results and reports",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions to file",
    )

    parser.add_argument(
        "--create-visualizations",
        action="store_true",
        help="Create evaluation plots and visualizations",
    )

    # Analysis options
    parser.add_argument(
        "--class-names", nargs="+", help="Custom class names for reporting"
    )

    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Confidence thresholds to analyze",
    )

    # System
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    return parser.parse_args()


def setup_config(args: argparse.Namespace) -> Config:
    """Setup configuration for evaluation."""
    # Load from config file if provided
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override inference settings with command line arguments
    config.inference.checkpoint_path = args.checkpoint
    config.inference.device = args.device
    config.inference.batch_size = args.batch_size
    config.inference.return_probabilities = True
    config.inference.return_features = True

    # Data configuration
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers

    # Evaluation configuration
    if args.class_names:
        config.evaluation.class_names = args.class_names

    # Logging
    if args.verbose:
        config.logging.log_level = "DEBUG"

    return config


def evaluate_at_threshold(
    y_true: list,
    y_pred_probs: list,
    threshold: float,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict:
    """Evaluate model at specific confidence threshold."""
    import numpy as np

    # Apply threshold
    y_pred = []
    filtered_indices = []

    for i, probs in enumerate(y_pred_probs):
        max_prob = max(probs)
        if max_prob >= threshold:
            y_pred.append(np.argmax(probs))
            filtered_indices.append(i)
        # else: skip low confidence predictions

    if not y_pred:
        return {
            "threshold": threshold,
            "predictions_above_threshold": 0,
            "coverage": 0.0,
        }

    # Filter true labels to match predictions
    y_true_filtered = [y_true[i] for i in filtered_indices]

    # Calculate metrics
    metrics = calculate_all_metrics(
        y_true_filtered,
        y_pred,
        y_prob=None,  # Not needed for threshold analysis
        num_classes=num_classes,
        class_names=class_names,
    )

    # Add threshold-specific info
    metrics.update(
        {
            "threshold": threshold,
            "predictions_above_threshold": len(y_pred),
            "coverage": len(y_pred) / len(y_true),
        }
    )

    return metrics


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()

    # Setup configuration
    config = setup_config(args)

    # Setup logger
    logger = get_logger()
    logger.info("Starting EPRA Violence Classification Evaluation")
    logger.info(f"Model checkpoint: {args.checkpoint}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Evaluating on: {args.split} split")

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info("Loading model...")
        model = create_model_from_checkpoint(args.checkpoint, config)

        # Get model info
        model_summary = model.summary()
        logger.info(f"Model: {model_summary['model_type']}")
        logger.info(f"Classes: {model_summary['num_classes']}")

        # Create dataloaders
        logger.info("Creating dataloaders...")
        dataloaders = get_dataloaders(
            data_dir=args.data_dir, config=config, dataset_type=args.dataset_type
        )

        # Select appropriate dataloader
        if args.split == "test":
            eval_loader = dataloaders.get("test")
            if eval_loader is None:
                logger.warning("Test set not available, using validation set")
                eval_loader = dataloaders.get("val")
        elif args.split == "val":
            eval_loader = dataloaders.get("val")
        else:  # train
            eval_loader = dataloaders.get("train")

        if eval_loader is None:
            raise ValueError(f"No {args.split} dataloader available")

        logger.info(f"Evaluation samples: {len(eval_loader.dataset)}")

        # Get class names
        class_names = args.class_names
        if class_names is None and hasattr(eval_loader.dataset, "get_class_names"):
            class_names = eval_loader.dataset.get_class_names()

        # Create inference pipeline
        pipeline = ViolenceInferencePipeline(config)

        # Run evaluation
        logger.info("Running evaluation...")
        y_true = []
        y_pred = []
        y_prob = []
        all_predictions = []

        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(
                tqdm(eval_loader, desc="Evaluating")
            ):
                images = images.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(images)
                logits = outputs["logits"]

                # Get predictions and probabilities
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # Store results
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(predictions.cpu().numpy().tolist())
                y_prob.extend(probabilities.cpu().numpy().tolist())

                # Store detailed predictions if requested
                if args.save_predictions:
                    for i in range(len(targets)):
                        pred_dict = {
                            "true_class": int(targets[i].item()),
                            "predicted_class": int(predictions[i].item()),
                            "probabilities": probabilities[i].cpu().numpy().tolist(),
                            "confidence": float(torch.max(probabilities[i]).item()),
                        }
                        if class_names:
                            pred_dict["true_class_name"] = (
                                class_names[pred_dict["true_class"]]
                                if pred_dict["true_class"] < len(class_names)
                                else f"class_{pred_dict['true_class']}"
                            )
                            pred_dict["predicted_class_name"] = (
                                class_names[pred_dict["predicted_class"]]
                                if pred_dict["predicted_class"] < len(class_names)
                                else f"class_{pred_dict['predicted_class']}"
                            )

                        all_predictions.append(pred_dict)

        logger.info("Calculating metrics...")

        # Calculate main metrics
        import numpy as np

        main_metrics = calculate_all_metrics(
            np.array(y_true),
            np.array(y_pred),
            np.array(y_prob),
            num_classes=model_summary["num_classes"],
            class_names=class_names,
        )

        # Calculate metrics at different confidence thresholds
        threshold_metrics = {}
        for threshold in args.confidence_thresholds:
            threshold_metrics[threshold] = evaluate_at_threshold(
                y_true, y_prob, threshold, model_summary["num_classes"], class_names
            )

        # Create evaluation report
        evaluation_results = {
            "model_info": {
                "checkpoint_path": args.checkpoint,
                "model_summary": model_summary,
                "evaluation_dataset": {
                    "path": args.data_dir,
                    "split": args.split,
                    "samples": len(eval_loader.dataset),
                    "classes": class_names or model_summary["num_classes"],
                },
            },
            "metrics": main_metrics,
            "threshold_analysis": threshold_metrics,
            "detailed_predictions": all_predictions if args.save_predictions else None,
        }

        # Save results
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        logger.info(f"Evaluation results saved to: {results_path}")

        # Create visualizations if requested
        if args.create_visualizations:
            logger.info("Creating visualizations...")

            viz_dir = output_dir / "visualizations"
            plot_paths = create_evaluation_report(main_metrics, save_dir=viz_dir)

            logger.info(f"Visualizations saved to: {viz_dir}")

        # Print metrics report
        report = format_metrics_report(main_metrics, "Model Evaluation Results")
        print(report)

        # Print threshold analysis summary
        print("\nThreshold Analysis:")
        print("-" * 40)
        print(f"{'Threshold':<10} {'Coverage':<10} {'Accuracy':<10} {'F1':<10}")
        print("-" * 40)

        for threshold in args.confidence_thresholds:
            metrics = threshold_metrics[threshold]
            coverage = metrics.get("coverage", 0)
            accuracy = metrics.get("accuracy", 0)
            f1 = metrics.get("f1", 0)
            print(f"{threshold:<10.2f} {coverage:<10.2f} {accuracy:<10.3f} {f1:<10.3f}")

        # Violence-specific summary (if binary)
        if model_summary["num_classes"] == 2:
            violence_metrics = [
                ("Violence Detection Rate", main_metrics.get("violence_recall", 0)),
                ("False Positive Rate", main_metrics.get("false_positive_rate", 0)),
                ("Violence Precision", main_metrics.get("violence_precision", 0)),
                ("Overall Accuracy", main_metrics.get("accuracy", 0)),
            ]

            print("\nViolence Detection Summary:")
            print("-" * 40)
            for metric_name, value in violence_metrics:
                print(f"{metric_name:<25}: {value:.3f}")

        logger.info("Evaluation completed successfully!")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
