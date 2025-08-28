#!/usr/bin/env python3
"""
Inference script for EPRA Image Classifier.

This script provides a unified interface for running inference on single images
or batches of images with comprehensive result analysis and reporting.
"""

import argparse
import json
import sys
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epra_classifier.inference.pipelines import ViolenceInferencePipeline
from epra_classifier.inference.postprocessing import ConfidenceFilter, ResultAggregator
from epra_classifier.utils.config import Config
from epra_classifier.utils.logging import get_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with EPRA Violence Classification Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")

    # Model
    parser.add_argument(
        "--checkpoint", "-m", type=str, required=True, help="Path to model checkpoint"
    )

    # Input
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to image file or directory containing images",
    )

    parser.add_argument(
        "--input-list",
        type=str,
        help="Path to text file containing list of image paths",
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Valid image extensions to process",
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=str, help="Path to save results (JSON format)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save detailed reports and visualizations",
    )

    # Inference settings
    parser.add_argument(
        "--batch-size", "-b", type=int, default=1, help="Batch size for inference"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for inference",
    )

    # Output options
    parser.add_argument(
        "--return-probabilities",
        action="store_true",
        help="Include class probabilities in results",
    )

    parser.add_argument(
        "--return-features",
        action="store_true",
        help="Include extracted features in results",
    )

    parser.add_argument(
        "--filter-low-confidence",
        action="store_true",
        help="Filter out low confidence predictions",
    )

    # Analysis options
    parser.add_argument(
        "--create-report",
        action="store_true",
        help="Create comprehensive analysis report",
    )

    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Include aggregated analysis of results",
    )

    # System
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations for consistent timing",
    )

    return parser.parse_args()


def get_image_paths(args: argparse.Namespace) -> list[Path]:
    """Get list of image paths from input arguments."""
    image_paths = []

    if args.input_list:
        # Load from file list
        with open(args.input_list) as f:
            paths = [line.strip() for line in f if line.strip()]
        image_paths.extend([Path(p) for p in paths])

    elif Path(args.input).is_file():
        # Single file
        image_paths.append(Path(args.input))

    elif Path(args.input).is_dir():
        # Directory - find all images
        input_dir = Path(args.input)
        for ext in args.extensions:
            image_paths.extend(input_dir.glob(f"**/*{ext}"))
            image_paths.extend(input_dir.glob(f"**/*{ext.upper()}"))

    else:
        raise ValueError(f"Input path not found: {args.input}")

    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))

    # Validate paths exist
    valid_paths = [p for p in image_paths if p.exists()]

    if len(valid_paths) != len(image_paths):
        missing = len(image_paths) - len(valid_paths)
        print(f"Warning: {missing} image paths not found and will be skipped")

    return valid_paths


def setup_inference_config(args: argparse.Namespace) -> Config:
    """Setup inference configuration."""
    # Load from config file if provided
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override inference settings with command line arguments
    config.inference.checkpoint_path = args.checkpoint
    config.inference.device = args.device
    config.inference.batch_size = args.batch_size
    config.inference.confidence_threshold = args.confidence_threshold
    config.inference.return_probabilities = args.return_probabilities
    config.inference.return_features = args.return_features

    # Logging
    if args.verbose:
        config.logging.log_level = "DEBUG"

    return config


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_arguments()

    # Setup configuration
    config = setup_inference_config(args)

    # Setup logger
    logger = get_logger()
    logger.info("Starting EPRA Violence Classification Inference")
    logger.info(f"Model checkpoint: {args.checkpoint}")
    logger.info(f"Input: {args.input}")

    try:
        # Get image paths
        logger.info("Collecting image paths...")
        image_paths = get_image_paths(args)

        if not image_paths:
            logger.error("No valid image paths found")
            sys.exit(1)

        logger.info(f"Found {len(image_paths)} images to process")

        # Create inference pipeline
        logger.info("Loading model and creating inference pipeline...")
        pipeline = ViolenceInferencePipeline(config)

        # Warmup model
        if args.warmup > 0:
            logger.info(f"Warming up model with {args.warmup} iterations...")
            pipeline.warmup(args.warmup)

        # Run inference
        logger.info("Running inference...")

        if len(image_paths) == 1:
            # Single image inference
            results = [pipeline.predict_single(image_paths[0])]
        else:
            # Batch inference
            batch_results = pipeline.predict_violence_batch(
                image_paths, filter_high_confidence=args.filter_low_confidence
            )
            results = batch_results["individual_predictions"]

        logger.info(f"Processed {len(results)} images")

        # Apply post-processing filters if requested
        if args.filter_low_confidence:
            logger.info("Applying confidence filter...")
            confidence_filter = ConfidenceFilter(
                min_confidence=args.confidence_threshold,
                violence_threshold=args.confidence_threshold + 0.1,
            )
            filtered_results = confidence_filter.filter(results)
            filter_stats = confidence_filter.get_statistics(results)
            logger.info(
                f"Filter passed: {filter_stats['passed_filter']}/{filter_stats['total_predictions']} predictions"
            )
        else:
            filtered_results = results

        # Create comprehensive results
        final_results = {
            "metadata": {
                "model_checkpoint": args.checkpoint,
                "total_images": len(image_paths),
                "processed_images": len(results),
                "filtered_images": len(filtered_results)
                if args.filter_low_confidence
                else len(results),
                "inference_config": config.inference.__dict__,
                "timestamp": pipeline.inference_times[-1]
                if pipeline.inference_times
                else None,
            },
            "performance": pipeline.get_performance_stats(),
            "predictions": filtered_results if args.filter_low_confidence else results,
        }

        # Add aggregated analysis if requested
        if args.aggregate_results:
            logger.info("Creating aggregated analysis...")
            aggregator = ResultAggregator()

            # Aggregate by timeframes (if timestamps available)
            timeframe_analysis = aggregator.aggregate_by_timeframe(
                results, timeframe_seconds=300
            )  # 5-minute windows

            if len(timeframe_analysis) > 1:
                trend_analysis = aggregator.analyze_trends(timeframe_analysis)
                final_results["aggregated_analysis"] = {
                    "timeframe_analysis": timeframe_analysis,
                    "trend_analysis": trend_analysis,
                }
            else:
                # Simple batch analysis
                batch_analysis = aggregator._aggregate_frame(results, 0)
                final_results["batch_analysis"] = batch_analysis

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            # Default output name
            input_name = Path(args.input).stem
            output_path = Path(f"predictions_{input_name}.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_path}")

        # Create detailed report if requested
        if args.create_report:
            logger.info("Creating comprehensive detection report...")

            report_dir = (
                Path(args.output_dir)
                if args.output_dir
                else output_path.parent / "reports"
            )
            report_dir.mkdir(parents=True, exist_ok=True)

            # Create detection report
            report = pipeline.create_detection_report(
                results, save_path=str(report_dir / "detection_report.json")
            )

            # Create visualizations (if matplotlib available)
            try:
                from epra_classifier.evaluation.visualization import (
                    create_evaluation_report,
                )

                # Extract metrics for visualization
                if "batch_analysis" in final_results:
                    metrics = final_results["batch_analysis"]
                    plot_paths = create_evaluation_report(
                        metrics, save_dir=report_dir / "visualizations"
                    )
                    logger.info(
                        f"Visualizations saved to: {report_dir / 'visualizations'}"
                    )

            except ImportError:
                logger.warning("Matplotlib not available - skipping visualizations")

            logger.info(f"Comprehensive report saved to: {report_dir}")

        # Print summary
        violence_count = sum(1 for r in filtered_results if r.get("is_violence", False))
        avg_confidence = (
            sum(r.get("confidence", 0) for r in filtered_results)
            / len(filtered_results)
            if filtered_results
            else 0
        )

        print(f"\n{'=' * 50}")
        print("INFERENCE SUMMARY")
        print(f"{'=' * 50}")
        print(f"Total images processed: {len(results)}")
        print(f"Violence detections: {violence_count}")
        print(
            f"Violence rate: {violence_count / len(results) * 100:.1f}%"
            if results
            else "N/A"
        )
        print(f"Average confidence: {avg_confidence:.3f}")
        print(
            f"Performance: {pipeline.get_performance_stats().get('throughput_fps', 0):.1f} FPS"
        )
        print(f"Results saved to: {output_path}")
        print(f"{'=' * 50}")

    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
