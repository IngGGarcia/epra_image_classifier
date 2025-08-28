"""
Violence inference pipeline for specialized violence detection.

This module provides optimized inference for violence classification
with violence-specific result formatting and analysis.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ...utils.config import Config, InferenceConfig
from .base_inference import BaseInferencePipeline, InferenceError


class ViolenceInferencePipeline(BaseInferencePipeline):
    """
    Specialized inference pipeline for violence detection.

    This pipeline provides violence-specific inference capabilities including
    violence level prediction, confidence analysis, and detailed reporting.
    """

    def __init__(self, config: Config | InferenceConfig):
        """
        Initialize violence inference pipeline.

        Args:
            config: Configuration object
        """
        super().__init__(config)

        # Violence-specific settings
        self.confidence_threshold = self.inference_config.confidence_threshold
        self.class_names = self._get_class_names()

        self.logger.info(
            f"Violence inference pipeline initialized with {len(self.class_names)} classes"
        )

    def _get_class_names(self) -> list[str]:
        """Get class names based on model type."""
        if hasattr(self.config, "model") and self.config.model.num_classes == 2:
            return ["non_violence", "violence"]
        elif hasattr(self.config, "evaluation") and self.config.evaluation.class_names:
            return self.config.evaluation.class_names
        else:
            # Default multiclass names
            return [f"level_{i}" for i in range(5)]

    def predict_single(self, image_path: str | Path) -> dict[str, Any]:
        """
        Predict violence level for a single image.

        Args:
            image_path: Path to image file

        Returns:
            Violence prediction results
        """
        if self.model is None:
            raise InferenceError("Model not loaded")

        image_path = Path(image_path)

        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Inference
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(image_tensor)
            logits = outputs["logits"]

            # Get predictions and probabilities
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        # Format result
        result = self._format_prediction_result(
            path=image_path,
            predicted_class=predicted_class,
            probabilities=probabilities[0].cpu().numpy(),
            inference_time=inference_time,
            features=outputs.get("features")[0] if "features" in outputs else None,
        )

        return result

    def _format_prediction_result(
        self,
        path: str | Path,
        predicted_class: int,
        probabilities: np.ndarray,
        inference_time: float,
        features: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Format violence prediction result with detailed analysis.

        Args:
            path: Image path
            predicted_class: Predicted class index
            probabilities: Class probabilities
            inference_time: Inference time
            features: Extracted features (optional)

        Returns:
            Formatted violence prediction result
        """
        # Basic prediction info
        result = {
            "image_path": str(path),
            "predicted_class": predicted_class,
            "predicted_class_name": self.class_names[predicted_class]
            if predicted_class < len(self.class_names)
            else f"class_{predicted_class}",
            "confidence": float(probabilities[predicted_class]),
            "inference_time_ms": inference_time * 1000,
            "timestamp": time.time(),
        }

        # Add probabilities if requested
        if self.inference_config.return_probabilities:
            class_probs = {}
            for i, prob in enumerate(probabilities):
                class_name = (
                    self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                )
                class_probs[class_name] = float(prob)
            result["class_probabilities"] = class_probs

        # Violence-specific analysis
        violence_analysis = self._analyze_violence_prediction(
            probabilities, predicted_class
        )
        result.update(violence_analysis)

        # Add features if requested
        if self.inference_config.return_features and features is not None:
            result["features"] = (
                features.cpu().numpy().tolist()
                if isinstance(features, torch.Tensor)
                else features
            )

        # Confidence-based filtering
        result["high_confidence"] = result["confidence"] >= self.confidence_threshold

        return result

    def _analyze_violence_prediction(
        self, probabilities: np.ndarray, predicted_class: int
    ) -> dict[str, Any]:
        """
        Analyze violence prediction for additional insights.

        Args:
            probabilities: Class probabilities
            predicted_class: Predicted class

        Returns:
            Violence analysis results
        """
        analysis = {}

        if len(self.class_names) == 2:
            # Binary violence classification
            violence_prob = probabilities[1]
            non_violence_prob = probabilities[0]

            analysis.update(
                {
                    "is_violence": predicted_class == 1,
                    "violence_probability": float(violence_prob),
                    "non_violence_probability": float(non_violence_prob),
                    "violence_confidence_level": self._get_confidence_level(
                        violence_prob
                    ),
                    "risk_assessment": self._get_risk_assessment(violence_prob),
                }
            )

        else:
            # Multiclass violence levels
            violence_prob = np.sum(probabilities[1:])  # Sum of all violence levels
            violence_level = predicted_class if predicted_class > 0 else 0

            analysis.update(
                {
                    "is_violence": predicted_class > 0,
                    "violence_probability": float(violence_prob),
                    "non_violence_probability": float(probabilities[0]),
                    "violence_level": violence_level,
                    "violence_severity": self._get_violence_severity(violence_level),
                    "confidence_level": self._get_confidence_level(
                        probabilities[predicted_class]
                    ),
                    "risk_assessment": self._get_multiclass_risk_assessment(
                        probabilities, predicted_class
                    ),
                }
            )

            # Level-specific probabilities
            level_probs = {}
            for i, prob in enumerate(probabilities):
                level_probs[f"level_{i}_probability"] = float(prob)
            analysis["level_probabilities"] = level_probs

        # Uncertainty analysis
        analysis["prediction_uncertainty"] = self._calculate_uncertainty(probabilities)

        return analysis

    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description."""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"

    def _get_risk_assessment(self, violence_prob: float) -> str:
        """Get risk assessment for binary classification."""
        if violence_prob >= 0.8:
            return "high_risk"
        elif violence_prob >= 0.6:
            return "medium_risk"
        elif violence_prob >= 0.4:
            return "low_risk"
        else:
            return "minimal_risk"

    def _get_multiclass_risk_assessment(
        self, probabilities: np.ndarray, predicted_class: int
    ) -> str:
        """Get risk assessment for multiclass classification."""
        if predicted_class == 0:
            return "no_risk"
        elif predicted_class == 1:
            return "low_risk"
        elif predicted_class == 2:
            return "medium_risk"
        elif predicted_class == 3:
            return "high_risk"
        else:
            return "critical_risk"

    def _get_violence_severity(self, level: int) -> str:
        """Get violence severity description."""
        severity_map = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "extreme"}
        return severity_map.get(level, "unknown")

    def _calculate_uncertainty(self, probabilities: np.ndarray) -> float:
        """Calculate prediction uncertainty using entropy."""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        probs = np.clip(probabilities, epsilon, 1.0 - epsilon)

        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs))

        # Normalize by max entropy for this number of classes
        max_entropy = np.log(len(probabilities))
        normalized_uncertainty = entropy / max_entropy if max_entropy > 0 else 0

        return float(normalized_uncertainty)

    def predict_violence_batch(
        self, image_paths: list[str | Path], filter_high_confidence: bool = False
    ) -> dict[str, Any]:
        """
        Predict violence for a batch of images with aggregated results.

        Args:
            image_paths: List of image paths
            filter_high_confidence: Whether to filter high confidence predictions

        Returns:
            Aggregated batch results
        """
        individual_results = self.predict_batch(image_paths)

        # Filter if requested
        if filter_high_confidence:
            individual_results = [
                r for r in individual_results if r.get("high_confidence", False)
            ]

        # Aggregate results
        batch_analysis = self._analyze_batch_results(individual_results)

        return {
            "individual_predictions": individual_results,
            "batch_analysis": batch_analysis,
            "total_images": len(image_paths),
            "processed_images": len(individual_results),
            "high_confidence_predictions": len(
                [r for r in individual_results if r.get("high_confidence", False)]
            ),
        }

    def _analyze_batch_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze batch results for aggregate insights."""
        if not results:
            return {"message": "No valid predictions"}

        # Count predictions by class
        class_counts = {}
        violence_count = 0
        total_confidence = 0
        violence_probabilities = []

        for result in results:
            class_name = result.get("predicted_class_name", "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            if result.get("is_violence", False):
                violence_count += 1
                violence_probabilities.append(result.get("violence_probability", 0))

            total_confidence += result.get("confidence", 0)

        # Calculate statistics
        avg_confidence = total_confidence / len(results)
        violence_ratio = violence_count / len(results)

        analysis = {
            "total_predictions": len(results),
            "class_distribution": class_counts,
            "violence_detections": violence_count,
            "violence_ratio": violence_ratio,
            "average_confidence": avg_confidence,
        }

        if violence_probabilities:
            analysis.update(
                {
                    "avg_violence_probability": np.mean(violence_probabilities),
                    "max_violence_probability": np.max(violence_probabilities),
                    "min_violence_probability": np.min(violence_probabilities),
                }
            )

        # Risk assessment
        if violence_ratio >= 0.7:
            analysis["batch_risk_level"] = "high"
        elif violence_ratio >= 0.3:
            analysis["batch_risk_level"] = "medium"
        else:
            analysis["batch_risk_level"] = "low"

        return analysis

    def create_detection_report(
        self, results: list[dict[str, Any]], save_path: str | None = None
    ) -> dict[str, Any]:
        """
        Create a comprehensive detection report.

        Args:
            results: Prediction results
            save_path: Path to save report (optional)

        Returns:
            Comprehensive report
        """
        if not results:
            return {"error": "No results to analyze"}

        # Performance statistics
        perf_stats = self.get_performance_stats()

        # Batch analysis
        batch_analysis = self._analyze_batch_results(results)

        # High-risk detections
        high_risk_detections = [
            r
            for r in results
            if r.get("risk_assessment") in ["high_risk", "critical_risk"]
        ]

        # Create comprehensive report
        report = {
            "summary": {
                "total_images": len(results),
                "violence_detections": batch_analysis.get("violence_detections", 0),
                "high_risk_detections": len(high_risk_detections),
                "average_confidence": batch_analysis.get("average_confidence", 0),
                "processing_time_total": sum(
                    r.get("inference_time_ms", 0) for r in results
                ),
            },
            "performance": perf_stats,
            "batch_analysis": batch_analysis,
            "high_risk_detections": high_risk_detections,
            "detailed_results": results,
            "generated_at": time.time(),
        }

        # Save report if path provided
        if save_path:
            import json

            with open(save_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Detection report saved to: {save_path}")

        return report


# Factory functions for easy creation
def create_violence_inference_pipeline(
    checkpoint_path: str,
    device: str = "auto",
    batch_size: int = 1,
    confidence_threshold: float = 0.5,
    return_probabilities: bool = True,
    return_features: bool = False,
) -> ViolenceInferencePipeline:
    """
    Create a violence inference pipeline with common settings.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to use for inference
        batch_size: Batch size for inference
        confidence_threshold: Confidence threshold for predictions
        return_probabilities: Whether to return class probabilities
        return_features: Whether to return extracted features

    Returns:
        Configured violence inference pipeline
    """
    from ...utils.config import Config, InferenceConfig

    config = Config()
    config.inference = InferenceConfig(
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        return_probabilities=return_probabilities,
        return_features=return_features,
    )

    return ViolenceInferencePipeline(config)
