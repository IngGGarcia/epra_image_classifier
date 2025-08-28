"""
Post-processing filters for inference results.

This module provides various filters and aggregators for processing
inference results including confidence filtering and temporal smoothing.
"""

from collections import deque
from typing import Any

import numpy as np

from ...utils.logging import get_logger


class ConfidenceFilter:
    """Filter predictions based on confidence thresholds."""

    def __init__(self, min_confidence: float = 0.5, violence_threshold: float = 0.6):
        """
        Initialize confidence filter.

        Args:
            min_confidence: Minimum confidence for any prediction
            violence_threshold: Special threshold for violence predictions
        """
        self.min_confidence = min_confidence
        self.violence_threshold = violence_threshold
        self.logger = get_logger()

    def filter(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter results based on confidence thresholds.

        Args:
            results: List of prediction results

        Returns:
            Filtered results
        """
        filtered = []

        for result in results:
            confidence = result.get("confidence", 0.0)
            is_violence = result.get("is_violence", False)

            # Apply appropriate threshold
            threshold = self.violence_threshold if is_violence else self.min_confidence

            if confidence >= threshold:
                result["filter_passed"] = True
                filtered.append(result)
            else:
                result["filter_passed"] = False
                result["filter_reason"] = (
                    f"Low confidence: {confidence:.3f} < {threshold:.3f}"
                )

        self.logger.info(
            f"Confidence filter: {len(filtered)}/{len(results)} predictions passed"
        )
        return filtered

    def get_statistics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Get filtering statistics."""
        if not results:
            return {}

        confidences = [r.get("confidence", 0) for r in results]
        passed = [r for r in results if r.get("filter_passed", False)]

        return {
            "total_predictions": len(results),
            "passed_filter": len(passed),
            "filter_rate": len(passed) / len(results),
            "avg_confidence": np.mean(confidences),
            "min_confidence_threshold": self.min_confidence,
            "violence_threshold": self.violence_threshold,
        }


class TemporalSmoothingFilter:
    """Apply temporal smoothing to predictions over time."""

    def __init__(self, window_size: int = 5, smoothing_factor: float = 0.7):
        """
        Initialize temporal smoothing filter.

        Args:
            window_size: Size of the temporal window
            smoothing_factor: Weight for smoothing (0-1)
        """
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.history = deque(maxlen=window_size)
        self.logger = get_logger()

    def add_prediction(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Add a prediction and apply temporal smoothing.

        Args:
            result: Prediction result

        Returns:
            Smoothed prediction result
        """
        # Extract key metrics
        violence_prob = result.get("violence_probability", 0.0)
        predicted_class = result.get("predicted_class", 0)
        confidence = result.get("confidence", 0.0)

        # Add to history
        self.history.append(
            {
                "violence_probability": violence_prob,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "timestamp": result.get("timestamp", 0),
            }
        )

        # Apply smoothing if we have enough history
        if len(self.history) >= 2:
            smoothed_result = self._apply_smoothing(result)
        else:
            smoothed_result = result.copy()
            smoothed_result["smoothed"] = False

        return smoothed_result

    def _apply_smoothing(self, current_result: dict[str, Any]) -> dict[str, Any]:
        """Apply temporal smoothing algorithm."""
        # Calculate weighted averages
        weights = np.array(
            [
                self.smoothing_factor ** (len(self.history) - i - 1)
                for i in range(len(self.history))
            ]
        )
        weights = weights / np.sum(weights)

        # Smooth violence probability
        violence_probs = [h["violence_probability"] for h in self.history]
        smoothed_violence_prob = np.average(violence_probs, weights=weights)

        # Smooth confidence
        confidences = [h["confidence"] for h in self.history]
        smoothed_confidence = np.average(confidences, weights=weights)

        # Determine smoothed class (majority vote with recency weighting)
        class_votes = {}
        for i, h in enumerate(self.history):
            cls = h["predicted_class"]
            if cls not in class_votes:
                class_votes[cls] = 0
            class_votes[cls] += weights[i]

        smoothed_class = max(class_votes.items(), key=lambda x: x[1])[0]

        # Create smoothed result
        smoothed_result = current_result.copy()
        smoothed_result.update(
            {
                "smoothed": True,
                "original_violence_probability": current_result.get(
                    "violence_probability", 0
                ),
                "original_confidence": current_result.get("confidence", 0),
                "original_predicted_class": current_result.get("predicted_class", 0),
                "violence_probability": float(smoothed_violence_prob),
                "confidence": float(smoothed_confidence),
                "predicted_class": int(smoothed_class),
                "smoothing_window_size": len(self.history),
            }
        )

        return smoothed_result

    def reset(self) -> None:
        """Reset the temporal history."""
        self.history.clear()
        self.logger.info("Temporal smoothing filter reset")


class ResultAggregator:
    """Aggregate and analyze collections of prediction results."""

    def __init__(self):
        """Initialize result aggregator."""
        self.logger = get_logger()

    def aggregate_by_timeframe(
        self, results: list[dict[str, Any]], timeframe_seconds: float = 60.0
    ) -> list[dict[str, Any]]:
        """
        Aggregate results by time windows.

        Args:
            results: List of prediction results
            timeframe_seconds: Size of time window in seconds

        Returns:
            List of aggregated results per timeframe
        """
        if not results:
            return []

        # Sort by timestamp
        sorted_results = sorted(results, key=lambda x: x.get("timestamp", 0))

        # Group by timeframes
        timeframes = []
        current_frame = []
        frame_start = sorted_results[0].get("timestamp", 0)

        for result in sorted_results:
            timestamp = result.get("timestamp", 0)

            if timestamp - frame_start <= timeframe_seconds:
                current_frame.append(result)
            else:
                # Process current frame
                if current_frame:
                    timeframes.append(self._aggregate_frame(current_frame, frame_start))

                # Start new frame
                current_frame = [result]
                frame_start = timestamp

        # Process last frame
        if current_frame:
            timeframes.append(self._aggregate_frame(current_frame, frame_start))

        return timeframes

    def _aggregate_frame(
        self, frame_results: list[dict[str, Any]], start_time: float
    ) -> dict[str, Any]:
        """Aggregate results within a single timeframe."""
        if not frame_results:
            return {}

        # Count predictions by class
        class_counts = {}
        violence_count = 0
        total_confidence = 0
        violence_probabilities = []
        high_confidence_count = 0

        for result in frame_results:
            # Class counting
            class_name = result.get("predicted_class_name", "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Violence analysis
            if result.get("is_violence", False):
                violence_count += 1
                violence_probabilities.append(result.get("violence_probability", 0))

            # Confidence analysis
            confidence = result.get("confidence", 0)
            total_confidence += confidence

            if confidence >= 0.7:  # High confidence threshold
                high_confidence_count += 1

        # Calculate aggregated metrics
        frame_analysis = {
            "timeframe_start": start_time,
            "timeframe_duration": frame_results[-1].get("timestamp", start_time)
            - start_time,
            "total_predictions": len(frame_results),
            "class_distribution": class_counts,
            "violence_detections": violence_count,
            "violence_ratio": violence_count / len(frame_results),
            "average_confidence": total_confidence / len(frame_results),
            "high_confidence_ratio": high_confidence_count / len(frame_results),
        }

        if violence_probabilities:
            frame_analysis.update(
                {
                    "avg_violence_probability": np.mean(violence_probabilities),
                    "max_violence_probability": np.max(violence_probabilities),
                    "violence_probability_std": np.std(violence_probabilities),
                }
            )

        # Risk assessment for timeframe
        if frame_analysis["violence_ratio"] >= 0.5:
            frame_analysis["risk_level"] = "high"
        elif frame_analysis["violence_ratio"] >= 0.2:
            frame_analysis["risk_level"] = "medium"
        else:
            frame_analysis["risk_level"] = "low"

        return frame_analysis

    def analyze_trends(self, aggregated_frames: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze trends across multiple timeframes."""
        if len(aggregated_frames) < 2:
            return {"message": "Need at least 2 timeframes for trend analysis"}

        # Extract time series data
        violence_ratios = [f.get("violence_ratio", 0) for f in aggregated_frames]
        avg_confidences = [f.get("average_confidence", 0) for f in aggregated_frames]

        # Calculate trends
        violence_trend = self._calculate_trend(violence_ratios)
        confidence_trend = self._calculate_trend(avg_confidences)

        # Risk escalation analysis
        risk_levels = [f.get("risk_level", "low") for f in aggregated_frames]
        risk_escalation = self._analyze_risk_escalation(risk_levels)

        return {
            "timeframes_analyzed": len(aggregated_frames),
            "violence_trend": violence_trend,
            "confidence_trend": confidence_trend,
            "risk_escalation": risk_escalation,
            "overall_violence_rate": np.mean(violence_ratios),
            "violence_rate_std": np.std(violence_ratios),
            "peak_violence_timeframe": np.argmax(violence_ratios),
            "current_risk_level": risk_levels[-1] if risk_levels else "unknown",
        }

    def _calculate_trend(self, values: list[float]) -> dict[str, Any]:
        """Calculate trend for a time series."""
        if len(values) < 2:
            return {"direction": "stable", "slope": 0}

        # Simple linear regression
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]

        # Determine trend direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "direction": direction,
            "slope": float(slope),
            "start_value": values[0],
            "end_value": values[-1],
            "change_percentage": ((values[-1] - values[0]) / values[0] * 100)
            if values[0] != 0
            else 0,
        }

    def _analyze_risk_escalation(self, risk_levels: list[str]) -> dict[str, Any]:
        """Analyze risk level escalation patterns."""
        risk_map = {"low": 1, "medium": 2, "high": 3}
        risk_values = [risk_map.get(level, 1) for level in risk_levels]

        escalations = 0
        de_escalations = 0

        for i in range(1, len(risk_values)):
            if risk_values[i] > risk_values[i - 1]:
                escalations += 1
            elif risk_values[i] < risk_values[i - 1]:
                de_escalations += 1

        return {
            "escalations": escalations,
            "de_escalations": de_escalations,
            "net_escalation": escalations - de_escalations,
            "current_risk": risk_levels[-1] if risk_levels else "unknown",
            "peak_risk": max(risk_levels, key=lambda x: risk_map.get(x, 0))
            if risk_levels
            else "unknown",
            "escalation_rate": escalations / (len(risk_levels) - 1)
            if len(risk_levels) > 1
            else 0,
        }
