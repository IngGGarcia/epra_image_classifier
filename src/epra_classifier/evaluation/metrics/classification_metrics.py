"""
Comprehensive classification metrics for violence detection.

This module provides detailed metrics calculation for both binary and
multiclass violence classification with specialized violence-specific metrics.
"""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ...utils.logging import get_logger


class ClassificationMetrics:
    """Base class for classification metrics calculation."""

    def __init__(self, num_classes: int):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.logger = get_logger()

    def calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Calculate basic classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)

        Returns:
            Dictionary of basic metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Add AUC if probabilities are provided
        if y_prob is not None:
            try:
                if self.num_classes == 2:
                    metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics["auc"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="weighted"
                    )
            except ValueError:
                metrics["auc"] = 0.0

        return metrics

    def calculate_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    def calculate_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Calculate per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with per-class metrics
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        return {"precision": precision, "recall": recall, "f1": f1, "support": support}


class BinaryMetrics(ClassificationMetrics):
    """Specialized metrics for binary violence classification."""

    def __init__(self):
        """Initialize binary metrics calculator."""
        super().__init__(num_classes=2)

    def calculate_binary_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Calculate comprehensive binary classification metrics.

        Args:
            y_true: True labels (0: non-violence, 1: violence)
            y_pred: Predicted labels
            y_prob: Predicted probabilities

        Returns:
            Comprehensive metrics dictionary
        """
        # Basic metrics
        metrics = self.calculate_basic_metrics(y_true, y_pred, y_prob)

        # Binary-specific metrics
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Violence detection metrics
        violence_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        violence_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        violence_f1 = (
            2
            * (violence_precision * violence_recall)
            / (violence_precision + violence_recall)
            if (violence_precision + violence_recall) > 0
            else 0.0
        )

        # Non-violence metrics
        non_violence_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        non_violence_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics.update(
            {
                # Confusion matrix components
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                # Violence-specific metrics
                "violence_precision": violence_precision,
                "violence_recall": violence_recall,
                "violence_f1": violence_f1,
                # Non-violence metrics
                "non_violence_precision": non_violence_precision,
                "non_violence_recall": non_violence_recall,
                # Additional binary metrics
                "specificity": specificity,
                "sensitivity": violence_recall,  # Same as recall for violence class
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            }
        )

        # ROC and PR curves if probabilities provided
        if y_prob is not None:
            roc_metrics = self._calculate_roc_metrics(y_true, y_prob)
            pr_metrics = self._calculate_pr_metrics(y_true, y_prob)
            metrics.update(roc_metrics)
            metrics.update(pr_metrics)

        return metrics

    def _calculate_roc_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> dict[str, Any]:
        """Calculate ROC curve metrics."""
        try:
            # Use violence class probabilities
            violence_probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob

            fpr, tpr, thresholds = roc_curve(y_true, violence_probs)
            roc_auc = auc(fpr, tpr)

            # Find optimal threshold (Youden's index)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            return {
                "roc_auc": roc_auc,
                "roc_fpr": fpr.tolist(),
                "roc_tpr": tpr.tolist(),
                "roc_thresholds": thresholds.tolist(),
                "optimal_threshold": optimal_threshold,
                "optimal_tpr": tpr[optimal_idx],
                "optimal_fpr": fpr[optimal_idx],
            }
        except Exception as e:
            self.logger.warning(f"Could not calculate ROC metrics: {e}")
            return {"roc_auc": 0.0}

    def _calculate_pr_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> dict[str, Any]:
        """Calculate Precision-Recall curve metrics."""
        try:
            # Use violence class probabilities
            violence_probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob

            precision, recall, thresholds = precision_recall_curve(
                y_true, violence_probs
            )
            pr_auc = auc(recall, precision)

            # Find threshold for best F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_f1_idx = np.argmax(f1_scores)
            best_threshold = (
                thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
            )

            return {
                "pr_auc": pr_auc,
                "pr_precision": precision.tolist(),
                "pr_recall": recall.tolist(),
                "pr_thresholds": thresholds.tolist(),
                "best_f1_threshold": best_threshold,
                "best_f1_score": f1_scores[best_f1_idx],
                "best_f1_precision": precision[best_f1_idx],
                "best_f1_recall": recall[best_f1_idx],
            }
        except Exception as e:
            self.logger.warning(f"Could not calculate PR metrics: {e}")
            return {"pr_auc": 0.0}

    def calculate_violence_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Calculate violence-specific analysis metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities

        Returns:
            Violence analysis metrics
        """
        violence_indices = np.where(y_true == 1)[0]
        non_violence_indices = np.where(y_true == 0)[0]

        analysis = {
            "total_samples": len(y_true),
            "violence_samples": len(violence_indices),
            "non_violence_samples": len(non_violence_indices),
            "violence_ratio": len(violence_indices) / len(y_true),
            "class_balance": min(len(violence_indices), len(non_violence_indices))
            / max(len(violence_indices), len(non_violence_indices))
            if max(len(violence_indices), len(non_violence_indices)) > 0
            else 0,
        }

        if (
            y_prob is not None
            and len(violence_indices) > 0
            and len(non_violence_indices) > 0
        ):
            violence_probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob

            analysis.update(
                {
                    "avg_violence_confidence": np.mean(
                        violence_probs[violence_indices]
                    ),
                    "avg_non_violence_confidence": np.mean(
                        1 - violence_probs[non_violence_indices]
                    ),
                    "violence_confidence_std": np.std(violence_probs[violence_indices]),
                    "non_violence_confidence_std": np.std(
                        1 - violence_probs[non_violence_indices]
                    ),
                }
            )

        return analysis


class MulticlassMetrics(ClassificationMetrics):
    """Specialized metrics for multiclass violence level classification."""

    def __init__(self, num_classes: int = 5):
        """
        Initialize multiclass metrics calculator.

        Args:
            num_classes: Number of violence levels
        """
        super().__init__(num_classes=num_classes)

    def calculate_multiclass_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        class_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate comprehensive multiclass metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            class_names: Names of classes

        Returns:
            Comprehensive metrics dictionary
        """
        if class_names is None:
            class_names = [f"level_{i}" for i in range(self.num_classes)]

        # Basic metrics
        metrics = self.calculate_basic_metrics(y_true, y_pred, y_prob)

        # Per-class metrics
        per_class = self.calculate_per_class_metrics(y_true, y_pred)

        # Add per-class metrics with class names
        for i, class_name in enumerate(class_names):
            if i < len(per_class["precision"]):
                metrics[f"{class_name}_precision"] = per_class["precision"][i]
                metrics[f"{class_name}_recall"] = per_class["recall"][i]
                metrics[f"{class_name}_f1"] = per_class["f1"][i]
                metrics[f"{class_name}_support"] = int(per_class["support"][i])

        # Confusion matrix
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Violence-specific metrics (treating level 0 as non-violence)
        violence_metrics = self._calculate_violence_level_metrics(
            y_true, y_pred, y_prob
        )
        metrics.update(violence_metrics)

        # Class distribution analysis
        distribution_metrics = self._calculate_class_distribution(
            y_true, y_pred, class_names
        )
        metrics.update(distribution_metrics)

        return metrics

    def _calculate_violence_level_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Calculate violence level specific metrics."""
        # Binary violence detection (0 vs 1+)
        binary_true = (np.array(y_true) > 0).astype(int)
        binary_pred = (np.array(y_pred) > 0).astype(int)

        binary_metrics = {
            "violence_detection_accuracy": accuracy_score(binary_true, binary_pred),
            "violence_detection_precision": precision_score(
                binary_true, binary_pred, zero_division=0
            ),
            "violence_detection_recall": recall_score(
                binary_true, binary_pred, zero_division=0
            ),
            "violence_detection_f1": f1_score(
                binary_true, binary_pred, zero_division=0
            ),
        }

        # Severity prediction accuracy (for violence cases only)
        violence_indices = np.where(np.array(y_true) > 0)[0]
        if len(violence_indices) > 0:
            violence_true = np.array(y_true)[violence_indices]
            violence_pred = np.array(y_pred)[violence_indices]

            binary_metrics.update(
                {
                    "severity_accuracy": accuracy_score(violence_true, violence_pred),
                    "severity_mae": np.mean(np.abs(violence_true - violence_pred)),
                    "severity_mse": np.mean((violence_true - violence_pred) ** 2),
                }
            )

        # Violence probability analysis
        if y_prob is not None:
            # Sum of all violence level probabilities
            violence_prob = (
                np.sum(y_prob[:, 1:], axis=1) if y_prob.shape[1] > 1 else y_prob
            )

            try:
                binary_metrics["violence_auc"] = roc_auc_score(
                    binary_true, violence_prob
                )
            except ValueError:
                binary_metrics["violence_auc"] = 0.0

        return binary_metrics

    def _calculate_class_distribution(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
    ) -> dict[str, Any]:
        """Calculate class distribution metrics."""
        true_dist = np.bincount(y_true, minlength=self.num_classes)
        pred_dist = np.bincount(y_pred, minlength=self.num_classes)

        distribution = {
            "true_distribution": (true_dist / len(y_true)).tolist(),
            "predicted_distribution": (pred_dist / len(y_pred)).tolist(),
        }

        # Per-class sample counts
        for i, class_name in enumerate(class_names):
            if i < len(true_dist):
                distribution[f"{class_name}_true_count"] = int(true_dist[i])
                distribution[f"{class_name}_pred_count"] = int(pred_dist[i])

        return distribution


def calculate_all_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    y_prob: torch.Tensor | np.ndarray | None = None,
    num_classes: int | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Calculate all appropriate metrics based on the number of classes.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        num_classes: Number of classes (auto-detected if None)
        class_names: Class names (auto-generated if None)

    Returns:
        Comprehensive metrics dictionary
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    # Determine number of classes
    if num_classes is None:
        num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))

    # Calculate metrics based on classification type
    if num_classes == 2:
        calculator = BinaryMetrics()
        return calculator.calculate_binary_metrics(y_true, y_pred, y_prob)
    else:
        calculator = MulticlassMetrics(num_classes)
        return calculator.calculate_multiclass_metrics(
            y_true, y_pred, y_prob, class_names
        )


def format_metrics_report(
    metrics: dict[str, Any], title: str = "Classification Report"
) -> str:
    """
    Format metrics into a readable report.

    Args:
        metrics: Metrics dictionary
        title: Report title

    Returns:
        Formatted report string
    """
    report = f"\n{'=' * 60}\n{title.center(60)}\n{'=' * 60}\n\n"

    # Main metrics
    main_metrics = ["accuracy", "precision", "recall", "f1"]
    for metric in main_metrics:
        if metric in metrics:
            report += f"{metric.capitalize():20}: {metrics[metric]:.4f}\n"

    # AUC metrics
    auc_metrics = ["auc", "roc_auc", "pr_auc", "violence_auc"]
    auc_found = False
    for metric in auc_metrics:
        if metric in metrics:
            if not auc_found:
                report += "\nAUC Metrics:\n" + "-" * 20 + "\n"
                auc_found = True
            report += f"{metric.upper():20}: {metrics[metric]:.4f}\n"

    # Violence-specific metrics
    violence_metrics = [
        k for k in metrics.keys() if "violence" in k and k not in auc_metrics
    ]
    if violence_metrics:
        report += "\nViolence Detection:\n" + "-" * 20 + "\n"
        for metric in violence_metrics:
            if isinstance(metrics[metric], (int, float)):
                report += (
                    f"{metric.replace('_', ' ').title():20}: {metrics[metric]:.4f}\n"
                )

    # Class-specific metrics (for multiclass)
    class_metrics = [
        k for k in metrics.keys() if any(cls in k for cls in ["level_", "class_"])
    ]
    if class_metrics:
        report += "\nPer-Class Metrics:\n" + "-" * 20 + "\n"
        # Group by class
        classes = set()
        for metric in class_metrics:
            for part in metric.split("_"):
                if part.startswith("level") or part.startswith("class"):
                    classes.add("_".join(metric.split("_")[:2]))

        for class_name in sorted(classes):
            precision = metrics.get(f"{class_name}_precision", 0)
            recall = metrics.get(f"{class_name}_recall", 0)
            f1 = metrics.get(f"{class_name}_f1", 0)
            support = metrics.get(f"{class_name}_support", 0)

            report += f"{class_name:15}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} S={support}\n"

    report += "\n" + "=" * 60 + "\n"
    return report
