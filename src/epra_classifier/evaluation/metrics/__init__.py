"""
Metrics module for model evaluation.

Provides comprehensive metrics calculation for both binary and multiclass
violence classification tasks.
"""

from .classification_metrics import (
    BinaryMetrics,
    ClassificationMetrics,
    MulticlassMetrics,
    calculate_all_metrics,
    format_metrics_report,
)


__all__ = [
    "ClassificationMetrics",
    "BinaryMetrics",
    "MulticlassMetrics",
    "calculate_all_metrics",
    "format_metrics_report",
]
