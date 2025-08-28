"""
Visualization module for model evaluation.

Provides comprehensive visualization tools for model performance analysis
including confusion matrices, ROC curves, and training plots.
"""

from .plots import (
    PlotManager,
    create_evaluation_report,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_history,
)


__all__ = [
    "PlotManager",
    "create_evaluation_report",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_training_history",
    "plot_class_distribution",
]
