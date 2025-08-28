"""
Comprehensive plotting utilities for model evaluation.

This module provides visualization tools for training monitoring,
performance analysis, and result interpretation.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from ...utils.config import EvaluationConfig
from ...utils.logging import get_logger


class PlotManager:
    """Manager for creating and saving evaluation plots."""

    def __init__(
        self, config: EvaluationConfig | None = None, save_dir: str | Path | None = None
    ):
        """
        Initialize plot manager.

        Args:
            config: Evaluation configuration
            save_dir: Directory to save plots
        """
        self.config = config or EvaluationConfig()
        self.save_dir = Path(save_dir) if save_dir else Path("plots")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()

        # Set plotting style
        plt.style.use(
            "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
        )
        sns.set_palette("husl")

    def save_plot(self, fig: Figure, filename: str, dpi: int = 300) -> str:
        """
        Save plot to file.

        Args:
            fig: Matplotlib figure
            filename: Filename (without extension)
            dpi: Resolution for saving

        Returns:
            Path to saved file
        """
        if not filename.endswith((".png", ".jpg", ".pdf", ".svg")):
            filename += ".png"

        filepath = self.save_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
        self.logger.info(f"Plot saved: {filepath}")

        if self.config.save_plots:
            return str(filepath)
        else:
            # Clean up if not saving
            filepath.unlink(missing_ok=True)
            return ""


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: str | None = None,
) -> Figure:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Class names for labels
        title: Plot title
        normalize: Whether to normalize values
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curve(
    fpr: list[float] | np.ndarray,
    tpr: list[float] | np.ndarray,
    auc_score: float,
    title: str = "ROC Curve",
    save_path: str | None = None,
) -> Figure:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        title: Plot title
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(
    precision: list[float] | np.ndarray,
    recall: list[float] | np.ndarray,
    auc_score: float,
    title: str = "Precision-Recall Curve",
    save_path: str | None = None,
) -> Figure:
    """
    Plot Precision-Recall curve.

    Args:
        precision: Precision values
        recall: Recall values
        auc_score: PR AUC score
        title: Plot title
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, linewidth=2, label=f"PR Curve (AUC = {auc_score:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    save_path: str | None = None,
) -> Figure:
    """
    Plot training history.

    Args:
        history: Training history dictionary
        title: Plot title
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    epochs = history.get("epochs", range(len(history.get("train_loss", []))))

    # Loss plot
    ax = axes[0, 0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Model Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot (if available)
    ax = axes[0, 1]
    train_acc = []
    val_acc = []

    if "val_metrics" in history:
        for metrics in history["val_metrics"]:
            if isinstance(metrics, dict) and "accuracy" in metrics:
                val_acc.append(metrics["accuracy"])

    if val_acc:
        ax.plot(
            epochs[: len(val_acc)], val_acc, label="Validation Accuracy", linewidth=2
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate plot
    ax = axes[1, 0]
    if "learning_rates" in history:
        ax.plot(epochs, history["learning_rates"], linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    # F1 Score plot (if available)
    ax = axes[1, 1]
    val_f1 = []

    if "val_metrics" in history:
        for metrics in history["val_metrics"]:
            if isinstance(metrics, dict) and "f1" in metrics:
                val_f1.append(metrics["f1"])

    if val_f1:
        ax.plot(epochs[: len(val_f1)], val_f1, label="Validation F1", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_title("Model F1 Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_class_distribution(
    true_dist: list[float] | np.ndarray,
    pred_dist: list[float] | np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Class Distribution",
    save_path: str | None = None,
) -> Figure:
    """
    Plot class distribution comparison.

    Args:
        true_dist: True class distribution
        pred_dist: Predicted class distribution
        class_names: Class names
        title: Plot title
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(true_dist))]

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2, true_dist, width, label="True Distribution", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2, pred_dist, width, label="Predicted Distribution", alpha=0.8
    )

    ax.set_xlabel("Classes", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_violence_analysis(
    metrics: dict[str, Any],
    title: str = "Violence Detection Analysis",
    save_path: str | None = None,
) -> Figure:
    """
    Plot violence-specific analysis.

    Args:
        metrics: Metrics dictionary with violence analysis
        title: Plot title
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Violence vs Non-violence distribution
    ax = axes[0, 0]
    if "violence_samples" in metrics and "non_violence_samples" in metrics:
        labels = ["Non-Violence", "Violence"]
        sizes = [metrics["non_violence_samples"], metrics["violence_samples"]]
        colors = ["lightblue", "lightcoral"]

        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("Dataset Distribution")

    # Performance metrics
    ax = axes[0, 1]
    perf_metrics = ["accuracy", "precision", "recall", "f1"]
    values = [metrics.get(metric, 0) for metric in perf_metrics]

    bars = ax.bar(
        perf_metrics, values, color=["skyblue", "lightgreen", "orange", "pink"]
    )
    ax.set_ylabel("Score")
    ax.set_title("Performance Metrics")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Confusion matrix visualization (if available)
    ax = axes[1, 0]
    if "true_positives" in metrics:
        cm_data = np.array(
            [
                [metrics.get("true_negatives", 0), metrics.get("false_positives", 0)],
                [metrics.get("false_negatives", 0), metrics.get("true_positives", 0)],
            ]
        )

        sns.heatmap(
            cm_data,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Non-Violence", "Predicted Violence"],
            yticklabels=["Actual Non-Violence", "Actual Violence"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix")

    # ROC curve (if available)
    ax = axes[1, 1]
    if "roc_fpr" in metrics and "roc_tpr" in metrics:
        fpr = metrics["roc_fpr"]
        tpr = metrics["roc_tpr"]
        auc_score = metrics.get("roc_auc", 0)

        ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc_score:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_evaluation_report(
    metrics: dict[str, Any],
    history: dict[str, list[float]] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, str]:
    """
    Create comprehensive evaluation report with all plots.

    Args:
        metrics: Evaluation metrics
        history: Training history (optional)
        save_dir: Directory to save plots

    Returns:
        Dictionary of saved plot paths
    """
    if save_dir is None:
        save_dir = Path("evaluation_report")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_plots = {}

    # Confusion matrix
    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        class_names = None

        # Try to get class names from metrics
        if "true_distribution" in metrics:
            num_classes = len(metrics["true_distribution"])
            class_names = [f"Level {i}" for i in range(num_classes)]

        fig = plot_confusion_matrix(
            cm, class_names, save_path=str(save_dir / "confusion_matrix.png")
        )
        saved_plots["confusion_matrix"] = str(save_dir / "confusion_matrix.png")
        plt.close(fig)

    # ROC curve
    if "roc_fpr" in metrics and "roc_tpr" in metrics:
        fig = plot_roc_curve(
            metrics["roc_fpr"],
            metrics["roc_tpr"],
            metrics.get("roc_auc", 0),
            save_path=str(save_dir / "roc_curve.png"),
        )
        saved_plots["roc_curve"] = str(save_dir / "roc_curve.png")
        plt.close(fig)

    # PR curve
    if "pr_precision" in metrics and "pr_recall" in metrics:
        fig = plot_precision_recall_curve(
            metrics["pr_precision"],
            metrics["pr_recall"],
            metrics.get("pr_auc", 0),
            save_path=str(save_dir / "precision_recall_curve.png"),
        )
        saved_plots["pr_curve"] = str(save_dir / "precision_recall_curve.png")
        plt.close(fig)

    # Training history
    if history:
        fig = plot_training_history(
            history, save_path=str(save_dir / "training_history.png")
        )
        saved_plots["training_history"] = str(save_dir / "training_history.png")
        plt.close(fig)

    # Class distribution
    if "true_distribution" in metrics and "predicted_distribution" in metrics:
        fig = plot_class_distribution(
            metrics["true_distribution"],
            metrics["predicted_distribution"],
            save_path=str(save_dir / "class_distribution.png"),
        )
        saved_plots["class_distribution"] = str(save_dir / "class_distribution.png")
        plt.close(fig)

    # Violence analysis
    if any(key.startswith("violence_") for key in metrics.keys()):
        fig = plot_violence_analysis(
            metrics, save_path=str(save_dir / "violence_analysis.png")
        )
        saved_plots["violence_analysis"] = str(save_dir / "violence_analysis.png")
        plt.close(fig)

    return saved_plots
