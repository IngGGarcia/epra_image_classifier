"""
Training module for EPRA Image Classifier.

Provides unified training logic for both binary and multiclass models
with modern training techniques and best practices.
"""

from .base_trainer import BaseTrainer
from .violence_trainer import ViolenceTrainer


__all__ = ["BaseTrainer", "ViolenceTrainer"]
