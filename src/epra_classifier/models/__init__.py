"""
Models module for EPRA Image Classifier.

Contains model architectures for binary and multiclass classification,
along with base classes for common functionality.
"""

from .base import BaseModel, BasePipeline
from .binary import BinaryViolenceModel
from .model_factory import (
    create_model,
    create_model_from_checkpoint,
    get_available_models,
)
from .multiclass import MulticlassViolenceModel


__all__ = [
    "BaseModel",
    "BasePipeline",
    "BinaryViolenceModel",
    "MulticlassViolenceModel",
    "create_model",
    "create_model_from_checkpoint",
    "get_available_models",
]
