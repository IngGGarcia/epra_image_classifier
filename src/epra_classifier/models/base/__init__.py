"""
Base models and pipelines for EPRA Image Classifier.

This module provides base classes and common functionality shared
across different model architectures and pipelines.
"""

from .base_model import BaseModel
from .base_pipeline import BasePipeline


__all__ = ["BaseModel", "BasePipeline"]
