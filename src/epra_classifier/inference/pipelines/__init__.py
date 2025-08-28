"""
Inference pipelines for EPRA Image Classifier.

Provides optimized inference pipelines for both single image prediction
and batch processing with support for different model types.
"""

from .base_inference import BaseInferencePipeline
from .violence_inference import ViolenceInferencePipeline


__all__ = ["BaseInferencePipeline", "ViolenceInferencePipeline"]
