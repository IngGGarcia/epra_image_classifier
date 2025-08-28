"""
Base inference pipeline for model prediction.

This module provides a unified interface for model inference with
optimization techniques for production deployment.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from ...models.base.base_model import BaseModel
from ...models.model_factory import create_model_from_checkpoint
from ...utils.config import Config, InferenceConfig
from ...utils.helpers import get_device, validate_image
from ...utils.logging import get_logger


class InferenceError(Exception):
    """Base exception for inference-related errors."""

    pass


class BaseInferencePipeline(ABC):
    """
    Base inference pipeline for model prediction.

    This abstract base class provides common functionality for inference
    including model loading, preprocessing, and result formatting.
    """

    def __init__(self, config: Config | InferenceConfig):
        """
        Initialize inference pipeline.

        Args:
            config: Configuration object
        """
        if isinstance(config, Config):
            self.config = config
            self.inference_config = config.inference
            self.data_config = config.data
        else:
            self.inference_config = config
            self.data_config = None
            # Create minimal config for compatibility
            from ...utils.config import Config as FullConfig
            from ...utils.config import DataConfig

            self.config = FullConfig()
            self.config.inference = config
            self.data_config = DataConfig()

        self.logger = get_logger()

        # Setup device
        self.device = get_device(self.inference_config.device)
        self.logger.info(f"Inference device: {self.device}")

        # Initialize model and transforms
        self.model: BaseModel | None = None
        self.transform = None

        # Performance tracking
        self.inference_times = []

        # Load model if checkpoint provided
        if self.inference_config.checkpoint_path:
            self.load_model(self.inference_config.checkpoint_path)

    def load_model(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
        """
        try:
            self.logger.info(f"Loading model from: {checkpoint_path}")

            # Load model using factory
            self.model = create_model_from_checkpoint(checkpoint_path, self.config)
            self.model.to(self.device)
            self.model.eval()

            # Setup transforms
            self._setup_transforms()

            # Enable optimizations
            self._optimize_model()

            self.logger.info("Model loaded successfully")

        except Exception as e:
            raise InferenceError(f"Failed to load model: {e}") from e

    def _setup_transforms(self) -> None:
        """Setup image transforms for preprocessing."""
        from ...data.transforms import get_inference_transforms

        if self.data_config is not None:
            self.transform = get_inference_transforms(self.data_config)
        else:
            # Fallback to default transforms
            from torchvision import transforms

            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _optimize_model(self) -> None:
        """Apply model optimizations for inference."""
        if self.model is None:
            return

        # Set to evaluation mode
        self.model.eval()

        # Enable inference optimizations
        torch.backends.cudnn.benchmark = True

        # Compile model if available (PyTorch 2.0+)
        try:
            if hasattr(torch, "compile") and self.config.system.compile_model:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled for optimization")
        except Exception as e:
            self.logger.warning(f"Model compilation failed: {e}")

    def preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor
        """
        image_path = Path(image_path)

        # Validate image
        if not validate_image(image_path):
            raise InferenceError(f"Invalid image file: {image_path}")

        try:
            # Load image
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Apply transforms
            if self.transform is not None:
                # Check if it's Albumentations transform (requires numpy array and named arguments)
                try:
                    import numpy as np

                    image_np = np.array(image)
                    transformed = self.transform(image=image_np)
                    if isinstance(transformed, dict):
                        image_tensor = transformed["image"]
                    else:
                        image_tensor = transformed
                except (TypeError, KeyError):
                    # Fallback for torchvision transforms
                    image_tensor = self.transform(image)
            else:
                raise InferenceError("No transforms configured")

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor

        except Exception as e:
            raise InferenceError(f"Failed to preprocess image {image_path}: {e}") from e

    @abstractmethod
    def predict_single(self, image_path: str | Path) -> dict[str, Any]:
        """
        Predict on a single image.

        Args:
            image_path: Path to image file

        Returns:
            Prediction results
        """
        pass

    def predict_batch(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        """
        Predict on a batch of images.

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction results
        """
        results = []

        # Process in batches
        batch_size = self.inference_config.batch_size

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_results = self._predict_batch_internal(batch_paths)
            results.extend(batch_results)

        return results

    def _predict_batch_internal(
        self, image_paths: list[str | Path]
    ) -> list[dict[str, Any]]:
        """Internal batch prediction method."""
        if self.model is None:
            raise InferenceError("Model not loaded")

        batch_tensors = []
        valid_paths = []

        # Preprocess all images
        for path in image_paths:
            try:
                tensor = self.preprocess_image(path)
                batch_tensors.append(tensor)
                valid_paths.append(path)
            except Exception as e:
                self.logger.warning(f"Skipping invalid image {path}: {e}")

        if not batch_tensors:
            return []

        # Stack into batch
        batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)

        # Predict
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            logits = outputs["logits"]

            # Get predictions and probabilities
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1)

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time / len(valid_paths))  # Per image time

        # Format results
        results = []
        for i, path in enumerate(valid_paths):
            result = self._format_prediction_result(
                path=path,
                predicted_class=predicted_classes[i].item(),
                probabilities=probabilities[i].cpu().numpy(),
                inference_time=inference_time / len(valid_paths),
                features=outputs.get("features", [None] * len(valid_paths))[i]
                if "features" in outputs
                else None,
            )
            results.append(result)

        return results

    @abstractmethod
    def _format_prediction_result(
        self,
        path: str | Path,
        predicted_class: int,
        probabilities: torch.Tensor,
        inference_time: float,
        features: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Format prediction result.

        Args:
            path: Image path
            predicted_class: Predicted class index
            probabilities: Class probabilities
            inference_time: Inference time
            features: Extracted features (optional)

        Returns:
            Formatted result dictionary
        """
        pass

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get inference performance statistics.

        Returns:
            Performance statistics
        """
        if not self.inference_times:
            return {"message": "No inference performed yet"}

        import numpy as np

        times = np.array(self.inference_times)

        return {
            "total_inferences": len(self.inference_times),
            "avg_time_per_image": float(np.mean(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
            "throughput_fps": 1.0 / float(np.mean(times)) if np.mean(times) > 0 else 0,
            "device": str(self.device),
        }

    def warmup(self, num_iterations: int = 5) -> None:
        """
        Warm up the model for consistent performance measurements.

        Args:
            num_iterations: Number of warmup iterations
        """
        if self.model is None:
            return

        self.logger.info(f"Warming up model with {num_iterations} iterations")

        # Create dummy input
        dummy_input = torch.randn(1, 3, *self.data_config.image_size).to(self.device)

        # Warmup iterations
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)

        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.logger.info("Model warmup completed")

    def clear_cache(self) -> None:
        """Clear performance tracking cache."""
        self.inference_times.clear()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def export_model(self, export_path: str, format: str = "torchscript") -> None:
        """
        Export model for deployment.

        Args:
            export_path: Path to save exported model
            format: Export format ("torchscript", "onnx")
        """
        if self.model is None:
            raise InferenceError("Model not loaded")

        self.logger.info(f"Exporting model to {export_path} in {format} format")

        try:
            dummy_input = torch.randn(1, 3, *self.data_config.image_size).to(
                self.device
            )

            if format.lower() == "torchscript":
                # TorchScript export
                traced_model = torch.jit.trace(self.model, dummy_input)
                traced_model.save(export_path)

            elif format.lower() == "onnx":
                # ONNX export
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    export_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )
            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"Model exported successfully to {export_path}")

        except Exception as e:
            raise InferenceError(f"Failed to export model: {e}") from e
