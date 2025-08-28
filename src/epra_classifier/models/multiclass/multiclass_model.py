"""
Multiclass violence classification model.

This module implements an EfficientNet-based model for multiclass violence
classification with advanced attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ...utils.config import ModelConfig
from ..base.base_model import BaseModel


class ContextAnalysisModule(nn.Module):
    """Context analysis module for understanding spatial relationships."""

    def __init__(self, in_channels: int):
        """
        Initialize context analysis module.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.local_context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply context analysis."""
        global_ctx = self.global_context(x)
        local_ctx = self.local_context(x)

        # Combine global and local context
        context_weights = global_ctx * local_ctx
        return x * context_weights


class MulticlassViolenceModel(BaseModel):
    """
    Multiclass violence classification model based on EfficientNet.

    This model uses EfficientNet as backbone with spatial attention and
    context analysis for multiclass violence level classification.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize multiclass violence model.

        Args:
            config: Model configuration
        """
        # Ensure multiclass configuration
        if config.num_classes < 3:
            config.num_classes = 5  # Default to 5 violence levels

        # Update feature dimension for EfficientNet
        if config.model_type.startswith("efficientnet"):
            config.feature_dim = 1280  # EfficientNet-B0 output dimension

        super().__init__(config)

        # Add context analysis if specified
        if config.use_context_analysis:
            self._add_context_analysis()

    def _build_backbone(self) -> nn.Module:
        """Build EfficientNet backbone."""
        model_type = self.config.model_type.lower()

        if model_type == "efficientnet_b0":
            backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
                if self.config.pretrained
                else None
            )
        elif model_type == "efficientnet_b1":
            backbone = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1
                if self.config.pretrained
                else None
            )
        elif model_type == "efficientnet_b2":
            backbone = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
                if self.config.pretrained
                else None
            )
        else:
            # Default to EfficientNet-B0
            backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
                if self.config.pretrained
                else None
            )

        # Remove final classification layer
        backbone = nn.Sequential(*list(backbone.children())[:-1])

        return backbone

    def _get_backbone_output_dim(self) -> int:
        """Get EfficientNet feature dimension."""
        model_type = self.config.model_type.lower()

        # EfficientNet output dimensions
        dimensions = {
            "efficientnet_b0": 1280,
            "efficientnet_b1": 1280,
            "efficientnet_b2": 1408,
            "efficientnet_b3": 1536,
            "efficientnet_b4": 1792,
        }

        return dimensions.get(model_type, 1280)  # Default to B0

    def _add_context_analysis(self) -> None:
        """Add context analysis module."""
        backbone_dim = self._get_backbone_output_dim()
        self.context_module = ContextAnalysisModule(backbone_dim)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features with optional context analysis and attention.

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not initialized")

        # Extract features using backbone
        features = self.backbone(x)

        # Apply context analysis if available
        if hasattr(self, "context_module"):
            features = self.context_module(features)

        # Apply spatial attention BEFORE pooling if available and features are 4D
        if (
            self.attention is not None
            and hasattr(self.attention, "spatial_attention")
            and len(features.shape) == 4
        ):
            features = self.attention(features)

        # Apply global average pooling to convert to 2D
        if len(features.shape) == 4:  # (batch, channels, height, width)
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        return features

    def predict_violence_level(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict violence level for multiclass classification.

        Args:
            x: Input tensor

        Returns:
            Tuple of (predicted_levels, confidence_scores)
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(x)
            logits = output["logits"]

            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)

            # Get predicted levels and confidence
            predicted_levels = torch.argmax(probabilities, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]

            return predicted_levels, confidence_scores

    def get_violence_probabilities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Get detailed violence level probabilities.

        Args:
            x: Input tensor

        Returns:
            Dictionary with violence level probabilities
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(x)
            logits = output["logits"]

            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)

            # Create level mapping
            level_probs = {}
            for i in range(self.config.num_classes):
                level_probs[f"level_{i}"] = probabilities[:, i]

            # Add aggregate measures
            level_probs["non_violence"] = probabilities[:, 0]  # Level 0
            level_probs["violence"] = torch.sum(
                probabilities[:, 1:], dim=1
            )  # Levels 1+

            return level_probs

    def extract_violence_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features specifically for violence analysis.

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        # Temporarily enable feature return
        original_return_features = getattr(self, "return_features", False)
        self.return_features = True

        try:
            output = self.forward(x)
            features = output.get("attended_features", output.get("features"))
            return features
        finally:
            self.return_features = original_return_features


def create_multiclass_model(config: ModelConfig) -> MulticlassViolenceModel:
    """
    Factory function to create a multiclass violence model.

    Args:
        config: Model configuration

    Returns:
        Initialized multiclass model
    """
    # Ensure multiclass configuration
    if config.num_classes < 3:
        config.num_classes = 5

    # Set default model type for multiclass
    if not config.model_type.startswith("efficientnet"):
        config.model_type = "efficientnet_b0"

    return MulticlassViolenceModel(config)
