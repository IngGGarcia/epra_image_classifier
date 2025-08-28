"""
Binary violence classification model.

This module implements a ResNet-based model for binary violence classification
with attention mechanisms and modern training techniques.
"""

import torch
import torch.nn as nn
from torchvision import models

from ...utils.config import ModelConfig
from ..base.base_model import BaseModel


class BinaryViolenceModel(BaseModel):
    """
    Binary violence classification model based on ResNet18 with attention.

    This model uses ResNet18 as backbone with optional attention mechanisms
    for binary classification of violence in images.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize binary violence model.

        Args:
            config: Model configuration
        """
        # Ensure binary configuration
        if config.num_classes != 2:
            config.num_classes = 2

        super().__init__(config)

    def _build_backbone(self) -> nn.Module:
        """Build ResNet18 backbone."""
        # Load pretrained ResNet18
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
            if self.config.pretrained
            else None
        )

        # Remove final classification layer
        backbone = nn.Sequential(*list(backbone.children())[:-1])

        return backbone

    def _get_backbone_output_dim(self) -> int:
        """Get ResNet18 feature dimension."""
        return 512  # ResNet18 final layer dimension

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass with optional feature extraction.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Dictionary containing logits and optionally features
        """
        # Call parent forward
        output = super().forward(x)

        # For binary classification, we can return single logit or two logits
        # Keep two logits for consistency with cross-entropy loss
        logits = output["logits"]

        # Ensure we have the right shape for binary classification
        if logits.shape[1] == 1:
            # Single output - convert to two outputs for CrossEntropyLoss
            prob = torch.sigmoid(logits)
            logits = torch.cat([1 - prob, prob], dim=1)
            output["logits"] = logits

        return output

    def predict_violence_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get violence probability for binary classification.

        Args:
            x: Input tensor

        Returns:
            Violence probabilities (0 = non-violence, 1 = violence)
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(x)
            logits = output["logits"]

            # Apply softmax and return violence probability
            probabilities = torch.softmax(logits, dim=1)
            violence_probs = probabilities[:, 1]  # Index 1 is violence class

            return violence_probs

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


def create_binary_model(config: ModelConfig) -> BinaryViolenceModel:
    """
    Factory function to create a binary violence model.

    Args:
        config: Model configuration

    Returns:
        Initialized binary model
    """
    # Ensure binary configuration
    config.num_classes = 2

    return BinaryViolenceModel(config)
