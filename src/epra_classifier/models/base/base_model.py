"""
Base model class for image classification tasks.

This module provides a unified base model that implements common functionality
for both binary and multiclass classification tasks, including feature extraction,
attention mechanisms, and standard model operations.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.config import ModelConfig


class BaseAttention(nn.Module):
    """Base attention mechanism that can be used across different models."""

    def __init__(self, in_features: int, attention_dim: int = 512):
        """
        Initialize attention mechanism.

        Args:
            in_features: Number of input features
            attention_dim: Dimension of attention space
        """
        super().__init__()
        self.attention_dim = attention_dim

        self.attention = nn.Sequential(
            nn.Linear(in_features, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Calculate attention weights
        attention_weights = self.attention(x)  # (batch_size, sequence_length, 1)

        # Apply attention weights
        attended = torch.sum(x * attention_weights, dim=1)  # (batch_size, features)

        return attended, attention_weights.squeeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for 2D feature maps."""

    def __init__(self, in_channels: int):
        """
        Initialize spatial attention.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Attended tensor
        """
        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        # Apply spatial attention
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights

        return x


class BaseModel(nn.Module, ABC):
    """
    Abstract base model for image classification.

    This base class provides common functionality for all classification models,
    including feature extraction, classification heads, and utility methods.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the base model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Initialize backbone (to be implemented by subclasses)
        self.backbone: nn.Module | None = None
        self.feature_dim: int = config.feature_dim

        # Initialize attention if specified
        self.use_attention = config.use_attention
        self.attention: nn.Module | None = None

        # Initialize classifier head (to be implemented by subclasses)
        self.classifier: nn.Module | None = None

        # Initialize components
        self._build_model()

    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        """
        Build the backbone architecture.

        Returns:
            Backbone model
        """
        pass

    @abstractmethod
    def _get_backbone_output_dim(self) -> int:
        """
        Get the output dimension of the backbone.

        Returns:
            Output dimension of backbone
        """
        pass

    def _build_attention(self) -> nn.Module | None:
        """
        Build attention mechanism if specified in config.

        Returns:
            Attention module or None
        """
        if not self.use_attention:
            return None

        backbone_dim = self._get_backbone_output_dim()

        if self.config.use_spatial_attention:
            return SpatialAttention(backbone_dim)
        else:
            return BaseAttention(backbone_dim, self.config.hidden_size)

    def _build_classifier(self) -> nn.Module:
        """
        Build the classification head.

        Returns:
            Classification head
        """
        input_dim = self._get_backbone_output_dim()

        layers = []

        # Optional hidden layer
        if self.config.hidden_size != input_dim:
            layers.extend(
                [
                    nn.Linear(input_dim, self.config.hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.config.dropout_rate),
                ]
            )
            input_dim = self.config.hidden_size

        # Output layer
        layers.append(nn.Linear(input_dim, self.config.num_classes))

        return nn.Sequential(*layers)

    def _build_model(self) -> None:
        """Build all model components."""
        # Build backbone
        self.backbone = self._build_backbone()

        # Build attention
        self.attention = self._build_attention()

        # Build classifier
        self.classifier = self._build_classifier()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input using the backbone.

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not initialized")

        # Extract features using backbone
        features = self.backbone(x)

        # Apply global average pooling if features are 2D
        if len(features.shape) == 4:  # (batch, channels, height, width)
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        return features

    def apply_attention(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply attention mechanism to features.

        Args:
            features: Feature tensor

        Returns:
            Tuple of (attended_features, attention_weights)
        """
        if self.attention is None:
            return features, None

        if isinstance(self.attention, SpatialAttention):
            # Spatial attention requires 4D features (batch, channels, height, width)
            # If features are 2D, spatial attention was already applied in extract_features
            if len(features.shape) == 2:
                return features, None  # Skip spatial attention for 2D features

            attended = self.attention(features)
            return attended, None
        else:
            # For sequence attention, reshape if needed
            if len(features.shape) == 2:
                features = features.unsqueeze(1)  # Add sequence dimension

            attended, weights = self.attention(features)
            return attended, weights

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing logits and optionally features and attention weights
        """
        # Extract features
        features = self.extract_features(x)

        # Apply attention if available
        attended_features, attention_weights = self.apply_attention(features)

        # Get predictions
        if self.classifier is None:
            raise RuntimeError("Classifier not initialized")

        logits = self.classifier(attended_features)

        # Prepare output
        output = {"logits": logits}

        # Include features if requested (useful for debugging/analysis)
        if hasattr(self, "return_features") and self.return_features:
            output["features"] = features
            output["attended_features"] = attended_features

            if attention_weights is not None:
                output["attention_weights"] = attention_weights

        return output

    def predict(
        self, x: torch.Tensor, return_probabilities: bool = True
    ) -> dict[str, Any]:
        """
        Make predictions on input data.

        Args:
            x: Input tensor
            return_probabilities: Whether to return probabilities

        Returns:
            Dictionary containing predictions
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(x)
            logits = output["logits"]

            # Get predicted classes
            predicted_classes = torch.argmax(logits, dim=1)

            # Prepare result
            result = {"predicted_classes": predicted_classes, "logits": logits}

            if return_probabilities:
                if self.config.num_classes == 2:
                    # Binary classification - use sigmoid
                    probabilities = torch.sigmoid(logits)
                else:
                    # Multiclass classification - use softmax
                    probabilities = F.softmax(logits, dim=1)

                result["probabilities"] = probabilities
                result["confidence"] = torch.max(probabilities, dim=1)[0]

            # Include additional outputs if available
            for key in ["features", "attended_features", "attention_weights"]:
                if key in output:
                    result[key] = output[key]

        return result

    def get_feature_dim(self) -> int:
        """
        Get the feature dimension of the model.

        Returns:
            Feature dimension
        """
        return self._get_backbone_output_dim()

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        if self.backbone is None:
            return

        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        if self.backbone is None:
            return

        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """
        Get list of trainable parameters.

        Returns:
            List of trainable parameters
        """
        return [param for param in self.parameters() if param.requires_grad]

    def summary(self) -> dict[str, Any]:
        """
        Get model summary information.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": self.__class__.__name__,
            "num_classes": self.config.num_classes,
            "feature_dim": self.feature_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "use_attention": self.use_attention,
            "dropout_rate": self.config.dropout_rate,
            "backbone_type": self.config.model_type,
        }


class ClassificationHead(nn.Module):
    """Flexible classification head that can be used with different backbones."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = False,
    ):
        """
        Initialize classification head.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension (None for direct classification)
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        layers = []

        if hidden_dim is not None:
            # Hidden layer
            layers.append(nn.Linear(input_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.extend([nn.ReLU(inplace=True), nn.Dropout(dropout_rate)])

            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head."""
        return self.classifier(x)
