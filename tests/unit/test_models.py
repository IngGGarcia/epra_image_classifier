"""
Unit tests for model architectures.

Tests model creation, forward passes, and basic functionality
for both binary and multiclass models.
"""

import pytest
import torch
import torch.nn as nn

from epra_classifier.models import (
    BinaryViolenceModel,
    MulticlassViolenceModel,
    create_model,
    get_available_models,
)
from epra_classifier.models.model_factory import validate_model_config
from epra_classifier.utils.config import ModelConfig


class TestBinaryViolenceModel:
    """Test binary violence classification model."""

    def test_model_creation(self, binary_config):
        """Test binary model creation."""
        model = BinaryViolenceModel(binary_config.model)

        assert model is not None
        assert isinstance(model, BinaryViolenceModel)
        assert model.config.num_classes == 2

    def test_forward_pass(self, binary_config, sample_tensor):
        """Test forward pass through binary model."""
        model = BinaryViolenceModel(binary_config.model)
        model.eval()

        with torch.no_grad():
            output = model(sample_tensor)

        assert "logits" in output
        assert output["logits"].shape == (2, 2)  # (batch_size, num_classes)
        assert not torch.isnan(output["logits"]).any()

    def test_predict_violence_probability(self, binary_config, sample_tensor):
        """Test violence probability prediction."""
        model = BinaryViolenceModel(binary_config.model)

        probs = model.predict_violence_probability(sample_tensor)

        assert probs.shape == (2,)  # batch_size
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_feature_extraction(self, binary_config, sample_tensor):
        """Test feature extraction."""
        model = BinaryViolenceModel(binary_config.model)

        features = model.extract_violence_features(sample_tensor)

        assert features is not None
        assert features.shape[0] == 2  # batch_size
        assert len(features.shape) == 2  # (batch_size, features)

    def test_model_summary(self, binary_config):
        """Test model summary generation."""
        model = BinaryViolenceModel(binary_config.model)
        summary = model.summary()

        assert isinstance(summary, dict)
        assert "model_type" in summary
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert summary["num_classes"] == 2


class TestMulticlassViolenceModel:
    """Test multiclass violence classification model."""

    def test_model_creation(self, multiclass_config):
        """Test multiclass model creation."""
        model = MulticlassViolenceModel(multiclass_config.model)

        assert model is not None
        assert isinstance(model, MulticlassViolenceModel)
        assert model.config.num_classes == 5

    def test_forward_pass(self, multiclass_config, sample_tensor):
        """Test forward pass through multiclass model."""
        model = MulticlassViolenceModel(multiclass_config.model)
        model.eval()

        with torch.no_grad():
            output = model(sample_tensor)

        assert "logits" in output
        assert output["logits"].shape == (2, 5)  # (batch_size, num_classes)
        assert not torch.isnan(output["logits"]).any()

    def test_predict_violence_level(self, multiclass_config, sample_tensor):
        """Test violence level prediction."""
        model = MulticlassViolenceModel(multiclass_config.model)

        levels, confidence = model.predict_violence_level(sample_tensor)

        assert levels.shape == (2,)  # batch_size
        assert confidence.shape == (2,)  # batch_size
        assert torch.all(levels >= 0) and torch.all(levels < 5)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)

    def test_violence_probabilities(self, multiclass_config, sample_tensor):
        """Test detailed violence probabilities."""
        model = MulticlassViolenceModel(multiclass_config.model)

        probs_dict = model.get_violence_probabilities(sample_tensor)

        assert isinstance(probs_dict, dict)
        assert "non_violence" in probs_dict
        assert "violence" in probs_dict

        # Check that all level probabilities are present
        for i in range(5):
            assert f"level_{i}" in probs_dict

        # Check probability constraints
        for key, probs in probs_dict.items():
            if key.startswith("level_"):
                assert torch.all(probs >= 0) and torch.all(probs <= 1)


class TestModelFactory:
    """Test model factory functionality."""

    def test_create_binary_model(self, binary_config):
        """Test binary model creation through factory."""
        model = create_model(binary_config, model_type="binary")

        assert isinstance(model, BinaryViolenceModel)
        assert model.config.num_classes == 2

    def test_create_multiclass_model(self, multiclass_config):
        """Test multiclass model creation through factory."""
        model = create_model(multiclass_config, model_type="multiclass")

        assert isinstance(model, MulticlassViolenceModel)
        assert model.config.num_classes == 5

    def test_auto_detect_binary(self, binary_config):
        """Test auto-detection of binary model."""
        model = create_model(binary_config)  # No model_type specified

        assert isinstance(model, BinaryViolenceModel)

    def test_auto_detect_multiclass(self, multiclass_config):
        """Test auto-detection of multiclass model."""
        model = create_model(multiclass_config)  # No model_type specified

        assert isinstance(model, MulticlassViolenceModel)

    def test_invalid_model_type(self, binary_config):
        """Test handling of invalid model type."""
        with pytest.raises(ValueError):
            create_model(binary_config, model_type="invalid")

    def test_get_available_models(self):
        """Test getting available model information."""
        models_info = get_available_models()

        assert isinstance(models_info, dict)
        assert "binary" in models_info
        assert "multiclass" in models_info

        # Check binary model info
        binary_info = models_info["binary"]
        assert "class" in binary_info
        assert "description" in binary_info
        assert "num_classes" in binary_info

        # Check multiclass model info
        multiclass_info = models_info["multiclass"]
        assert "class" in multiclass_info
        assert "description" in multiclass_info


class TestModelValidation:
    """Test model configuration validation."""

    def test_valid_binary_config(self):
        """Test validation of valid binary config."""
        config = ModelConfig(model_type="resnet18", num_classes=2, dropout_rate=0.3)

        assert validate_model_config(config, "binary") is True

    def test_valid_multiclass_config(self):
        """Test validation of valid multiclass config."""
        config = ModelConfig(
            model_type="efficientnet_b0", num_classes=5, dropout_rate=0.3
        )

        assert validate_model_config(config, "multiclass") is True

    def test_invalid_binary_classes(self):
        """Test validation failure for wrong number of classes in binary."""
        config = ModelConfig(
            model_type="resnet18",
            num_classes=5,  # Should be 2 for binary
        )

        with pytest.raises(ValueError):
            validate_model_config(config, "binary")

    def test_invalid_multiclass_classes(self):
        """Test validation failure for wrong number of classes in multiclass."""
        config = ModelConfig(
            model_type="efficientnet_b0",
            num_classes=1,  # Should be >= 3 for multiclass
        )

        with pytest.raises(ValueError):
            validate_model_config(config, "multiclass")

    def test_invalid_dropout_rate(self):
        """Test validation failure for invalid dropout rate."""
        config = ModelConfig(
            model_type="resnet18",
            num_classes=2,
            dropout_rate=1.5,  # Should be < 1
        )

        with pytest.raises(ValueError):
            validate_model_config(config, "binary")

    def test_invalid_model_type_for_binary(self):
        """Test validation failure for invalid backbone for binary."""
        config = ModelConfig(model_type="invalid_backbone", num_classes=2)

        with pytest.raises(ValueError):
            validate_model_config(config, "binary")


class TestModelComponents:
    """Test individual model components."""

    def test_attention_mechanism(self, binary_config):
        """Test attention mechanism."""
        binary_config.model.use_attention = True
        model = BinaryViolenceModel(binary_config.model)

        # Test that attention is properly initialized
        assert model.attention is not None

        # Test forward pass with attention
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))

        assert "logits" in output

    def test_freeze_unfreeze_backbone(self, binary_config):
        """Test backbone freezing and unfreezing."""
        model = BinaryViolenceModel(binary_config.model)

        # Test freezing
        model.freeze_backbone()

        # Check that backbone parameters are frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad

        # Test unfreezing
        model.unfreeze_backbone()

        # Check that backbone parameters are unfrozen
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_trainable_parameters(self, binary_config):
        """Test getting trainable parameters."""
        model = BinaryViolenceModel(binary_config.model)

        trainable_params = model.get_trainable_parameters()

        assert isinstance(trainable_params, list)
        assert len(trainable_params) > 0
        assert all(isinstance(p, nn.Parameter) for p in trainable_params)
        assert all(p.requires_grad for p in trainable_params)


@pytest.mark.parametrize(
    "model_class,config_fixture",
    [
        (BinaryViolenceModel, "binary_config"),
        (MulticlassViolenceModel, "multiclass_config"),
    ],
)
def test_model_reproducibility(model_class, config_fixture, request):
    """Test that models produce consistent outputs."""
    config = request.getfixturevalue(config_fixture)

    # Create two identical models
    torch.manual_seed(42)
    model1 = model_class(config.model)

    torch.manual_seed(42)
    model2 = model_class(config.model)

    # Test with same input
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 3, 64, 64)

    model1.eval()
    model2.eval()

    with torch.no_grad():
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)

    # Outputs should be identical
    assert torch.allclose(output1["logits"], output2["logits"], atol=1e-6)
