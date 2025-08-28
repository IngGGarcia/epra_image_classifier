"""
Unit tests for configuration management.

Tests the configuration system including loading, saving,
validation, and type conversion.
"""

import pytest

from epra_classifier.utils.config import Config, DataConfig, ModelConfig, TrainingConfig


class TestDataConfig:
    """Test DataConfig functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig()

        assert config.image_size == (224, 224)
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.mean == [0.485, 0.456, 0.406]
        assert config.std == [0.229, 0.224, 0.225]

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DataConfig(image_size=(128, 128), batch_size=16, num_workers=2)

        assert config.image_size == (128, 128)
        assert config.batch_size == 16
        assert config.num_workers == 2

    def test_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = DataConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "image_size" in config_dict
        assert "batch_size" in config_dict
        assert config_dict["image_size"] == (224, 224)

    def test_from_dict(self):
        """Test configuration from dictionary creation."""
        config_dict = {"image_size": (128, 128), "batch_size": 16, "num_workers": 2}

        config = DataConfig.from_dict(config_dict)
        assert config.image_size == (128, 128)
        assert config.batch_size == 16
        assert config.num_workers == 2


class TestModelConfig:
    """Test ModelConfig functionality."""

    def test_default_config(self):
        """Test default model configuration."""
        config = ModelConfig()

        assert config.model_type == "resnet18"
        assert config.num_classes == 2
        assert config.hidden_size == 512
        assert config.dropout_rate == 0.3
        assert config.pretrained is True

    def test_binary_config(self):
        """Test binary classification configuration."""
        config = ModelConfig(num_classes=2)
        assert config.num_classes == 2

    def test_multiclass_config(self):
        """Test multiclass configuration."""
        config = ModelConfig(num_classes=5, model_type="efficientnet_b0")

        assert config.num_classes == 5
        assert config.model_type == "efficientnet_b0"


class TestTrainingConfig:
    """Test TrainingConfig functionality."""

    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.num_epochs == 50
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"
        assert config.scheduler == "reduce_on_plateau"

    def test_custom_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(num_epochs=10, learning_rate=0.01, optimizer="sgd")

        assert config.num_epochs == 10
        assert config.learning_rate == 0.01
        assert config.optimizer == "sgd"


class TestConfig:
    """Test main Config class functionality."""

    def test_default_config(self):
        """Test default complete configuration."""
        config = Config()

        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_config_to_dict(self):
        """Test complete configuration to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
        assert isinstance(config_dict["data"], dict)

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "data": {"batch_size": 16, "image_size": (128, 128)},
            "model": {"num_classes": 5, "model_type": "efficientnet_b0"},
            "training": {"num_epochs": 10, "learning_rate": 0.01},
        }

        config = Config.from_dict(config_dict)

        assert config.data.batch_size == 16
        assert config.data.image_size == (128, 128)
        assert config.model.num_classes == 5
        assert config.model.model_type == "efficientnet_b0"
        assert config.training.num_epochs == 10

    def test_yaml_save_load(self, temp_dir):
        """Test YAML save and load functionality."""
        config = Config()
        config.data.batch_size = 16
        config.model.num_classes = 5

        # Save to YAML
        yaml_path = temp_dir / "test_config.yaml"
        config.to_yaml(str(yaml_path))

        assert yaml_path.exists()

        # Load from YAML
        loaded_config = Config.from_yaml(str(yaml_path))

        assert loaded_config.data.batch_size == 16
        assert loaded_config.model.num_classes == 5

    def test_json_save_load(self, temp_dir):
        """Test JSON save and load functionality."""
        config = Config()
        config.data.batch_size = 16
        config.model.num_classes = 5

        # Save to JSON
        json_path = temp_dir / "test_config.json"
        config.to_json(str(json_path))

        assert json_path.exists()

        # Load from JSON
        loaded_config = Config.from_json(str(json_path))

        assert loaded_config.data.batch_size == 16
        assert loaded_config.model.num_classes == 5

    def test_update_from_args(self):
        """Test configuration update from arguments."""
        config = Config()

        args = {"batch_size": 64, "learning_rate": 0.01, "num_classes": 10}

        config.update_from_args(args)

        assert config.data.batch_size == 64
        assert config.training.learning_rate == 0.01
        assert config.model.num_classes == 10

    def test_invalid_yaml_file(self, temp_dir):
        """Test handling of invalid YAML file."""
        yaml_path = temp_dir / "invalid.yaml"
        yaml_path.write_text("invalid: yaml: content:")

        with pytest.raises(Exception):  # Should raise YAML parsing error
            Config.from_yaml(str(yaml_path))

    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("nonexistent.yaml")


@pytest.mark.parametrize("config_class", [DataConfig, ModelConfig, TrainingConfig])
def test_config_immutability_after_creation(config_class):
    """Test that configurations maintain consistency after creation."""
    config = config_class()
    original_dict = config.to_dict()

    # Convert to dict and back
    new_config = config_class.from_dict(original_dict)
    new_dict = new_config.to_dict()

    assert original_dict == new_dict


def test_config_validation():
    """Test configuration validation."""
    # Valid configuration should not raise
    config = ModelConfig(num_classes=5, dropout_rate=0.5, hidden_size=256)

    # Invalid configurations should be handled gracefully
    # (depending on implementation, might validate ranges)
    config_dict = config.to_dict()
    assert config_dict["num_classes"] == 5
    assert config_dict["dropout_rate"] == 0.5
