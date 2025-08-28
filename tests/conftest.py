"""
Pytest configuration and fixtures for EPRA Image Classifier tests.

This module provides shared fixtures and configuration for all tests,
including mock datasets, temporary directories, and test configurations.
"""

# Add src to path for imports
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epra_classifier.utils.config import Config, DataConfig, ModelConfig, TrainingConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config() -> Config:
    """Create a sample configuration for testing."""
    config = Config()

    # Data config
    config.data = DataConfig(
        image_size=(64, 64),  # Small for testing
        batch_size=2,
        num_workers=0,  # No multiprocessing in tests
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    # Model config
    config.model = ModelConfig(
        model_type="resnet18",
        num_classes=2,
        hidden_size=64,  # Small for testing
        dropout_rate=0.1,
        use_attention=False,  # Simple for testing
        pretrained=False,  # Faster loading
    )

    # Training config
    config.training = TrainingConfig(
        num_epochs=2,  # Very short for testing
        learning_rate=0.01,
        batch_size=2,
        early_stopping_patience=1,
    )

    return config


@pytest.fixture
def binary_config(sample_config) -> Config:
    """Create binary classification configuration."""
    config = sample_config
    config.model.num_classes = 2
    return config


@pytest.fixture
def multiclass_config(sample_config) -> Config:
    """Create multiclass classification configuration."""
    config = sample_config
    config.model.num_classes = 5
    config.model.model_type = "efficientnet_b0"
    return config


@pytest.fixture
def mock_image_rgb() -> Image.Image:
    """Create a mock RGB image for testing."""
    return Image.new("RGB", (64, 64), color="red")


@pytest.fixture
def mock_image_grayscale() -> Image.Image:
    """Create a mock grayscale image for testing."""
    return Image.new("L", (64, 64), color=128)


@pytest.fixture
def sample_dataset_dir(temp_dir, mock_image_rgb):
    """Create a sample dataset directory structure."""
    # Create binary dataset structure
    binary_dir = temp_dir / "binary_dataset"

    # Create class directories
    violence_dir = binary_dir / "violence"
    non_violence_dir = binary_dir / "non_violence"

    violence_dir.mkdir(parents=True)
    non_violence_dir.mkdir(parents=True)

    # Create sample images
    for i in range(3):
        # Violence images
        img_path = violence_dir / f"violence_{i}.jpg"
        mock_image_rgb.save(img_path)

        # Non-violence images
        img_path = non_violence_dir / f"non_violence_{i}.jpg"
        mock_image_rgb.save(img_path)

    return binary_dir


@pytest.fixture
def multiclass_dataset_dir(temp_dir, mock_image_rgb):
    """Create a multiclass dataset directory structure."""
    multiclass_dir = temp_dir / "multiclass_dataset"

    # Create level directories
    for level in range(5):
        level_dir = multiclass_dir / f"level_{level}"
        level_dir.mkdir(parents=True)

        # Create sample images for each level
        for i in range(2):
            img_path = level_dir / f"level_{level}_image_{i}.jpg"
            mock_image_rgb.save(img_path)

    return multiclass_dir


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 3, 64, 64)  # Batch of 2 images


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cpu")  # Always use CPU for tests


@pytest.fixture(autouse=True)
def set_deterministic():
    """Set deterministic behavior for reproducible tests."""
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def mock_checkpoint_data():
    """Create mock checkpoint data for testing."""
    return {
        "model_state_dict": {
            "backbone.0.weight": torch.randn(64, 3, 7, 7),
            "classifier.weight": torch.randn(2, 64),
            "classifier.bias": torch.randn(2),
        },
        "optimizer_state_dict": {},
        "epoch": 5,
        "loss": 0.5,
        "config": {
            "model": {"model_type": "resnet18", "num_classes": 2, "hidden_size": 64}
        },
    }


@pytest.fixture
def sample_metrics():
    """Create sample metrics for testing."""
    return {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1": 0.85,
        "auc": 0.90,
        "confusion_matrix": [[10, 2], [1, 12]],
        "violence_precision": 0.80,
        "violence_recall": 0.85,
    }


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Skip tests if dependencies are not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    # Add unit test marker to unit tests
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
