"""
Integration tests for training pipeline.

Tests the complete training workflow including data loading,
model training, and checkpointing.
"""

from pathlib import Path

import pytest
import torch

from epra_classifier.data.loaders import get_dataloaders
from epra_classifier.models import create_model
from epra_classifier.models.base import BasePipeline
from epra_classifier.training.trainers import ViolenceTrainer
from epra_classifier.utils.config import Config


@pytest.mark.integration
class TestTrainingPipeline:
    """Test complete training pipeline integration."""

    def test_binary_training_pipeline(self, binary_config, sample_dataset_dir):
        """Test complete binary training pipeline."""
        # Update config for test dataset
        binary_config.data.batch_size = 2
        binary_config.training.num_epochs = 2
        binary_config.training.early_stopping_patience = 1

        # Create dataloaders
        dataloaders = get_dataloaders(
            data_dir=sample_dataset_dir, config=binary_config, dataset_type="violence"
        )

        assert "train" in dataloaders
        train_loader = dataloaders["train"]
        val_loader = dataloaders.get("val")

        # Create model
        model = create_model(binary_config, model_type="binary")
        assert model is not None

        # Create pipeline
        pipeline = BasePipeline(model, binary_config)
        assert pipeline is not None

        # Create trainer
        trainer = ViolenceTrainer(pipeline, binary_config)
        assert trainer is not None

        # Run training for few epochs
        history = trainer.train(
            train_loader=train_loader, val_loader=val_loader, num_epochs=2
        )

        # Verify training history
        assert isinstance(history, dict)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2  # 2 epochs
        assert all(isinstance(loss, float) for loss in history["train_loss"])

    def test_multiclass_training_pipeline(
        self, multiclass_config, multiclass_dataset_dir
    ):
        """Test complete multiclass training pipeline."""
        # Update config for test dataset
        multiclass_config.data.batch_size = 2
        multiclass_config.training.num_epochs = 1
        multiclass_config.model.pretrained = False  # Faster for testing

        # Create dataloaders
        dataloaders = get_dataloaders(
            data_dir=multiclass_dataset_dir,
            config=multiclass_config,
            dataset_type="violence",
        )

        train_loader = dataloaders["train"]

        # Create model and pipeline
        model = create_model(multiclass_config, model_type="multiclass")
        pipeline = BasePipeline(model, multiclass_config)
        trainer = ViolenceTrainer(pipeline, multiclass_config)

        # Run training
        history = trainer.train(
            train_loader=train_loader,
            val_loader=None,  # No validation for this test
            num_epochs=1,
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1

    def test_checkpointing(self, binary_config, sample_dataset_dir, temp_dir):
        """Test model checkpointing during training."""
        # Setup
        binary_config.data.batch_size = 2
        binary_config.training.num_epochs = 2
        binary_config.training.save_every_n_epochs = 1
        binary_config.training.checkpoint_dir = str(temp_dir / "checkpoints")

        # Create pipeline
        dataloaders = get_dataloaders(sample_dataset_dir, binary_config)
        model = create_model(binary_config, model_type="binary")
        pipeline = BasePipeline(model, binary_config)
        trainer = ViolenceTrainer(pipeline, binary_config)

        # Train with checkpointing
        trainer.train(
            train_loader=dataloaders["train"],
            val_loader=dataloaders.get("val"),
            num_epochs=2,
        )

        # Check that checkpoints were saved
        checkpoint_dir = Path(binary_config.training.checkpoint_dir)
        assert checkpoint_dir.exists()

        # Look for checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0

    def test_resume_training(self, binary_config, sample_dataset_dir, temp_dir):
        """Test resuming training from checkpoint."""
        # Setup
        binary_config.data.batch_size = 2
        binary_config.training.num_epochs = 1
        binary_config.training.checkpoint_dir = str(temp_dir / "checkpoints")

        # First training session
        dataloaders = get_dataloaders(sample_dataset_dir, binary_config)
        model = create_model(binary_config, model_type="binary")
        pipeline = BasePipeline(model, binary_config)
        trainer = ViolenceTrainer(pipeline, binary_config)

        # Train and save checkpoint
        trainer.train(dataloaders["train"], num_epochs=1)
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        pipeline.save_checkpoint(
            0, {"loss": 0.5}, is_best=True, custom_path=str(checkpoint_path)
        )

        # Second training session - resume from checkpoint
        model2 = create_model(binary_config, model_type="binary")
        pipeline2 = BasePipeline(model2, binary_config)
        pipeline2.load_checkpoint(str(checkpoint_path))

        # Verify that model state was loaded
        assert pipeline2.model is not None

        # Continue training
        trainer2 = ViolenceTrainer(pipeline2, binary_config)
        history = trainer2.train(dataloaders["train"], num_epochs=1)

        assert "train_loss" in history


@pytest.mark.integration
class TestDataPipeline:
    """Test data loading and processing pipeline."""

    def test_dataset_loading(self, binary_config, sample_dataset_dir):
        """Test dataset loading with real directory structure."""
        dataloaders = get_dataloaders(
            data_dir=sample_dataset_dir, config=binary_config, dataset_type="violence"
        )

        assert "train" in dataloaders
        train_loader = dataloaders["train"]

        # Test data loading
        for batch_idx, (images, labels) in enumerate(train_loader):
            assert images.shape[0] <= binary_config.data.batch_size
            assert images.shape[1:] == (3, 64, 64)  # C, H, W from config
            assert labels.dtype == torch.long
            assert torch.all(labels >= 0) and torch.all(labels < 2)

            if batch_idx >= 2:  # Test only a few batches
                break

    def test_transforms_application(self, binary_config, sample_dataset_dir):
        """Test that transforms are properly applied."""
        dataloaders = get_dataloaders(sample_dataset_dir, binary_config)
        train_loader = dataloaders["train"]

        # Get a batch
        images, _ = next(iter(train_loader))

        # Check that images are properly transformed
        assert images.dtype == torch.float32
        assert images.shape[1:] == (3, 64, 64)  # Resized to config size

        # Check normalization (roughly)
        assert images.min() >= -3.0  # Reasonable range after normalization
        assert images.max() <= 3.0

    def test_different_dataset_structures(self, temp_dir, mock_image_rgb):
        """Test loading datasets with different structures."""
        # Create flat structure dataset
        flat_dir = temp_dir / "flat_dataset"
        flat_dir.mkdir()

        # Create images with violence/non-violence in names
        violence_img = flat_dir / "violence_image_1.jpg"
        non_violence_img = flat_dir / "normal_image_1.jpg"

        mock_image_rgb.save(violence_img)
        mock_image_rgb.save(non_violence_img)

        # Test loading
        config = Config()
        config.data.batch_size = 1

        dataloaders = get_dataloaders(
            data_dir=flat_dir, config=config, dataset_type="violence"
        )

        assert "train" in dataloaders
        assert len(dataloaders["train"].dataset) == 2


@pytest.mark.integration
@pytest.mark.slow
class TestModelTrainingValidation:
    """Test model training with validation."""

    def test_overfitting_prevention(self, binary_config, sample_dataset_dir):
        """Test that early stopping prevents overfitting."""
        # Configure for early stopping
        binary_config.training.num_epochs = 10
        binary_config.training.early_stopping_patience = 2
        binary_config.data.batch_size = 2

        # Create very small dataset to encourage overfitting
        dataloaders = get_dataloaders(sample_dataset_dir, binary_config)

        model = create_model(binary_config, model_type="binary")
        pipeline = BasePipeline(model, binary_config)
        trainer = ViolenceTrainer(pipeline, binary_config)

        history = trainer.train(
            train_loader=dataloaders["train"],
            val_loader=dataloaders.get("val"),
            num_epochs=10,
        )

        # Training should stop early due to early stopping
        actual_epochs = len(history["train_loss"])
        assert actual_epochs < 10  # Should stop before 10 epochs

    def test_metrics_tracking(self, binary_config, sample_dataset_dir):
        """Test that training metrics are properly tracked."""
        binary_config.data.batch_size = 2
        binary_config.training.num_epochs = 2

        dataloaders = get_dataloaders(sample_dataset_dir, binary_config)
        model = create_model(binary_config, model_type="binary")
        pipeline = BasePipeline(model, binary_config)
        trainer = ViolenceTrainer(pipeline, binary_config)

        history = trainer.train(
            train_loader=dataloaders["train"],
            val_loader=dataloaders.get("val"),
            num_epochs=2,
        )

        # Check that all expected metrics are tracked
        assert "train_loss" in history
        assert "epochs" in history
        assert "learning_rates" in history

        if "val_loss" in history:
            assert len(history["val_loss"]) == len(history["train_loss"])

        # Verify metric types
        assert all(isinstance(loss, (int, float)) for loss in history["train_loss"])
        assert all(isinstance(lr, (int, float)) for lr in history["learning_rates"])


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    def test_binary_end_to_end(self, binary_config, sample_dataset_dir, temp_dir):
        """Test complete binary classification workflow."""
        # Configure for quick test
        binary_config.data.batch_size = 2
        binary_config.training.num_epochs = 1
        binary_config.training.checkpoint_dir = str(temp_dir / "checkpoints")

        # 1. Data loading
        dataloaders = get_dataloaders(sample_dataset_dir, binary_config)

        # 2. Model creation
        model = create_model(binary_config, model_type="binary")

        # 3. Training setup
        pipeline = BasePipeline(model, binary_config)
        trainer = ViolenceTrainer(pipeline, binary_config)

        # 4. Training
        history = trainer.train(
            train_loader=dataloaders["train"],
            val_loader=dataloaders.get("val"),
            num_epochs=1,
        )

        # 5. Save final model
        final_model_path = temp_dir / "final_model.pth"
        pipeline.save_checkpoint(
            0, {"loss": 0.5}, is_best=True, custom_path=str(final_model_path)
        )

        # 6. Verify complete workflow
        assert history is not None
        assert final_model_path.exists()

        # 7. Test loading saved model
        new_model = create_model(binary_config, model_type="binary")
        new_pipeline = BasePipeline(new_model, binary_config)
        new_pipeline.load_checkpoint(str(final_model_path))

        # 8. Test inference with loaded model
        test_input = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = new_pipeline.model(test_input)

        assert "logits" in output
        assert output["logits"].shape == (1, 2)
