"""
Helper utilities for EPRA Image Classifier.

This module provides common utility functions used throughout the
image classification system.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic behavior (may affect performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str | None = None) -> torch.device:
    """
    Get the appropriate device for computation.

    Args:
        device: Device specification ("auto", "cpu", "cuda", or specific device)

    Returns:
        torch.device: Device object
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cpu":
        return torch.device("cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    else:
        # Specific device like "cuda:0"
        return torch.device(device)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def create_directory(path: str | Path, exist_ok: bool = True) -> Path:
    """
    Create directory with proper error handling.

    Args:
        path: Directory path to create
        exist_ok: Whether to ignore if directory already exists

    Returns:
        Path object of created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


def save_dict_to_json(data: dict[str, Any], file_path: str | Path) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
    """
    import json

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_dict_from_json(file_path: str | Path) -> dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded dictionary
    """
    import json

    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def validate_image(image_path: str | Path) -> bool:
    """
    Validate if file is a valid image.

    Args:
        image_path: Path to image file

    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(image_path: str | Path) -> dict[str, Any]:
    """
    Get information about an image file.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "channels": len(img.getbands()) if img.mode in ["RGB", "RGBA"] else 1,
                "file_size_kb": Path(image_path).stat().st_size / 1024,
            }
    except Exception as e:
        return {"error": str(e)}


def normalize_path(path: str | Path) -> str:
    """
    Normalize path to handle different OS path separators.

    Args:
        path: Input path

    Returns:
        Normalized path string
    """
    return str(Path(path).resolve())


def format_time(seconds: float) -> str:
    """
    Format time duration in a human-readable format.

    Args:
        seconds: Time duration in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_class_weights(class_counts: dict[str, int]) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.

    Args:
        class_counts: Dictionary mapping class names to counts

    Returns:
        Tensor of class weights
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)

    weights = []
    for class_name in sorted(class_counts.keys()):
        weight = total_samples / (num_classes * class_counts[class_name])
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def calculate_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> dict[str, float]:
    """
    Calculate basic classification metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)

    if num_classes == 2:
        # Binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="binary"
        )
    else:
        # Multiclass classification
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="weighted"
        )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def split_dataset(
    dataset_size: int,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset indices into train, validation, and test sets.

    Args:
        dataset_size: Total size of dataset
        train_split: Fraction for training set
        val_split: Fraction for validation set
        test_split: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-7, (
        "Splits must sum to 1.0"
    )

    if random_seed is not None:
        np.random.seed(random_seed)

    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_size = int(dataset_size * train_split)
    val_size = int(dataset_size * val_split)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    return train_indices, val_indices, test_indices


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.

    Args:
        tensor: PyTorch tensor

    Returns:
        Numpy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()

    if tensor.is_cuda:
        tensor = tensor.cpu()

    return tensor.numpy()


def numpy_to_tensor(
    array: np.ndarray, device: torch.device | None = None
) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    Args:
        array: Numpy array
        device: Target device for tensor

    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(array)

    if device is not None:
        tensor = tensor.to(device)

    return tensor
