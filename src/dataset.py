import os
import json
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_project_root():
    """Return the absolute path to the project root directory."""
    # This function helps ensure all paths are relative to the project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_normalization_params():
    """Return ImageNet normalization parameters."""
    mean_norms = np.array([0.485, 0.456, 0.406])
    std_norms = np.array([0.229, 0.224, 0.225])
    return mean_norms, std_norms


def get_transforms():
    """Get standard ImageNet preprocessing transforms."""
    mean_norms, std_norms = get_normalization_params()
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_norms, std=std_norms)
    ])


def load_dataset(dataset_path=None, transform=None):
    """
    Load ImageNet test dataset.
    
    Args:
        dataset_path: Path to the test dataset
        transform: Transformation to apply to images
    
    Returns:
        dataset: Loaded dataset
    """
    if dataset_path is None:
        dataset_path = os.path.join(get_project_root(), "data", "TestDataSet")
    
    if transform is None:
        transform = get_transforms()
    
    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transform
    )
    
    return dataset


def get_class_names(dataset_path=None):
    """
    Load class names from the labels_list.json file.
    
    Args:
        dataset_path: Path to the test dataset
    
    Returns:
        class_names: Dictionary mapping indices to class names
    """
    if dataset_path is None:
        dataset_path = os.path.join(get_project_root(), "data", "TestDataSet")
    
    json_path = os.path.join(dataset_path, "labels_list.json")
    
    with open(json_path, 'r') as f:
        labels_list = json.load(f)
    
    class_idx_to_name = {}
    for label in labels_list:
        parts = label.split(": ", 1)
        idx = int(parts[0])
        name = parts[1]
        class_idx_to_name[idx] = name
    
    return class_idx_to_name


def create_data_loader(dataset, batch_size=64, shuffle=False, num_workers=4):
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
    
    Returns:
        data_loader: DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) 