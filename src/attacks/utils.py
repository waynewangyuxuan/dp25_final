import os
import torch
import numpy as np
from src.dataset import get_project_root


def get_data_dir():
    """Return the absolute path to the data directory."""
    return os.path.join(get_project_root(), "data")


def clip_perturbation(perturbed_image, original_image, epsilon):
    """
    Clip the perturbation to ensure it's within the epsilon ball.
    
    Args:
        perturbed_image: Perturbed image tensor
        original_image: Original image tensor
        epsilon: Maximum perturbation magnitude
    
    Returns:
        Clipped perturbed image
    """
    return original_image + torch.clamp(perturbed_image - original_image, -epsilon, epsilon)


def save_adversarial_dataset(adversarial_images, dataset, output_dir, mean=None, std=None):
    """
    Save adversarial images to disk.
    
    Args:
        adversarial_images: Tensor of adversarial images (N, C, H, W)
        dataset: Original dataset
        output_dir: Directory to save adversarial images
        mean: Mean for denormalization
        std: Standard deviation for denormalization
    """
    import torchvision.transforms.functional as TF
    from PIL import Image
    
    # Make sure the output directory is absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(get_data_dir(), output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use ImageNet normalization if not provided
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    for i, (img, label) in enumerate(zip(adversarial_images, dataset.targets)):
        # Denormalize
        img = img * std + mean
        
        # Convert to PIL
        img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        pil_img = Image.fromarray(img)
        
        # Get original file path and name
        original_path = dataset.samples[i][0]
        class_dir = os.path.basename(os.path.dirname(original_path))
        filename = os.path.basename(original_path)
        
        # Create class directory
        class_output_dir = os.path.join(output_dir, class_dir)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Save image
        save_path = os.path.join(class_output_dir, filename)
        pil_img.save(save_path) 