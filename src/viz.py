import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from src.dataset import get_normalization_params, get_project_root


def denormalize(images):
    """
    Denormalize images from ImageNet normalization.
    
    Args:
        images: Normalized images tensor (B, C, H, W)
    
    Returns:
        Denormalized images tensor (B, C, H, W) in range [0, 1]
    """
    mean, std = get_normalization_params()
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    
    if images.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return images * std + mean


def get_figures_dir():
    """Return the absolute path to the figures directory."""
    return os.path.join(get_project_root(), "figures")


def visualize_examples(original_images, adversarial_images, true_labels, 
                      predicted_labels_orig, predicted_labels_adv, 
                      class_idx_to_name, num_examples=5, save_path=None):
    """
    Visualize original and adversarial examples side by side.
    
    Args:
        original_images: Tensor of original images (B, C, H, W)
        adversarial_images: Tensor of adversarial images (B, C, H, W)
        true_labels: Tensor of true labels (B)
        predicted_labels_orig: Tensor of predicted labels for original images (B)
        predicted_labels_adv: Tensor of predicted labels for adversarial images (B)
        class_idx_to_name: Dictionary mapping indices to class names
        num_examples: Number of examples to visualize
        save_path: Path to save the figure
    """
    # Convert to numpy and denormalize
    with torch.no_grad():
        original_images = denormalize(original_images)
        adversarial_images = denormalize(adversarial_images)
    
    # Convert to numpy
    original_images = original_images.cpu().numpy().transpose(0, 2, 3, 1)
    adversarial_images = adversarial_images.cpu().numpy().transpose(0, 2, 3, 1)
    true_labels = true_labels.cpu().numpy()
    predicted_labels_orig = predicted_labels_orig.cpu().numpy()
    predicted_labels_adv = predicted_labels_adv.cpu().numpy()
    
    # Clip to [0, 1] to avoid visualization artifacts
    original_images = np.clip(original_images, 0, 1)
    adversarial_images = np.clip(adversarial_images, 0, 1)
    
    # Get the difference (magnified for visibility)
    differences = np.abs(adversarial_images - original_images)
    # Scale up differences for better visibility
    differences = differences * 5.0
    differences = np.clip(differences, 0, 1)
    
    # Visualize
    num_examples = min(num_examples, len(original_images))
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 3 * num_examples))
    
    if num_examples == 1:
        axes = [axes]
    
    for i in range(num_examples):
        # Original image
        axes[i][0].imshow(original_images[i])
        true_label_name = class_idx_to_name[true_labels[i]] if true_labels[i] in class_idx_to_name else f"Unknown ({true_labels[i]})"
        pred_orig_name = class_idx_to_name[predicted_labels_orig[i]] if predicted_labels_orig[i] in class_idx_to_name else f"Unknown ({predicted_labels_orig[i]})"
        axes[i][0].set_title(f"Original\nTrue: {true_label_name}\nPred: {pred_orig_name}")
        axes[i][0].axis('off')
        
        # Adversarial image
        axes[i][1].imshow(adversarial_images[i])
        pred_adv_name = class_idx_to_name[predicted_labels_adv[i]] if predicted_labels_adv[i] in class_idx_to_name else f"Unknown ({predicted_labels_adv[i]})"
        axes[i][1].set_title(f"Adversarial\nTrue: {true_label_name}\nPred: {pred_adv_name}")
        axes[i][1].axis('off')
        
        # Difference
        axes[i][2].imshow(differences[i])
        axes[i][2].set_title(f"Difference (x5)")
        axes[i][2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Make sure the path is absolute
        if not os.path.isabs(save_path):
            save_path = os.path.join(get_figures_dir(), os.path.basename(save_path))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def visualize_perturbation_histogram(perturbations, epsilon, save_path=None):
    """
    Visualize a histogram of perturbation magnitudes.
    
    Args:
        perturbations: Tensor of perturbations (B, C, H, W)
        epsilon: Epsilon value used for attack
        save_path: Path to save the figure
    """
    # Compute L-inf norm for each image in the batch
    with torch.no_grad():
        l_inf_norms = torch.max(torch.abs(perturbations.view(perturbations.size(0), -1)), dim=1)[0]
        l_inf_norms = l_inf_norms.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(l_inf_norms, bins=20, alpha=0.7)
    plt.axvline(x=epsilon, color='r', linestyle='--', label=f'ε = {epsilon}')
    plt.xlabel('L-inf Norm of Perturbation')
    plt.ylabel('Count')
    plt.title('Perturbation Magnitude Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        # Make sure the path is absolute
        if not os.path.isabs(save_path):
            save_path = os.path.join(get_figures_dir(), os.path.basename(save_path))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show() 