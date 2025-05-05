import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler


def fgsm_attack(model, images, labels, epsilon=0.02, targeted=False):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: The model to attack
        images: Input images tensor (B, C, H, W)
        labels: Ground truth labels for untargeted attack, target labels for targeted attack
        epsilon: Maximum perturbation magnitude (default: 0.02)
        targeted: Whether to perform a targeted attack (default: False)
    
    Returns:
        adversarial_images: Perturbed input images
    """
    # Make sure model is in evaluation mode
    model.eval()
    
    # Clone the images and make sure they require gradients
    adversarial_images = images.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(adversarial_images)
    
    # Calculate loss
    if targeted:
        # For targeted attacks, we want to minimize loss (maximize probability of target class)
        loss = -F.cross_entropy(outputs, labels)
    else:
        # For untargeted attacks, we want to maximize loss (minimize probability of correct class)
        loss = F.cross_entropy(outputs, labels)
    
    # Backward pass to calculate gradients
    model.zero_grad()
    loss.backward()
    
    # Get sign of gradients
    grad_sign = adversarial_images.grad.sign()
    
    # Create perturbation
    if targeted:
        # For targeted attacks, move in negative gradient direction to minimize loss
        perturbation = -epsilon * grad_sign
    else:
        # For untargeted attacks, move in gradient direction to maximize loss
        perturbation = epsilon * grad_sign
    
    # Add perturbation to original images
    # IMPORTANT: Don't use adversarial_images here since it has gradients attached
    # Instead, use the original images directly
    adversarial_examples = images.clone().detach() + perturbation
    
    # Clip to ensure valid pixel range
    adversarial_examples = torch.clamp(adversarial_examples, 0, 1)
    
    # Double-check the perturbation is exactly within epsilon ball
    # This explicitly enforces: ||x_adv - x||_∞ <= epsilon
    delta = adversarial_examples - images
    delta = torch.clamp(delta, -epsilon, epsilon)
    adversarial_examples = images + delta
    
    # Verify L-infinity constraint is satisfied
    l_inf_distance = (adversarial_examples - images).abs().max().item()
    if l_inf_distance > epsilon + 1e-5:  # Allow small numerical error
        print(f"WARNING: L_inf constraint violated! {l_inf_distance} > {epsilon}")
    
    return adversarial_examples.detach()


def generate_fgsm_examples(model, dataloader, epsilon=0.02, targeted=False, device='cuda'):
    """
    Generate adversarial examples for all images in the dataloader using FGSM.
    
    Args:
        model: The model to attack
        dataloader: DataLoader containing images to attack
        epsilon: Maximum perturbation magnitude
        targeted: Whether to perform a targeted attack
        device: Device to run the attack on
    
    Returns:
        all_adv_images: Tensor of all adversarial images
        all_orig_images: Tensor of all original images
        all_orig_labels: Tensor of all original labels
        all_orig_preds: Tensor of predictions on original images
        all_adv_preds: Tensor of predictions on adversarial images
        successful_idxs: Indices of successful attacks
    """
    model.eval()
    
    all_adv_images = []
    all_orig_images = []
    all_orig_labels = []
    all_orig_preds = []
    all_adv_preds = []
    
    # Importantly, we need to disable dataloader shuffling to ensure
    # image order remains consistent across batches
    dataloader_shuffling_state = getattr(dataloader, 'shuffle', False)
    if dataloader_shuffling_state:
        print("Warning: Disabling dataloader shuffling for consistent adversarial example generation")
        dataloader.shuffle = False
    
    # Process each batch
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Get predictions on original images
        with torch.no_grad():
            outputs = model(images)
            _, orig_preds = torch.max(outputs, 1)
        
        # Generate adversarial examples - No torch.no_grad() here since we need gradients
        adv_images = fgsm_attack(model, images, labels, epsilon, targeted)
        
        # Get predictions on adversarial images
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_preds = torch.max(adv_outputs, 1)
        
        # Store for later
        all_orig_images.append(images.detach())
        all_orig_labels.append(labels.detach())
        all_orig_preds.append(orig_preds.detach())
        all_adv_images.append(adv_images.detach())
        all_adv_preds.append(adv_preds.detach())
    
    # Restore dataloader state
    if dataloader_shuffling_state:
        dataloader.shuffle = True
    
    # Concatenate all batches
    all_orig_images = torch.cat(all_orig_images)
    all_adv_images = torch.cat(all_adv_images)
    all_orig_labels = torch.cat(all_orig_labels)
    all_orig_preds = torch.cat(all_orig_preds)
    all_adv_preds = torch.cat(all_adv_preds)
    
    # Find successful attacks (predictions changed)
    successful_mask = all_orig_preds != all_adv_preds
    successful_idxs = torch.nonzero(successful_mask).squeeze()
    
    return all_adv_images, all_orig_images, all_orig_labels, all_orig_preds, all_adv_preds, successful_idxs


def compute_linf_distance(adv, orig, mean, std, denormalize=False):
    """
    Compute L-infinity distance between original and adversarial examples.

    Args:
        adv (torch.Tensor): Adversarial images (B, C, H, W).
        orig (torch.Tensor): Original images (B, C, H, W).
        mean (list of float): Mean used in normalization.
        std (list of float): Std used in normalization.
        denormalize (bool): Whether to compute distance in pixel space.

    Returns:
        float: Maximum L-infinity distance over the batch.
    """
    if denormalize:
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(orig.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(orig.device)
        adv = adv * std + mean
        orig = orig * std + mean

    linf = (adv - orig).abs().view(adv.size(0), -1).max(dim=1)[0]
    return linf.max().item()