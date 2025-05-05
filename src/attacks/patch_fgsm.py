import torch
import torch.nn.functional as F
import numpy as np
import random

def generate_patch_fgsm_examples(model, images, targets, epsilon=0.3, targeted=True, target_class=None, device="cpu", 
                                iterations=10, step_size=0.1):
    """
    Generate adversarial examples using Projected Gradient Descent (PGD) with perturbations 
    restricted to a 32x32 patch. This is an iterative extension of FGSM for more effective attacks.
    
    Args:
        model: The target model
        images: Batch of images (already normalized for the model)
        targets: True labels for untargeted attack or target labels for targeted attack
        epsilon: Maximum perturbation value (L-infinity norm) in [0,1] pixel space
        targeted: If True, perform a targeted attack; if False, perform an untargeted attack
        target_class: If targeted is True and target_class is provided, use these class indices as targets
        device: Device to use ("cpu" or "cuda")
        iterations: Number of attack iterations (higher = more effective but slower)
        step_size: Step size for each iteration (alpha in PGD)
        
    Returns:
        Tuple containing (adversarial examples, L-infinity distances from original images)
    """
    # Keep original images for reference
    original_images = images.clone().detach()
    
    # Start with a copy of the original images
    adv_images = images.clone().detach().to(device)
    
    # For ImageNet models, images are normalized with mean and std:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Create tensors for these values
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)
    
    # Calculate epsilon in normalized space for each channel
    # This compensates for the normalization
    epsilon_normalized = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)
    step_size_normalized = torch.tensor([step_size / s for s in std], device=device).view(1, 3, 1, 1)
    
    # Create a mask for the 32x32 patch
    batch_size, channels, height, width = adv_images.shape
    mask = torch.zeros_like(adv_images).to(device)
    
    # For each image in the batch, select a random patch location
    patch_locations = []
    for i in range(batch_size):
        # Get a strategic patch location - focus on the center of the image
        # (most ImageNet classifiers focus on central parts)
        center_h = height // 2 - 16
        center_w = width // 2 - 16
        
        # Add some randomness (±20 pixels around center)
        h_offset = random.randint(-20, 20) 
        w_offset = random.randint(-20, 20)
        
        # Ensure we stay within bounds
        h_start = max(0, min(height - 32, center_h + h_offset))
        w_start = max(0, min(width - 32, center_w + w_offset))
        
        patch_locations.append((h_start, w_start))
        
        # Set the patch area to 1 in the mask
        mask[i, :, h_start:h_start+32, w_start:w_start+32] = 1.0
    
    # PGD Attack Loop
    for iteration in range(iterations):
        # Enable gradient computation for current images
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # If targeted attack and no target_class provided, get the least likely class
        if iteration == 0 and targeted and target_class is None:
            _, target_class = outputs.sort(dim=1)
            target_class = target_class[:, 0]  # Get the least likely class
        
        # For targeted attack, use the target class; for untargeted, use the true labels
        attack_targets = target_class if targeted else targets
        
        # Calculate loss
        if targeted:
            # For targeted attack, minimize loss for target class (maximize probability)
            loss = -F.cross_entropy(outputs, attack_targets)
        else:
            # For untargeted attack, maximize loss for true class (minimize probability)
            loss = F.cross_entropy(outputs, attack_targets)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Get sign of gradients (direction of attack)
        grad_sign = adv_images.grad.sign()
        
        # Update the images with a small step in the direction of the gradient
        # For iteration-based attacks, we use smaller steps
        adv_images = adv_images.detach() + step_size_normalized * grad_sign * mask
        
        # Project back to epsilon-ball around original image (L-infinity constraint)
        # We need to do this in normalized space
        delta = adv_images - original_images
        delta = torch.clamp(delta, -epsilon_normalized, epsilon_normalized)
        adv_images = original_images + delta
        
        # Clamp back to valid image range
        # First convert to pixel space [0,1]
        adv_images_pixel = adv_images * std_tensor + mean_tensor
        adv_images_pixel = torch.clamp(adv_images_pixel, 0, 1)
        
        # Convert back to normalized space
        adv_images = (adv_images_pixel - mean_tensor) / std_tensor
        
        # Print progress for the first few iterations
        if iteration < 3 or iteration == iterations - 1:
            # Check how we're doing at this iteration
            with torch.no_grad():
                outputs = model(adv_images)
                _, preds = torch.max(outputs, 1)
                accuracy = (preds == targets).float().mean().item() * 100
                print(f"Iteration {iteration+1}/{iterations}: Current accuracy: {accuracy:.2f}%")
    
    # Calculate L-infinity distance in pixel space
    l_inf_distances = []
    
    for i in range(batch_size):
        # Get the patch location
        h_start, w_start = patch_locations[i]
        
        # Convert original and adversarial patches to pixel space
        orig_patch = original_images[i:i+1, :, h_start:h_start+32, w_start:w_start+32]
        adv_patch = adv_images[i:i+1, :, h_start:h_start+32, w_start:w_start+32]
        
        # Convert from normalized to pixel space [0,1]
        orig_patch_pixel = orig_patch * std_tensor + mean_tensor
        adv_patch_pixel = adv_patch * std_tensor + mean_tensor
        
        # Calculate L-infinity distance in pixel space
        l_inf = (adv_patch_pixel - orig_patch_pixel).abs().max().item()
        l_inf_distances.append(l_inf)
        
        # Debug output for first few images
        if i < 3:
            print(f"Image {i} (patch at {h_start},{w_start}):")
            print(f"  L-infinity distance in pixel space: {l_inf:.6f}")
            
            # Show per-channel max differences
            diff = (adv_patch_pixel - orig_patch_pixel).abs()
            print(f"  Max diff per channel - R: {diff[0,0].max().item():.6f}, " 
                  f"G: {diff[0,1].max().item():.6f}, B: {diff[0,2].max().item():.6f}")
    
    # Convert list to tensor
    l_inf_tensor = torch.tensor(l_inf_distances, device=device)
    
    # Final debug info
    print(f"Attack Summary:")
    print(f"  Epsilon: {epsilon:.6f}")
    print(f"  Max L-infinity distance: {l_inf_tensor.max().item():.6f}")
    print(f"  Average L-infinity distance: {l_inf_tensor.mean().item():.6f}")
    
    return adv_images, l_inf_tensor

def patch_fgsm_attack(model, image, label, epsilon=0.3, targeted=True, target_class=None, device="cpu", 
                     iterations=10, step_size=0.1):
    """
    Perform PGD attack on a single image with perturbations restricted to a 32x32 patch.
    
    Args:
        model: The target model
        image: Single input image (should be a tensor with shape [1, C, H, W])
        label: True label (for untargeted attack) or target label (for targeted attack)
        epsilon: Maximum perturbation value (L-infinity norm)
        targeted: If True, perform a targeted attack; if False, perform an untargeted attack
        target_class: If targeted is True and target_class is None, use least likely class
        device: Device to use ("cpu" or "cuda")
        iterations: Number of attack iterations
        step_size: Step size for each iteration
        
    Returns:
        Adversarial example
    """
    # If target_class is an integer, convert to tensor
    if isinstance(target_class, int):
        target_class = torch.tensor([target_class]).to(device)
    
    # Call the batch version with a single image
    adv_example, _ = generate_patch_fgsm_examples(
        model, image, torch.tensor([label]).to(device), 
        epsilon, targeted, target_class, device, iterations, step_size
    )
    
    return adv_example[0]  # Return the first (and only) image in the batch 