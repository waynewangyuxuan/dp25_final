import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler


def pgd_attack(model, images, labels, epsilon=0.02, alpha=0.005, iterations=10, targeted=False, random_start=True, verbose=True):
    """
    Implements the Projected Gradient Descent (PGD) attack.
    
    Args:
        model: The model to attack
        images: Input images tensor (B, C, H, W)
        labels: Ground truth labels for untargeted attack, target labels for targeted attack
        epsilon: Maximum perturbation magnitude (default: 0.02)
        alpha: Step size for each iteration (default: 0.005)
        iterations: Number of iterations (default: 10)
        targeted: Whether to perform a targeted attack (default: False)
        random_start: Whether to start with a random perturbation (default: True)
        verbose: Whether to print progress information (default: True)
    
    Returns:
        adversarial_images: Perturbed input images
    """
    # Make sure model is in evaluation mode
    model.eval()
    
    # For ImageNet models, images are normalized with these values:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Create tensors for normalization values
    mean_tensor = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    
    # Convert epsilon from pixel space to normalized space for each channel
    # Since normalized images use different std per channel, we need channel-specific epsilon
    epsilon_normalized = torch.tensor([epsilon / s for s in std], device=images.device).view(1, 3, 1, 1)
    alpha_normalized = torch.tensor([alpha / s for s in std], device=images.device).view(1, 3, 1, 1)
    
    # Clone the images
    x_natural = images.clone().detach()
    adversarial_images = x_natural.clone()
    
    # Start with random perturbation if specified
    if random_start:
        # Generate random noise in normalized space
        # Using a smaller initial perturbation (25% of epsilon) for more realistic initialization
        noise = torch.zeros_like(adversarial_images).uniform_(-0.25, 0.25)
        # Scale noise to match epsilon constraint in normalized space
        scaled_noise = noise * epsilon_normalized
        # Apply noise
        adversarial_images = adversarial_images + scaled_noise
        
        # Project back to valid image range
        # First convert to pixel space [0,1]
        adv_images_pixel = adversarial_images * std_tensor + mean_tensor
        adv_images_pixel = torch.clamp(adv_images_pixel, 0, 1)
        # Convert back to normalized space
        adversarial_images = (adv_images_pixel - mean_tensor) / std_tensor
    
    # Track attack progress with simplified output
    if iterations > 0 and verbose:
        # Print just once before iterations start
        print(f"Starting PGD attack with epsilon={epsilon:.4f}, iterations={iterations}")
    
    # For tracking progress efficiently
    start_accuracy = None
    
    for i in range(iterations):
        # Enable gradient computation for fresh gradient calculation
        # First make sure we detach from any previous computation graph
        adversarial_images = adversarial_images.detach()
        adversarial_images.requires_grad_(True)
        
        # Forward pass
        outputs = model(adversarial_images)
        
        # Calculate loss - for the attack direction
        if targeted:
            # For targeted attacks, we want to minimize loss (maximize prob of target class)
            loss = -F.cross_entropy(outputs, labels)
        else:
            # For untargeted attacks, we want to maximize loss (minimize prob of correct class)
            loss = F.cross_entropy(outputs, labels)
        
        # Backward pass to calculate gradients
        model.zero_grad()  # Clear old gradients
        loss.backward()  # Compute new gradients
        
        # Get gradient sign
        grad_sign = adversarial_images.grad.sign()
        
        # Detach from computation graph explicitly - important to prevent gradient accumulation!
        adversarial_images = adversarial_images.detach()
        
        # Update with signed gradient
        if targeted:
            adversarial_images = adversarial_images - alpha_normalized * grad_sign
        else:
            adversarial_images = adversarial_images + alpha_normalized * grad_sign
        
        # Project back to epsilon ball around original image
        delta = adversarial_images - x_natural
        delta = torch.clamp(delta, -epsilon_normalized, epsilon_normalized)
        adversarial_images = x_natural + delta
        
        # Project back to valid image range
        # First convert to pixel space [0,1]
        adv_images_pixel = adversarial_images * std_tensor + mean_tensor
        adv_images_pixel = torch.clamp(adv_images_pixel, 0, 1)
        # Convert back to normalized space
        adversarial_images = (adv_images_pixel - mean_tensor) / std_tensor
        
        # Only print at the start, middle, and end of the process to avoid flooding terminal
        if verbose and (i == 0 or i == iterations - 1 or i == iterations // 2):
            with torch.no_grad():
                outputs = model(adversarial_images)
                _, preds = torch.max(outputs, 1)
                
                # Calculate accuracy - Important: For untargeted attacks, we need to check against true labels
                # This ensures we're measuring actual model accuracy, not how close we are to targets
                accuracy = (preds == labels).float().mean().item() * 100
                
                # Maximum perturbation (in pixel space)
                delta_pixel = (adversarial_images * std_tensor + mean_tensor) - (x_natural * std_tensor + mean_tensor)
                max_perturbation = delta_pixel.abs().max().item()
                
                # Store starting accuracy
                if i == 0:
                    start_accuracy = accuracy
                
                # Only print at key points
                if i == 0:
                    print(f"  Initial accuracy: {accuracy:.2f}%")
                elif i == iterations - 1:
                    print(f"  Final accuracy: {accuracy:.2f}% (drop of {start_accuracy - accuracy:.2f}%), Max L∞: {max_perturbation:.6f}")
                else:
                    print(f"  Progress [{i+1}/{iterations}]: accuracy: {accuracy:.2f}%")
    
    # Verify L-infinity constraint is satisfied (in pixel space)
    adv_pixel = adversarial_images * std_tensor + mean_tensor
    orig_pixel = x_natural * std_tensor + mean_tensor
    delta_pixel = adv_pixel - orig_pixel
    l_inf_distance = delta_pixel.abs().max().item()
    
    if l_inf_distance > epsilon + 1e-5 and verbose:  # Allow small numerical error
        print(f"WARNING: L_inf constraint violated! {l_inf_distance:.6f} > {epsilon}")
    
    return adversarial_images.detach()


def targeted_class_selection(outputs, true_labels, strategy='least_likely'):
    """
    Select target classes for targeted attack.
    
    Args:
        outputs: Model outputs/logits
        true_labels: Ground truth labels
        strategy: Strategy to select target class ('least_likely', 'second_most_likely', 'random')
    
    Returns:
        target_labels: Selected target labels
    """
    if strategy == 'least_likely':
        # Select class with lowest probability
        target_labels = outputs.argmin(dim=1)
    elif strategy == 'second_most_likely':
        # Sort probabilities and select second highest
        sorted_outputs = outputs.argsort(dim=1, descending=True)
        target_labels = sorted_outputs[:, 1]
    elif strategy == 'random':
        # Select random class that's different from true class
        batch_size = outputs.shape[0]
        num_classes = outputs.shape[1]
        target_labels = torch.randint(0, num_classes, (batch_size,)).to(outputs.device)
        # Make sure target is different from true label
        same_mask = target_labels == true_labels
        while same_mask.any():
            target_labels[same_mask] = torch.randint(0, num_classes, (same_mask.sum(),)).to(outputs.device)
            same_mask = target_labels == true_labels
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return target_labels


def generate_pgd_examples(model, dataloader, epsilon=0.02, alpha=0.005, iterations=10, 
                          targeted=False, target_strategy='least_likely', device='cuda'):
    """
    Generate adversarial examples for all images in the dataloader using PGD.
    
    Args:
        model: The model to attack
        dataloader: DataLoader containing images to attack
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        iterations: Number of iterations
        targeted: Whether to perform a targeted attack
        target_strategy: Strategy to select target class if targeted
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
    
    # Process each batch
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Get predictions on original images
        with torch.no_grad():
            outputs = model(images)
            _, orig_preds = torch.max(outputs, 1)
        
        attack_labels = labels
        if targeted:
            # Select target classes for targeted attack
            attack_labels = targeted_class_selection(outputs, labels, target_strategy)
        
        # Generate adversarial examples
        adv_images = pgd_attack(model, images, attack_labels, epsilon, alpha, iterations, targeted)
        
        # Get predictions on adversarial images
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_preds = torch.max(adv_outputs, 1)
        
        # Store for later
        all_orig_images.append(images.detach().cpu())
        all_orig_labels.append(labels.detach().cpu())
        all_orig_preds.append(orig_preds.detach().cpu())
        all_adv_images.append(adv_images.detach().cpu())
        all_adv_preds.append(adv_preds.detach().cpu())
    
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


def compute_linf_distance(adv, orig, mean=None, std=None, denormalize=True):
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
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
        
    if denormalize:
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(orig.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(orig.device)
        adv = adv * std + mean
        orig = orig * std + mean

    linf = (adv - orig).abs().view(adv.size(0), -1).max(dim=1)[0]
    return linf.max().item() 