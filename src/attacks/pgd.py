import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def pgd_attack(model, images, labels, epsilon=0.02, alpha=0.005, num_iter=10, 
               targeted=False, random_start=True, device='cuda'):
    """
    Implements the Projected Gradient Descent (PGD) attack.
    
    Args:
        model: The model to attack
        images: Input images tensor (B, C, H, W)
        labels: Ground truth labels for untargeted attack, target labels for targeted attack
        epsilon: Maximum perturbation magnitude (default: 0.02)
        alpha: Step size for each iteration (default: 0.005)
        num_iter: Number of iterations (default: 10)
        targeted: Whether to perform a targeted attack (default: False)
        random_start: Whether to start with a random perturbation (default: True)
        device: Device to run the attack on (default: 'cuda')
    
    Returns:
        adversarial_images: Perturbed input images
    """
    # Make sure model is in evaluation mode
    model.eval()
    
    # Initialize adversarial examples with the original images
    adversarial_images = images.clone().detach()
    
    # Add random uniform noise if random_start is True
    if random_start:
        # Add uniform random noise within the epsilon ball
        noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(device)
        adversarial_images = adversarial_images + noise
        # Clip to ensure valid pixel range [0, 1]
        adversarial_images = torch.clamp(adversarial_images, 0, 1)
    
    for i in range(num_iter):
        # Set requires_grad attribute of tensor
        adversarial_images.requires_grad = True
        
        # Forward pass
        outputs = model(adversarial_images)
        
        # Calculate loss
        if targeted:
            # For targeted attacks, minimize loss (maximize probability of target class)
            loss = -F.cross_entropy(outputs, labels)
        else:
            # For untargeted attacks, maximize loss (minimize probability of correct class)
            loss = F.cross_entropy(outputs, labels)
        
        # Zero all existing gradients
        model.zero_grad()
        
        # Calculate gradients
        loss.backward()
        
        # Get sign of gradients
        grad_sign = adversarial_images.grad.sign()
        
        # Create perturbation
        if targeted:
            perturbation = -alpha * grad_sign
        else:
            perturbation = alpha * grad_sign
        
        # Add perturbation to adversarial examples
        with torch.no_grad():
            adversarial_images = adversarial_images.detach() + perturbation
            
            # Project back to epsilon ball
            delta = adversarial_images - images
            delta = torch.clamp(delta, -epsilon, epsilon)
            adversarial_images = images + delta
            
            # Clip to ensure valid pixel range [0, 1]
            adversarial_images = torch.clamp(adversarial_images, 0, 1)
    
    # Verify L-infinity constraint is satisfied
    l_inf_distance = (adversarial_images - images).abs().max().item()
    if l_inf_distance > epsilon + 1e-5:  # Allow small numerical error
        print(f"WARNING: L-infinity constraint violated! {l_inf_distance} > {epsilon}")
    
    return adversarial_images


def targeted_pgd_attack(model, images, target_labels, epsilon=0.02, alpha=0.005, 
                         num_iter=10, random_start=True, device='cuda'):
    """
    Wrapper for targeted PGD attack.
    
    Args:
        model: The model to attack
        images: Input images tensor (B, C, H, W)
        target_labels: Target class labels to fool the model into predicting
        epsilon: Maximum perturbation magnitude (default: 0.02)
        alpha: Step size for each iteration (default: 0.005)
        num_iter: Number of iterations (default: 10)
        random_start: Whether to start with a random perturbation (default: True)
        device: Device to run the attack on (default: 'cuda')
    
    Returns:
        adversarial_images: Perturbed input images targeting the specified labels
    """
    return pgd_attack(model, images, target_labels, epsilon, alpha, num_iter, 
                      targeted=True, random_start=random_start, device=device)


def find_least_likely_class(model, images, device='cuda'):
    """
    Find the least likely class for each image.
    
    Args:
        model: The model to use for prediction
        images: Input images tensor (B, C, H, W)
        device: Device to run the model on (default: 'cuda')
    
    Returns:
        least_likely: Tensor of least likely class indices
    """
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, least_likely = torch.min(probs, dim=1)
    return least_likely


def adaptive_pgd_attack(model, images, labels, epsilon=0.02, alpha=0.005, num_iter=20, 
                        momentum=0.9, targeted=False, adapt_steps=True, device='cuda'):
    """
    Implements an adaptive PGD attack with momentum and step size adaptation.
    
    Args:
        model: The model to attack
        images: Input images tensor (B, C, H, W)
        labels: Ground truth labels for untargeted attack, target labels for targeted attack
        epsilon: Maximum perturbation magnitude (default: 0.02)
        alpha: Initial step size (default: 0.005)
        num_iter: Number of iterations (default: 20)
        momentum: Momentum factor (default: 0.9)
        targeted: Whether to perform a targeted attack (default: False)
        adapt_steps: Whether to adapt step size during attack (default: True)
        device: Device to run the attack on (default: 'cuda')
    
    Returns:
        adversarial_images: Perturbed input images
    """
    # Make sure model is in evaluation mode
    model.eval()
    
    # Initialize adversarial examples with the original images
    adversarial_images = images.clone().detach()
    
    # Initialize momentum accumulator
    g = torch.zeros_like(images).to(device)
    
    # Add small random noise to start
    noise = torch.FloatTensor(images.shape).uniform_(-epsilon/3, epsilon/3).to(device)
    adversarial_images = torch.clamp(adversarial_images + noise, 0, 1)
    
    # Store previous loss values to track progress
    prev_losses = torch.zeros(images.size(0)).to(device)
    
    for i in range(num_iter):
        # Current iteration step size
        current_alpha = alpha
        
        if adapt_steps and i > 0:
            # Decay step size over iterations for better convergence
            current_alpha = alpha * (1 - i / num_iter)
        
        # Set requires_grad attribute of tensor
        adversarial_images.requires_grad = True
        
        # Forward pass
        outputs = model(adversarial_images)
        
        # Calculate loss for each image individually
        if targeted:
            # For targeted attacks, minimize loss (maximize probability of target class)
            losses = -F.cross_entropy(outputs, labels, reduction='none')
        else:
            # For untargeted attacks, maximize loss (minimize probability of correct class)
            losses = F.cross_entropy(outputs, labels, reduction='none')
        
        # Calculate mean loss for backward pass
        loss = losses.mean()
        
        # Zero all existing gradients
        model.zero_grad()
        
        # Calculate gradients
        loss.backward()
        
        # Compute gradients with momentum
        grad = adversarial_images.grad.detach()
        g = momentum * g + grad / torch.norm(grad, p=1)
        
        # Create perturbation 
        perturbation = current_alpha * g.sign()
        
        # Add perturbation to adversarial examples
        with torch.no_grad():
            # Update adversarial examples
            adversarial_images = adversarial_images.detach() + perturbation
            
            # Project back to epsilon ball
            delta = adversarial_images - images
            delta = torch.clamp(delta, -epsilon, epsilon)
            adversarial_images = images + delta
            
            # Clip to ensure valid pixel range [0, 1]
            adversarial_images = torch.clamp(adversarial_images, 0, 1)
            
            # Adaptive step size: If loss decreased, increase step size for next iteration
            # This is done per image in the batch
            if i > 0 and adapt_steps:
                loss_improved = losses > prev_losses
                # Increase step size for improved examples, decrease for others
                if loss_improved.any():
                    print(f"Iteration {i}: Loss improved for {loss_improved.sum().item()} examples")
            
            # Update previous losses
            prev_losses = losses.detach()
    
    # Verify L-infinity constraint is satisfied
    l_inf_distance = (adversarial_images - images).abs().max().item()
    if l_inf_distance > epsilon + 1e-5:  # Allow small numerical error
        print(f"WARNING: L-infinity constraint violated! {l_inf_distance} > {epsilon}")
    
    return adversarial_images


def ensemble_attack(models, images, labels, epsilon=0.02, alpha=0.005, num_iter=10,
                   targeted=False, random_start=True, device='cuda'):
    """
    Implements an ensemble PGD attack against multiple models.
    
    Args:
        models: List of models to attack
        images: Input images tensor (B, C, H, W)
        labels: Ground truth labels for untargeted attack, target labels for targeted attack
        epsilon: Maximum perturbation magnitude (default: 0.02)
        alpha: Step size for each iteration (default: 0.005)
        num_iter: Number of iterations (default: 10)
        targeted: Whether to perform a targeted attack (default: False)
        random_start: Whether to start with a random perturbation (default: True)
        device: Device to run the attack on (default: 'cuda')
    
    Returns:
        adversarial_images: Perturbed input images that fool all models
    """
    # Make sure all models are in evaluation mode
    for model in models:
        model.eval()
    
    # Initialize adversarial examples with the original images
    adversarial_images = images.clone().detach()
    
    # Add random uniform noise if random_start is True
    if random_start:
        # Add uniform random noise within the epsilon ball
        noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(device)
        adversarial_images = adversarial_images + noise
        # Clip to ensure valid pixel range [0, 1]
        adversarial_images = torch.clamp(adversarial_images, 0, 1)
    
    for i in range(num_iter):
        # Set requires_grad attribute of tensor
        adversarial_images.requires_grad = True
        
        # Accumulate loss across all models
        total_loss = 0
        
        for model in models:
            # Forward pass
            outputs = model(adversarial_images)
            
            # Calculate loss
            if targeted:
                # For targeted attacks, minimize loss (maximize probability of target class)
                loss = -F.cross_entropy(outputs, labels)
            else:
                # For untargeted attacks, maximize loss (minimize probability of correct class)
                loss = F.cross_entropy(outputs, labels)
            
            # Accumulate loss
            total_loss += loss / len(models)
        
        # Zero all existing gradients for all models
        for model in models:
            model.zero_grad()
        
        # Calculate gradients
        total_loss.backward()
        
        # Get sign of gradients
        grad_sign = adversarial_images.grad.sign()
        
        # Create perturbation
        if targeted:
            perturbation = -alpha * grad_sign
        else:
            perturbation = alpha * grad_sign
        
        # Add perturbation to adversarial examples
        with torch.no_grad():
            adversarial_images = adversarial_images.detach() + perturbation
            
            # Project back to epsilon ball
            delta = adversarial_images - images
            delta = torch.clamp(delta, -epsilon, epsilon)
            adversarial_images = images + delta
            
            # Clip to ensure valid pixel range [0, 1]
            adversarial_images = torch.clamp(adversarial_images, 0, 1)
    
    # Verify L-infinity constraint is satisfied
    l_inf_distance = (adversarial_images - images).abs().max().item()
    if l_inf_distance > epsilon + 1e-5:  # Allow small numerical error
        print(f"WARNING: L-infinity constraint violated! {l_inf_distance} > {epsilon}")
    
    return adversarial_images 