import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple, List

class AutoAttack:
    """
    AutoAttack: a stronger adversarial attack combining multiple attack strategies.
    Main paper: https://arxiv.org/abs/2003.01690
    
    This implementation focuses on APGD (Auto Projected Gradient Descent),
    the main component of AutoAttack, with both targeted and untargeted modes.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        norm: str = 'Linf', 
        eps: float = 0.02, 
        version: str = 'standard',
        seed: int = 0,
        verbose: bool = True
    ):
        """
        Initialize AutoAttack
        
        Args:
            model: PyTorch model
            norm: Perturbation norm ('Linf' or 'L2')
            eps: Perturbation budget
            version: Attack version ('standard' or 'plus')
            seed: Random seed
            verbose: Whether to print progress
        """
        self.model = model
        self.norm = norm
        self.eps = eps
        self.seed = seed
        self.verbose = verbose
        self.version = version
        
        # Set random seed
        torch.manual_seed(self.seed)
        
        # Number of restarts for APGD
        if version == 'standard':
            self.n_restarts = 1
        else:
            self.n_restarts = 5
            
        # Default attack parameters
        self.n_iter = 100  # Number of iterations
        self.eot_iter = 1  # EOT iterations
        
        # Keep track of successful attacks
        self.successful = 0
        self.total = 0
        
        # Model should be in eval mode
        self.model.eval()
    
    def check_oscillation(self, x_adv: torch.Tensor, j: int, k: int, y5: torch.Tensor, 
                          k3: int = 0, threshold: float = 0.75) -> bool:
        """
        Check if the objective function is oscillating
        
        Args:
            x_adv: Current adversarial examples
            j: Current iteration
            k: Number of iterations to check
            y5: History of loss values
            k3: Number of iterations to compare
            threshold: Oscillation threshold
            
        Returns:
            Whether oscillation is detected
        """
        if k == k3:
            return False
        
        # Calculate mean of last k values
        x_adv = x_adv.clone().detach().cpu().numpy()
        means = torch.mean(y5[-k:])
        
        # Check if all values are above the mean
        return (y5[-k:] > threshold * means).all()
    
    def get_alpha_schedule(self, n_iter: int) -> torch.Tensor:
        """
        Get alpha schedule for APGD
        
        Args:
            n_iter: Number of iterations
            
        Returns:
            Alpha schedule
        """
        alpha_schedule = torch.linspace(0, 1, n_iter + 1)[:-1]
        return alpha_schedule
    
    def apgd_attack(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        targeted: bool = False,
        n_iter: int = 100, 
        n_restarts: int = 1
    ) -> torch.Tensor:
        """
        Perform APGD attack
        
        Args:
            x: Original images
            y: Target labels (true labels for untargeted, target labels for targeted)
            targeted: Whether to perform targeted attack
            n_iter: Number of iterations
            n_restarts: Number of random restarts
            
        Returns:
            Best adversarial examples
        """
        # Set attack mode
        if targeted:
            self.targeted = True
            self.loss_target = nn.CrossEntropyLoss(reduction='none')
        else:
            self.targeted = False
            self.loss_target = nn.CrossEntropyLoss(reduction='none')
        
        # Initialize variables
        device = x.device
        batch_size = x.shape[0]
        
        # Store original images (required for strict projections)
        x_natural = x.clone().detach()
        
        # Initialize best perturbation and loss
        best_loss = torch.ones(batch_size).to(device) * float('-inf' if targeted else 'inf')
        best_x_adv = x.clone()
        
        # APGD parameters
        alpha_schedule = self.get_alpha_schedule(n_iter)
        step_size = 2.0 * self.eps / n_iter
        
        # Keep track of images to iterate on
        is_better = torch.ones(batch_size).bool().to(device)
        
        # Start with all images
        x_adv = x.clone()
        
        for restart in range(n_restarts):
            if restart > 0:
                # Random initialization
                random_noise = torch.FloatTensor(x.shape).uniform_(-self.eps, self.eps).to(device)
                x_adv = x_natural.clone() + random_noise
                # Ensure valid pixel range
                x_adv = torch.clamp(x_adv, 0, 1)
                # Double-check projection to epsilon ball
                delta = x_adv - x_natural
                delta = torch.clamp(delta, -self.eps, self.eps)
                x_adv = x_natural + delta
            
            # Initialize momentum and variables
            momentum = torch.zeros_like(x)
            best_loss_curr = torch.ones(batch_size).to(device) * float('-inf' if targeted else 'inf')
            best_curr = x_adv.clone()
            
            # Initialize loss history
            loss_history = torch.zeros(batch_size, n_iter).to(device)
            
            # Run attack iterations
            for i in range(n_iter):
                # Step size schedule
                alpha = alpha_schedule[i]
                
                # Forward
                x_adv.requires_grad_(True)
                with torch.enable_grad():
                    outputs = self.model(x_adv)
                    loss = self.loss_target(outputs, y)
                    if targeted:
                        loss = -loss  # Minimize loss for targeted attacks
                
                # Store loss history
                loss_history[:, i] = loss.detach()
                
                # Get current best
                is_better_curr = loss < best_loss_curr if not targeted else loss > best_loss_curr
                best_loss_curr[is_better_curr] = loss.detach()[is_better_curr]
                best_curr[is_better_curr] = x_adv.detach()[is_better_curr]
                
                # Check for oscillation and convergence
                if i == 0:
                    loss_best = loss.detach().clone()
                    o_best_x_adv = x_adv.clone()
                    
                # Update global best
                is_better_global = loss < best_loss if not targeted else loss > best_loss
                best_loss[is_better_global] = loss.detach()[is_better_global]
                best_x_adv[is_better_global] = x_adv.detach()[is_better_global]
                
                # Early stopping
                if (i + 1) % 10 == 0:
                    # Check for convergence
                    if loss.detach().max() <= loss_best.max() and not targeted:
                        break
                    loss_best = loss.detach().clone()
                
                # Backward
                grad = torch.autograd.grad(loss.sum(), x_adv)[0].detach()
                
                # Normalize gradient
                if self.norm == 'Linf':
                    grad_norm = torch.sign(grad)
                else:  # L2
                    grad_norm = grad / (torch.norm(grad.view(grad.shape[0], -1), dim=1)[:, None, None, None] + 1e-8)
                
                # Update with momentum
                momentum = 0.75 * momentum + grad_norm
                x_adv = x_adv.detach() + step_size * alpha * momentum
                
                # Project back to epsilon ball - FIRST projection
                delta = x_adv - x_natural
                if self.norm == 'Linf':
                    delta = torch.clamp(delta, -self.eps, self.eps)
                else:  # L2
                    delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=1)
                    factor = self.eps / delta_norm
                    factor = torch.min(factor, torch.ones_like(delta_norm))
                    delta = delta * factor[:, None, None, None]
                
                # Update adversarial images
                x_adv = x_natural + delta
                
                # Ensure valid pixel range [0, 1]
                x_adv = torch.clamp(x_adv, 0, 1)
                
                # SECOND projection back to epsilon ball (needed after clamping)
                delta = x_adv - x_natural
                if self.norm == 'Linf':
                    delta = torch.clamp(delta, -self.eps, self.eps)
                else:  # L2
                    delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=1)
                    factor = self.eps / delta_norm
                    factor = torch.min(factor, torch.ones_like(delta_norm))
                    delta = delta * factor[:, None, None, None]
                
                x_adv = x_natural + delta
            
            # Update best adversarial examples
            is_better = best_loss_curr < best_loss if not targeted else best_loss_curr > best_loss
            best_loss[is_better] = best_loss_curr[is_better]
            best_x_adv[is_better] = best_curr[is_better]
            
            if self.verbose:
                print(f"Restart {restart+1}/{n_restarts} completed")
                
        # Final strict epsilon constraint enforcement
        delta = best_x_adv - x_natural
        if self.norm == 'Linf':
            delta = torch.clamp(delta, -self.eps, self.eps)
        else:  # L2
            delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=1)
            factor = self.eps / delta_norm
            factor = torch.min(factor, torch.ones_like(delta_norm))
            delta = delta * factor[:, None, None, None]
        
        # Create final adversarial examples
        best_x_adv = torch.clamp(x_natural + delta, 0, 1)
        
        # Double-check L-infinity constraint (should never be violated now)
        delta_final = best_x_adv - x_natural
        if self.norm == 'Linf':
            delta_final = torch.clamp(delta_final, -self.eps, self.eps)
            best_x_adv = x_natural + delta_final
        
        # Verify successful attacks
        with torch.no_grad():
            outputs_orig = self.model(x)
            outputs_adv = self.model(best_x_adv)
            _, pred_orig = torch.max(outputs_orig, 1)
            _, pred_adv = torch.max(outputs_adv, 1)
            
            # Calculate success rate
            if targeted:
                # For targeted attacks, success if prediction matches target
                success = (pred_adv == y).float()
                self.successful += (pred_orig != y).sum().item() & (pred_adv == y).sum().item()
            else:
                # For untargeted attacks, success if prediction changes
                success = (pred_adv != y).float()
                self.successful += (pred_orig == y).sum().item() & (pred_adv != y).sum().item()
                
            self.total += x.shape[0]
            success_rate = success.mean().item() * 100
            
            if self.verbose:
                print(f"Attack success rate: {success_rate:.2f}%")
                
            # Verify L-infinity constraint - should always be satisfied now
            if self.norm == 'Linf':
                max_perturbation = (best_x_adv - x_natural).abs().max().item()
                if self.verbose:
                    print(f"Maximum perturbation: {max_perturbation:.6f} (ε={self.eps:.6f})")
                    
                    if max_perturbation > self.eps + 1e-5:
                        print(f"WARNING: L-infinity constraint violated! {max_perturbation:.6f} > {self.eps:.6f}")
        
        return best_x_adv
    
    def run_standard_attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Run standard AutoAttack (APGD-CE + APGD-T)
        
        Args:
            x: Original images
            y: Ground truth labels
            
        Returns:
            Adversarial examples
        """
        if self.verbose:
            print("Running AutoAttack with APGD...")
            
        # First run untargeted APGD
        if self.verbose:
            print("Step 1/2: Running untargeted APGD-CE")
        x_adv = self.apgd_attack(x, y, targeted=False, n_iter=self.n_iter, n_restarts=self.n_restarts)
        
        # Check which images are already adversarial
        with torch.no_grad():
            outputs = self.model(x_adv)
            _, pred = torch.max(outputs, 1)
            not_done = pred == y
            
        if not_done.sum() > 0 and self.version != 'rand':
            # Run targeted APGD on remaining images
            if self.verbose:
                print(f"Step 2/2: Running targeted APGD-T on {not_done.sum().item()} remaining images")
                
            # Get target classes (least likely)
            with torch.no_grad():
                outputs = self.model(x[not_done])
                # Exclude the ground truth class
                outputs[torch.arange(outputs.shape[0]), y[not_done]] = -float('inf')
                # Select least likely class
                _, targets = torch.min(outputs, 1)
                
            # Run targeted attack
            x_adv_targeted = self.apgd_attack(
                x[not_done], targets, targeted=True, 
                n_iter=self.n_iter, n_restarts=self.n_restarts
            )
            
            # Update adversarial examples
            x_adv[not_done] = x_adv_targeted
            
        return x_adv
    
    def run_plus_attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Run AutoAttack+ (more iterations and restarts)
        
        Args:
            x: Original images
            y: Ground truth labels
            
        Returns:
            Adversarial examples
        """
        if self.verbose:
            print("Running AutoAttack+ with APGD...")
            
        # Use more iterations and restarts
        self.n_iter = 200
        self.n_restarts = 5
            
        return self.run_standard_attack(x, y)
        
    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Main attack method
        
        Args:
            x: Original images
            y: Ground truth labels
            
        Returns:
            Adversarial examples
        """
        if self.version == 'standard':
            return self.run_standard_attack(x, y)
        else:  # 'plus' version
            return self.run_plus_attack(x, y)


def auto_attack(model, images, labels, eps=0.02, norm='Linf', version='standard', verbose=True):
    """
    Run AutoAttack on a batch of images
    
    Args:
        model: PyTorch model
        images: Original images
        labels: Ground truth labels
        eps: Perturbation budget
        norm: Perturbation norm ('Linf' or 'L2')
        version: AutoAttack version ('standard' or 'plus')
        verbose: Whether to print progress
        
    Returns:
        Adversarial examples
    """
    attacker = AutoAttack(model, norm=norm, eps=eps, version=version, verbose=verbose)
    return attacker.attack(images, labels) 