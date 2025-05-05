import torch
import numpy as np
from tqdm import tqdm


def accuracy(output, target, topk=(1,)):
    """
    Compute the top-k accuracy for the specified values of k.
    
    Args:
        output: Model output logits
        target: Ground truth labels
        topk: Tuple of k values to compute accuracy for
    
    Returns:
        list: Top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        # Check if targets match predictions
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def evaluate_model(model, data_loader, device='cuda', topk=(1, 5)):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        topk: Tuple of k values to compute accuracy for
    
    Returns:
        tuple: Tuple of top-k accuracies
    """
    model.eval()
    
    top_accuracies = [0] * len(topk)
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            batch_size = targets.size(0)
            total_samples += batch_size
            
            # Get accuracies for this batch
            batch_accuracies = accuracy(outputs, targets, topk=topk)
            
            # Update running totals (weighted by batch size)
            for i, acc in enumerate(batch_accuracies):
                top_accuracies[i] += acc.item() * batch_size
    
    # Compute final accuracies
    top_accuracies = [acc / total_samples for acc in top_accuracies]
    
    return tuple(top_accuracies)


def evaluate_with_imagenet_labels(model, data_loader, class_idx_to_name, device='cuda', topk=(1, 5)):
    """
    Evaluate model on a dataset and print ImageNet class predictions.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for the evaluation dataset
        class_idx_to_name: Dictionary mapping indices to class names
        device: Device to run evaluation on
        topk: Tuple of k values to compute accuracy for
    
    Returns:
        tuple: Tuple of top-k accuracies
    """
    model.eval()
    
    # Return regular evaluation if we're not printing examples
    return evaluate_model(model, data_loader, device, topk) 