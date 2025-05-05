import os
import json
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attacks.patch_fgsm import generate_patch_fgsm_examples
from src.dataset import load_dataset, create_data_loader, get_class_names

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Set constants
EPSILON = 0.8  # Higher epsilon for patch attack
TARGET_DIR = "data/adversarial_test_set_3"
FIGURES_DIR = "figures/task4"
LOGS_DIR = "logs"
NUM_EXAMPLES = 500  # Number of images to attack
NUM_VISUALIZATION = 5  # Number of examples to visualize
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGETED = False  # Use untargeted attack for better performance with PGD
ITERATIONS = 20   # Number of PGD iterations
STEP_SIZE = 0.1   # Step size for PGD
DATASET_PATH = "./data/TestDataSet"

# Performance tracking
correct_original_top1 = 0
correct_original_top5 = 0
correct_adversarial_top1 = 0
correct_adversarial_top5 = 0
total_images = 0

# Lists to store examples where attack was successful
successful_original_images = []
successful_adversarial_images = []
successful_true_labels = []
successful_original_preds = []
successful_adversarial_preds = []
successful_patch_locations = []


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Convert normalized image tensor to PIL Image"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def visualize_predictions(original_images, adv_images, original_preds, adv_preds, true_labels, 
                          class_names, patch_locations, filename="patch_attack_examples.png"):
    """Visualize original and adversarial examples with their predictions"""
    plt.figure(figsize=(20, 4 * NUM_VISUALIZATION))
    
    # Create a 3x5 grid for each example:
    # Row 1: Original, Adversarial, Difference (enhanced)
    # Row 2: Patch mask, Perturbation (enhanced)
    
    for i in range(min(len(original_images), NUM_VISUALIZATION)):
        # Get data for this example
        orig_img = denormalize(original_images[i].cpu())
        adv_img = denormalize(adv_images[i].cpu())
        diff = (adv_img - orig_img).abs()
        diff_enhanced = diff * 10  # Enhance the difference for visibility
        
        # Get class predictions
        try:
            true_label_idx = true_labels[i].item()
            orig_pred_idx = original_preds[i].item()
            adv_pred_idx = adv_preds[i].item()
            
            true_class = class_names[true_label_idx] if true_label_idx < len(class_names) else f"Class_{true_label_idx}"
            orig_pred_class = class_names[orig_pred_idx] if orig_pred_idx < len(class_names) else f"Class_{orig_pred_idx}"
            adv_pred_class = class_names[adv_pred_idx] if adv_pred_idx < len(class_names) else f"Class_{adv_pred_idx}"
        except Exception as e:
            print(f"Warning: Could not get class names for example {i}: {e}")
            true_class = f"Class_{true_labels[i].item()}"
            orig_pred_class = f"Class_{original_preds[i].item()}"
            adv_pred_class = f"Class_{adv_preds[i].item()}"
        
        # Create patch mask visualization
        mask = torch.zeros_like(orig_img)
        h_start, w_start = patch_locations[i]
        mask[:, h_start:h_start+32, w_start:w_start+32] = 1
        
        # Enhanced perturbation within patch
        patch_pert = diff * mask
        patch_pert_enhanced = patch_pert * 20  # Enhance for visibility
        
        # First row: Original image
        plt.subplot(3, NUM_VISUALIZATION, i + 1)
        plt.imshow(orig_img.permute(1, 2, 0).numpy())
        plt.title(f"Original\nTrue: {true_class}\nPred: {orig_pred_class}")
        plt.axis("off")
        
        # Second row: Adversarial image
        plt.subplot(3, NUM_VISUALIZATION, i + 1 + NUM_VISUALIZATION)
        plt.imshow(adv_img.permute(1, 2, 0).numpy())
        plt.title(f"Adversarial\nTrue: {true_class}\nPred: {adv_pred_class}")
        plt.axis("off")
        
        # Third row: Perturbation (enhanced for visibility)
        plt.subplot(3, NUM_VISUALIZATION, i + 1 + 2*NUM_VISUALIZATION)
        plt.imshow(patch_pert_enhanced.permute(1, 2, 0).numpy())
        plt.title(f"Perturbation (32x32)\nε = {EPSILON}")
        plt.axis("off")
    
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close()


def save_adversarial_images(images, labels, class_names, output_dir):
    """Save adversarial images to a directory structure matching the original dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert to PIL image for saving
        img = denormalize(image.cpu())
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img)
        
        # Get class folder name
        label_idx = label.item()
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class_{label_idx}"
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Save image
        img.save(os.path.join(class_dir, f"adv_patch_{i:04d}.png"))


def main():
    # Create directories
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    model = model.to(DEVICE)
    model.eval()
    
    # Print model information
    print(f"Model loaded with output size: {model.fc.out_features}")
    
    # Load data
    print("Loading dataset...")
    dataset = load_dataset(dataset_path=DATASET_PATH)
    dataloader = create_data_loader(dataset, batch_size=1, shuffle=False)
    
    # Load class names
    try:
        with open(os.path.join(DATASET_PATH, "labels_list.json"), 'r') as f:
            labels_list = json.load(f)
        
        # Create mapping from ImageNet class index to name
        class_idx_to_name = {}
        for label in labels_list:
            parts = label.split(": ", 1)
            if len(parts) == 2:
                idx = int(parts[0])
                name = parts[1]
                class_idx_to_name[idx] = name
            else:
                print(f"Warning: Could not parse label format: {label}")
    except Exception as e:
        print(f"Error loading labels_list.json directly: {e}")
        print("Trying to parse format...")
        # Fallback to get_class_names
        class_idx_to_name = get_class_names(DATASET_PATH)
    
    # Create a list where the index is the ImageNet class index and the value is the class name
    max_class_idx = max(class_idx_to_name.keys()) if class_idx_to_name else 1000
    class_names = []
    for i in range(max_class_idx + 1):
        class_names.append(class_idx_to_name.get(i, f"Unknown_{i}"))
    
    # Create a mapping from folder indices to ImageNet class indices
    folder_to_imagenet_map = {}
    
    # Debug info
    print("\nDEBUG INFO:")
    print(f"Dataset classes: {dataset.classes[:5]}... (total: {len(dataset.classes)})")
    print(f"Number of samples: {len(dataset)}")
    print(f"First few class_idx_to_name entries: {list(class_idx_to_name.items())[:5]}")
    
    # Log some class mapping examples
    for i, cls in enumerate(dataset.classes[:5]):
        print(f"Class {i}: {cls}")
    
    # Collect data for attack
    print("Preparing data for attack...")
    all_images = []
    all_labels = []
    all_folder_indices = []
    
    for images, labels in tqdm(dataloader, total=min(len(dataloader), NUM_EXAMPLES)):
        # Store original folder index
        folder_idx = labels[0].item()
        all_folder_indices.append(folder_idx)
        
        # Update folder to ImageNet map if needed
        folder_name = dataset.classes[folder_idx]
        
        if folder_name not in folder_to_imagenet_map:
            # Find corresponding ImageNet class
            found_match = False
            for idx, name in class_idx_to_name.items():
                if folder_name.lower() in name.lower():
                    folder_to_imagenet_map[folder_name] = idx
                    found_match = True
                    break
            
            if not found_match:
                # If we can't find a match, use folder indices with offset
                imagenet_idx = 401 + folder_idx  # ImageNet subset starts at class 401
                folder_to_imagenet_map[folder_name] = imagenet_idx
        
        # Get ImageNet class index for this label
        imagenet_idx = folder_to_imagenet_map[folder_name]
        imagenet_label = torch.tensor([imagenet_idx])
        
        all_images.append(images[0])
        all_labels.append(imagenet_label[0])
        
        if len(all_images) >= NUM_EXAMPLES:
            break
    
    images_tensor = torch.stack(all_images).to(DEVICE)
    labels_tensor = torch.stack(all_labels).to(DEVICE)
    
    # Global performance tracking
    global correct_original_top1, correct_original_top5, correct_adversarial_top1, correct_adversarial_top5, total_images
    
    # Evaluate original images
    print("Evaluating original images...")
    model.eval()
    with torch.no_grad():
        outputs = model(images_tensor)
        _, predicted = outputs.max(1)
        
        # Calculate Top-1 accuracy
        for i in range(labels_tensor.size(0)):
            total_images += 1
            if predicted[i].item() == labels_tensor[i].item():
                correct_original_top1 += 1
        
        # Calculate Top-5 accuracy
        _, top5_pred = outputs.topk(5, 1)
        for i in range(labels_tensor.size(0)):
            if labels_tensor[i] in top5_pred[i]:
                correct_original_top5 += 1
    
    original_top1_acc = 100.0 * correct_original_top1 / total_images
    original_top5_acc = 100.0 * correct_original_top5 / total_images
    
    print(f"Original Top-1 Accuracy: {original_top1_acc:.2f}%")
    print(f"Original Top-5 Accuracy: {original_top5_acc:.2f}%")
    
    # Generate patch locations for each image
    batch_size, channels, height, width = images_tensor.shape
    patch_locations = []
    for i in range(batch_size):
        max_h = height - 32
        max_w = width - 32
        h_start = random.randint(0, max_h)
        w_start = random.randint(0, max_w)
        patch_locations.append((h_start, w_start))
    
    # Generate adversarial examples
    print(f"Generating patch-based adversarial examples with ε={EPSILON}...")
    
    # For targeted attack, get least likely classes
    target_classes = None
    if TARGETED:
        with torch.no_grad():
            outputs = model(images_tensor)
            _, target_classes = outputs.sort(dim=1)
            target_classes = target_classes[:, 0]  # Least likely class
    
    # Generate adversarial examples with patch constraint
    adv_images, l_inf_distances = generate_patch_fgsm_examples(
        model, images_tensor, labels_tensor, 
        epsilon=EPSILON, targeted=TARGETED, 
        target_class=target_classes if TARGETED else None, 
        device=DEVICE,
        iterations=ITERATIONS,
        step_size=STEP_SIZE
    )
    
    # Verify the L-infinity distance is as expected
    print(f"Verifying L-infinity distances...")
    print(f"Maximum allowed epsilon: {EPSILON}")
    print(f"Calculated maximum L-infinity distance: {l_inf_distances.max().item():.6f}")
    
    if l_inf_distances.max().item() > EPSILON + 1e-5:
        print(f"WARNING: L-infinity distance exceeds epsilon by: {l_inf_distances.max().item() - EPSILON:.6f}")
    else:
        print(f"✓ L-infinity distances are within the epsilon bound")
    
    # Evaluate adversarial examples
    print("Evaluating adversarial examples...")
    model.eval()
    with torch.no_grad():
        adv_outputs = model(adv_images)
        _, adv_predicted = adv_outputs.max(1)
        
        # Calculate Top-1 accuracy for adversarial examples
        for i in range(labels_tensor.size(0)):
            if adv_predicted[i].item() == labels_tensor[i].item():
                correct_adversarial_top1 += 1
        
        # Calculate Top-5 accuracy for adversarial examples
        _, adv_top5_pred = adv_outputs.topk(5, 1)
        for i in range(labels_tensor.size(0)):
            if labels_tensor[i] in adv_top5_pred[i]:
                correct_adversarial_top5 += 1
        
        # Store successful attacks for visualization
        for i in range(batch_size):
            # Original and adversarial predictions
            orig_pred = predicted[i].item()
            adv_pred = adv_predicted[i].item()
            true_label = labels_tensor[i].item()
            
            # Check if attack was successful (original was correct but adversarial is wrong)
            if orig_pred == true_label and adv_pred != true_label:
                successful_original_images.append(images_tensor[i].detach().clone())
                successful_adversarial_images.append(adv_images[i].detach().clone())
                successful_true_labels.append(true_label)
                successful_original_preds.append(orig_pred)
                successful_adversarial_preds.append(adv_pred)
                successful_patch_locations.append(patch_locations[i])
                
                # Stop collecting after getting enough examples
                if len(successful_original_images) >= NUM_VISUALIZATION:
                    break
    
    adversarial_top1_acc = 100.0 * correct_adversarial_top1 / total_images
    adversarial_top5_acc = 100.0 * correct_adversarial_top5 / total_images
    
    print(f"Adversarial Top-1 Accuracy: {adversarial_top1_acc:.2f}%")
    print(f"Adversarial Top-5 Accuracy: {adversarial_top5_acc:.2f}%")
    
    # Calculate attack success rate
    attack_success = (predicted != adv_predicted).sum().item()
    attack_success_rate = 100.0 * attack_success / labels_tensor.size(0)
    print(f"Attack Success Rate: {attack_success_rate:.2f}%")
    
    # Visualize examples
    print("Generating visualizations...")
    
    # 1. Visualize successful examples
    if successful_original_images:
        num_examples = min(NUM_VISUALIZATION, len(successful_original_images))
        fig, axes = plt.subplots(num_examples, 3, figsize=(15, 4*num_examples))
        
        for i in range(num_examples):
            # Original image
            orig_img = denormalize(successful_original_images[i].cpu())
            axes[i, 0].imshow(orig_img.permute(1, 2, 0).numpy())
            
            true_label = successful_true_labels[i]
            orig_pred = successful_original_preds[i]
            true_class = class_names[true_label] if true_label < len(class_names) else f"Class_{true_label}"
            orig_pred_class = class_names[orig_pred] if orig_pred < len(class_names) else f"Class_{orig_pred}"
            
            axes[i, 0].set_title(f"Original\nTrue: {true_class}\nPred: {orig_pred_class}", fontsize=10)
            axes[i, 0].axis('off')
            
            # Adversarial image
            adv_img = denormalize(successful_adversarial_images[i].cpu())
            axes[i, 1].imshow(adv_img.permute(1, 2, 0).numpy())
            
            adv_pred = successful_adversarial_preds[i]
            adv_pred_class = class_names[adv_pred] if adv_pred < len(class_names) else f"Class_{adv_pred}"
            
            axes[i, 1].set_title(f"Adversarial\nTrue: {true_class}\nPred: {adv_pred_class}", fontsize=10)
            axes[i, 1].axis('off')
            
            # Perturbation (enhanced for visibility)
            diff = (adv_img - orig_img).abs()
            
            # Create patch mask
            mask = torch.zeros_like(orig_img)
            h_start, w_start = successful_patch_locations[i]
            mask[:, h_start:h_start+32, w_start:w_start+32] = 1
            
            # Enhanced perturbation within patch
            patch_pert = diff * mask
            patch_pert_enhanced = patch_pert * 10  # Enhance for visibility
            
            axes[i, 2].imshow(patch_pert_enhanced.permute(1, 2, 0).numpy())
            axes[i, 2].set_title(f"Perturbation (32x32)\nε = {EPSILON}", fontsize=10)
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, "successful_attacks.png"))
        plt.close()
        print(f"Successful attack examples saved to {FIGURES_DIR}/successful_attacks.png")
    else:
        print("No successful attacks found for visualization")
    
    # 2. Plot the distribution of L-infinity distances
    plt.figure(figsize=(10, 6))
    plt.hist(l_inf_distances.cpu().numpy(), bins=20)
    plt.xlabel("L-infinity Distance")
    plt.ylabel("Count")
    plt.title(f"Distribution of L-infinity Distances (Max: {l_inf_distances.max().item():.6f})")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "l_infinity_distances.png"))
    plt.close()
    print(f"L-infinity distance histogram saved to {FIGURES_DIR}/l_infinity_distances.png")
    
    # Save adversarial images
    print(f"Saving adversarial images to {TARGET_DIR}...")
    for i, (image, label, folder_idx) in enumerate(zip(adv_images, labels_tensor, all_folder_indices)):
        # Get folder name from the original dataset
        folder_name = dataset.classes[folder_idx]
        
        # Create target directory
        target_dir = os.path.join(TARGET_DIR, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Convert to PIL image and save
        img = denormalize(image.cpu())
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(target_dir, f"adv_patch_{i:04d}.png"))
    
    # Save results
    results = {
        "task": "Task 4: Patch Attack",
        "epsilon": EPSILON,
        "targeted": TARGETED,
        "patch_size": 32,
        "original_top1_accuracy": float(original_top1_acc),
        "original_top5_accuracy": float(original_top5_acc),
        "adversarial_top1_accuracy": float(adversarial_top1_acc),
        "adversarial_top5_accuracy": float(adversarial_top5_acc),
        "top1_accuracy_drop": float(original_top1_acc - adversarial_top1_acc),
        "top5_accuracy_drop": float(original_top5_acc - adversarial_top5_acc),
        "attack_success_rate": float(attack_success_rate),
        "avg_l_inf_distance": float(l_inf_distances.mean().item()),
        "max_l_inf_distance": float(l_inf_distances.max().item())
    }
    
    with open(os.path.join(LOGS_DIR, "task4_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print("Results saved to logs/task4_results.json")
    print("Task 4 completed!")


if __name__ == "__main__":
    main() 