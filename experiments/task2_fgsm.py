import os
import json
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attacks.fgsm import fgsm_attack

# Set path to the dataset
dataset_path = "./data/TestDataSet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
print("Loading ResNet-34 model...")
pretrained_model = models.resnet34(weights="IMAGENET1K_V1")
pretrained_model = pretrained_model.to(device)
pretrained_model.eval()

# Print model information
print(f"Model loaded with output size: {pretrained_model.fc.out_features}")

# Set up preprocessing as specified in the requirements
print("Creating dataset...")
mean_norms = np.array([0.485, 0.456, 0.406])
std_norms = np.array([0.229, 0.224, 0.225])
plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_norms, std=std_norms)
])

# Load dataset with ImageFolder
dataset = datasets.ImageFolder(root=dataset_path, transform=plain_transforms)
print(f"Dataset size: {len(dataset)}")

# Load label mapping from JSON
json_path = os.path.join(dataset_path, "labels_list.json")
with open(json_path, 'r') as f:
    labels_list = json.load(f)

# Create mapping from class index to name
class_idx_to_name = {}
for label in labels_list:
    parts = label.split(": ", 1)
    idx = int(parts[0])
    name = parts[1]
    class_idx_to_name[idx] = name

# Create dataloader - Important: shuffle=False to maintain consistent batch order
# when creating adversarial examples
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Function to calculate model confidence
def get_confidence_scores(model, images):
    """Calculate model confidence for predictions"""
    with torch.no_grad():
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predictions = torch.max(probs, dim=1)
    return confidence, predictions

def denormalize(tensor):
    """Convert normalized image tensor to numpy array for visualization"""
    tensor = tensor.clone().cpu().detach()
    for t, m, s in zip(tensor, mean_norms, std_norms):
        t.mul_(s).add_(m)
    return tensor.permute(1, 2, 0).numpy()

def visualize_adversarial_examples(original_images, adversarial_images, original_preds, adversarial_preds, 
                                   true_labels, class_names, save_path=None):
    """Visualize original and adversarial images with predictions"""
    n = len(original_images)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4*n))
    
    for i in range(n):
        # Original image
        axes[i, 0].imshow(denormalize(original_images[i]))
        true_label = class_names.get(true_labels[i], f"Unknown ({true_labels[i]})")
        pred_label = class_names.get(original_preds[i], f"Unknown ({original_preds[i]})")
        title = f"Original\nTrue: {true_label}\nPred: {pred_label}"
        axes[i, 0].set_title(title, fontsize=10)
        axes[i, 0].axis('off')
        
        # Adversarial image
        axes[i, 1].imshow(denormalize(adversarial_images[i]))
        adv_pred_label = class_names.get(adversarial_preds[i], f"Unknown ({adversarial_preds[i]})")
        title = f"Adversarial\nTrue: {true_label}\nPred: {adv_pred_label}"
        axes[i, 1].set_title(title, fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    plt.close()

# FGSM Attack parameters
epsilon = 0.02  # Attack budget

# Prepare for storing adversarial examples
os.makedirs("data/adversarial_test_set_1", exist_ok=True)
figures_dir = "./figures/task2"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Lists to store examples where attack was successful
successful_original_images = []
successful_adversarial_images = []
successful_true_labels = []
successful_original_preds = []
successful_adversarial_preds = []

# Performance tracking
correct_original_top1 = 0
correct_original_top5 = 0
correct_adversarial_top1 = 0
correct_adversarial_top5 = 0
total_images = 0
max_l_inf_distance = 0

# Create a mapping from folder indices to ImageNet class indices
folder_to_imagenet_map = {}

# Debug info
print("\nDEBUG INFO:")
print(f"Dataset classes: {dataset.classes[:5]}... (total: {len(dataset.classes)})")
print(f"Number of samples: {len(dataset)}")
print(f"First few class_idx_to_name entries: {list(class_idx_to_name.items())[:5]}")
print(f"Labels list sample: {labels_list[:2]}")

# Log some class mapping examples
for i, cls in enumerate(dataset.classes[:5]):
    print(f"Class {i}: {cls}")

# Before the main processing loop
first_batch = True
print(f"Generating adversarial examples with FGSM (epsilon={epsilon})...")

# First, evaluate model on original images to establish baseline
print("Evaluating model on original images...")
for images, labels in tqdm(dataloader, desc="Evaluating original images"):
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = pretrained_model(images)
        _, top1_preds = torch.max(outputs, 1)
        _, top5_preds = torch.topk(outputs, 5, dim=1)
    
    # Update folder to ImageNet map if needed
    for i, folder_idx in enumerate(labels):
        folder_idx = folder_idx.item()
        folder_name = dataset.classes[folder_idx]
        
        if folder_name not in folder_to_imagenet_map:
            # Find corresponding ImageNet class
            imagenet_idx = None
            for label in labels_list:
                if folder_name in label.lower():
                    imagenet_idx = int(label.split(":")[0])
                    folder_to_imagenet_map[folder_name] = imagenet_idx
                    break
            
            if imagenet_idx is None:
                # If we can't find a match, use folder indices with offset
                imagenet_idx = 401 + folder_idx
                folder_to_imagenet_map[folder_name] = imagenet_idx
    
    # Calculate accuracy
    batch_size = labels.size(0)
    total_images += batch_size
    
    for i in range(batch_size):
        folder_idx = labels[i].item()
        folder_name = dataset.classes[folder_idx]
        imagenet_idx = folder_to_imagenet_map[folder_name]
        
        # Top-1 accuracy
        if top1_preds[i].item() == imagenet_idx:
            correct_original_top1 += 1
        
        # Top-5 accuracy
        if imagenet_idx in top5_preds[i]:
            correct_original_top5 += 1

# Calculate original accuracy
original_top1_acc = 100 * correct_original_top1 / total_images
original_top5_acc = 100 * correct_original_top5 / total_images
print(f"Original Top-1 Accuracy: {original_top1_acc:.2f}%")
print(f"Original Top-5 Accuracy: {original_top5_acc:.2f}%")

# Reset counters for adversarial evaluation
total_images = 0
correct_adversarial_top1 = 0
correct_adversarial_top5 = 0

# Now generate and evaluate adversarial examples
print("Generating and evaluating adversarial examples...")
for images, labels in tqdm(dataloader, desc="Generating adversarial examples"):
    images = images.to(device)
    labels = labels.to(device)
    
    # Get original predictions for visualization and comparison
    with torch.no_grad():
        outputs = pretrained_model(images)
        orig_probs = torch.nn.functional.softmax(outputs, dim=1)
        orig_confidence, original_preds = torch.max(orig_probs, dim=1)
        _, top5_original = torch.topk(outputs, 5, dim=1)
    
    # Debug first batch
    if first_batch:
        print("\nFIRST BATCH DEBUG:")
        for i in range(min(5, len(labels))):
            folder_idx = labels[i].item()
            folder_name = dataset.classes[folder_idx]
            print(f"\nImage {i}:")
            print(f"  Folder index: {folder_idx}, Folder name: {folder_name}")
            
            # Get ImageNet class
            imagenet_idx = folder_to_imagenet_map[folder_name]
            print(f"  Mapped ImageNet index: {imagenet_idx}")
            print(f"  ImageNet class name: {class_idx_to_name.get(imagenet_idx, 'Unknown')}")
            
            # Show model prediction
            pred_idx = original_preds[i].item()
            print(f"  Model prediction: {pred_idx}")
            print(f"  Model confidence: {orig_confidence[i].item()*100:.2f}%")
            print(f"  Prediction class name: {class_idx_to_name.get(pred_idx, 'Unknown')}")
            print(f"  Is correct: {pred_idx == imagenet_idx}")
            
            # Show top 5 predictions
            print(f"  Top 5 predictions: {top5_original[i].tolist()}")
            print(f"  Is in top 5: {imagenet_idx in top5_original[i]}")
    
    # Convert folder indices to ImageNet indices for attack targets
    # Create target label tensor - use the true labels for untargeted attack
    imagenet_labels = []
    for i, folder_idx in enumerate(labels):
        folder_idx = folder_idx.item()
        folder_name = dataset.classes[folder_idx]
        imagenet_idx = folder_to_imagenet_map[folder_name]
        imagenet_labels.append(imagenet_idx)
    
    # Convert to tensor
    imagenet_labels_tensor = torch.tensor(imagenet_labels).to(device)
    
    # Generate adversarial examples - important: no torch.no_grad() here!
    # For an untargeted attack, we want to maximize the loss, not minimize it
    adversarial_images = fgsm_attack(pretrained_model, images, imagenet_labels_tensor, epsilon, targeted=False)
    
    # Calculate L-infinity distance between original and adversarial images
    l_inf_distance = (adversarial_images - images).abs().max().item()
    max_l_inf_distance = max(max_l_inf_distance, l_inf_distance)
    print(f"Batch L-infinity distance: {l_inf_distance:.6f}")
    
    # Get predictions on adversarial images
    with torch.no_grad():
        adv_outputs = pretrained_model(adversarial_images)
        adv_probs = torch.nn.functional.softmax(adv_outputs, dim=1)
        adv_confidence, adversarial_preds = torch.max(adv_probs, dim=1)
        _, top5_adversarial = torch.topk(adv_outputs, 5, dim=1)
    
    # Display first batch attack results
    if first_batch:
        print("\nFIRST BATCH ADVERSARIAL RESULTS:")
        for i in range(min(5, len(labels))):
            folder_idx = labels[i].item()
            folder_name = dataset.classes[folder_idx]
            imagenet_idx = folder_to_imagenet_map[folder_name]
            
            orig_pred = original_preds[i].item()
            adv_pred = adversarial_preds[i].item()
            
            print(f"\nImage {i}:")
            print(f"  True class: {class_idx_to_name.get(imagenet_idx, 'Unknown')} (idx: {imagenet_idx})")
            print(f"  Original prediction: {class_idx_to_name.get(orig_pred, 'Unknown')} ({orig_pred}) with {orig_confidence[i].item()*100:.2f}% confidence")
            print(f"  Adversarial prediction: {class_idx_to_name.get(adv_pred, 'Unknown')} ({adv_pred}) with {adv_confidence[i].item()*100:.2f}% confidence")
            print(f"  Attack successful: {orig_pred == imagenet_idx and adv_pred != imagenet_idx}")
            print(f"  Perturbation magnitude: {(adversarial_images[i] - images[i]).abs().max().item():.6f}")
        
        first_batch = False
    
    # Update performance metrics
    batch_size = images.size(0)
    total_images += batch_size
    
    # Evaluate accuracy on adversarial examples
    for i in range(batch_size):
        folder_idx = labels[i].item()
        folder_name = dataset.classes[folder_idx]
        imagenet_idx = folder_to_imagenet_map[folder_name]
        
        # Adversarial predictions accuracy
        if adversarial_preds[i].item() == imagenet_idx:
            correct_adversarial_top1 += 1
        
        if imagenet_idx in top5_adversarial[i]:
            correct_adversarial_top5 += 1
        
        # Store successful attacks for visualization (examples where original was correct but adversarial is wrong)
        orig_pred = original_preds[i].item()
        adv_pred = adversarial_preds[i].item()
        
        if orig_pred == imagenet_idx and adv_pred != imagenet_idx:
            successful_original_images.append(images[i].detach().clone())
            successful_adversarial_images.append(adversarial_images[i].detach().clone())
            successful_true_labels.append(imagenet_idx)
            successful_original_preds.append(orig_pred)
            successful_adversarial_preds.append(adv_pred)
            
            # Stop collecting after getting enough examples
            if len(successful_original_images) >= 5:
                break

# Calculate adversarial accuracy
adversarial_top1_acc = 100 * correct_adversarial_top1 / total_images
adversarial_top5_acc = 100 * correct_adversarial_top5 / total_images

# Print metrics
print("\nAttack Results:")
print(f"Maximum L-infinity distance: {max_l_inf_distance:.6f}")
print(f"Original Test Set - Top-1 Accuracy: {original_top1_acc:.2f}%")
print(f"Original Test Set - Top-5 Accuracy: {original_top5_acc:.2f}%")
print(f"Adversarial Test Set - Top-1 Accuracy: {adversarial_top1_acc:.2f}%")
print(f"Adversarial Test Set - Top-5 Accuracy: {adversarial_top5_acc:.2f}%")
print(f"Top-1 Accuracy Drop: {original_top1_acc - adversarial_top1_acc:.2f}%")
print(f"Top-5 Accuracy Drop: {original_top5_acc - adversarial_top5_acc:.2f}%")

# Visualize successful adversarial examples
if successful_original_images:
    num_examples = min(5, len(successful_original_images))
    visualize_adversarial_examples(
        successful_original_images[:num_examples],
        successful_adversarial_images[:num_examples],
        successful_original_preds[:num_examples],
        successful_adversarial_preds[:num_examples],
        successful_true_labels[:num_examples],
        class_idx_to_name,
        save_path=f"{figures_dir}/successful_attacks.png"
    )
    print(f"Successful attack examples saved to {figures_dir}/successful_attacks.png")
else:
    print("No successful attacks found for visualization")

# Plot the distribution of L-infinity distances
plt.figure(figsize=(10, 6))
plt.hist([l_inf_distance], bins=20)
plt.xlabel("L-infinity Distance")
plt.ylabel("Count")
plt.title(f"Distribution of L-infinity Distances (Max: {max_l_inf_distance:.6f})")
plt.grid(alpha=0.3)
plt.savefig(f"{figures_dir}/l_infinity_distances.png")
print(f"L-infinity distance histogram saved to {figures_dir}/l_infinity_distances.png")

# Visualize adversarial perturbations
if successful_original_images:
    num_examples = min(5, len(successful_original_images))
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 4*num_examples))
    
    for i in range(num_examples):
        # Original image
        axes[i, 0].imshow(denormalize(successful_original_images[i]))
        axes[i, 0].set_title("Original", fontsize=10)
        axes[i, 0].axis('off')
        
        # Adversarial image
        axes[i, 1].imshow(denormalize(successful_adversarial_images[i]))
        axes[i, 1].set_title("Adversarial", fontsize=10)
        axes[i, 1].axis('off')
        
        # Perturbation (scaled for visibility)
        perturbation = (successful_adversarial_images[i] - successful_original_images[i]).abs()
        # Scale perturbation for better visibility
        perturbation = perturbation / perturbation.max() if perturbation.max() > 0 else perturbation
        axes[i, 2].imshow(denormalize(perturbation * 0.5 + 0.5))
        axes[i, 2].set_title("Perturbation (scaled)", fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/adversarial_perturbations.png")
    print(f"Adversarial perturbations visualization saved to {figures_dir}/adversarial_perturbations.png")
else:
    print("No successful attacks found for perturbation visualization")

# Save the adversarial examples to create "Adversarial Test Set 1"
print("Saving adversarial examples to data/adversarial_test_set_1...")
for i, (image, label) in enumerate(tqdm(dataset, desc="Saving adversarial examples")):
    # Get folder name and image file name from the dataset
    image_path = dataset.samples[i][0]
    folder_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.basename(image_path)
    
    # Create the target directory if it doesn't exist
    target_dir = os.path.join("data/adversarial_test_set_1", folder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Process the image through the attack
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Get corresponding ImageNet class index
    folder_idx = dataset.class_to_idx[folder_name]
    imagenet_idx = folder_to_imagenet_map[folder_name]
    imagenet_label = torch.tensor([imagenet_idx]).to(device)
    
    # Generate adversarial example
    adversarial_image = fgsm_attack(pretrained_model, image, imagenet_label, epsilon, targeted=False)
    
    # Save the adversarial image
    adv_image = adversarial_image.squeeze(0)
    adv_image = denormalize(adv_image)
    adv_image = (adv_image * 255).astype(np.uint8)
    img = Image.fromarray(adv_image)
    img.save(os.path.join(target_dir, image_name))

# Verify saved adversarial examples
print("Verifying saved adversarial examples...")
adv_dataset_dir = "data/adversarial_test_set_1"
adv_class_dirs = os.listdir(adv_dataset_dir)
print(f"Number of class directories: {len(adv_class_dirs)}")
if len(adv_class_dirs) > 0:
    sample_class = adv_class_dirs[0]
    sample_class_dir = os.path.join(adv_dataset_dir, sample_class)
    sample_images = os.listdir(sample_class_dir)
    print(f"Sample class '{sample_class}' has {len(sample_images)} images")
    print(f"First few images: {sample_images[:5]}")
else:
    print("No class directories found in adversarial dataset directory!")

# Save results to JSON
results = {
    "task": "Task 2: FGSM Attack",
    "epsilon": epsilon,
    "max_l_infinity_distance": float(max_l_inf_distance),
    "original_top1_accuracy": float(original_top1_acc),
    "original_top5_accuracy": float(original_top5_acc),
    "adversarial_top1_accuracy": float(adversarial_top1_acc),
    "adversarial_top5_accuracy": float(adversarial_top5_acc),
    "top1_accuracy_drop": float(original_top1_acc - adversarial_top1_acc),
    "top5_accuracy_drop": float(original_top5_acc - adversarial_top5_acc)
}

with open("logs/fgsm_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("Results saved to logs/fgsm_results.json") 