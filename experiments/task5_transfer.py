import os
import json
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import random
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import load_dataset, create_data_loader, get_class_names

# Create logging directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logging")
os.makedirs(log_dir, exist_ok=True)

# Set up logging to both console and file
log_filename = os.path.join(log_dir, f"task5_transfer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Set constants
ORIGINAL_DATASET_PATH = "./data/TestDataSet"
ADV_DATASET_PATHS = [
    "./data/adversarial_test_set_1",  # FGSM (Task 2)
    "./data/adversarial_test_set_2",  # PGD (Task 3)
    "./data/adversarial_test_set_3",  # Patch-based (Task 4)
]
DATASET_NAMES = ["Original", "FGSM (ε=0.02)", "PGD (ε=0.02)", "Patch (ε=0.3)"]
FIGURES_DIR = "./figures/task5"
LOGS_DIR = "./logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Redirect stdout to both terminal and log file
    sys.stdout = Logger(log_filename)
    print(f"Logging to {log_filename}")
    
    # Create output directories
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Load DenseNet-121 model (different from ResNet-34 used in previous tasks)
    print("Loading DenseNet-121 model...")
    model = models.densenet121(weights="IMAGENET1K_V1")
    model = model.to(DEVICE)
    model.eval()
    
    # Print model information
    print(f"Model loaded with output size: {model.classifier.out_features}")
    
    # Initialize normalization transforms
    mean_norms = np.array([0.485, 0.456, 0.406])
    std_norms = np.array([0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_norms, std=std_norms)
    ])
    
    # Load class names from original dataset
    class_names = get_class_names(ORIGINAL_DATASET_PATH)
    
    # Load the original dataset to establish mapping from folder indices to ImageNet classes
    print("Loading original dataset...")
    orig_dataset = datasets.ImageFolder(root=ORIGINAL_DATASET_PATH, transform=transform)
    
    # Create mapping from folder indices to ImageNet class indices
    folder_to_imagenet_map = {}
    
    # Load label mapping from JSON
    json_path = os.path.join(ORIGINAL_DATASET_PATH, "labels_list.json")
    with open(json_path, 'r') as f:
        labels_list = json.load(f)
    
    # Create mapping from class index to name
    class_idx_to_name = {}
    for label in labels_list:
        parts = label.split(": ", 1)
        idx = int(parts[0])
        name = parts[1]
        class_idx_to_name[idx] = name
    
    # Print debug information
    print("\nDEBUG INFO:")
    print(f"Dataset classes: {orig_dataset.classes[:5]}... (total: {len(orig_dataset.classes)})")
    print(f"Number of samples: {len(orig_dataset)}")
    print(f"First few class_idx_to_name entries: {list(class_idx_to_name.items())[:5]}")
    
    # Print sample class mapping
    if len(labels_list) > 1:
        print(f"Labels list sample: {labels_list[:2]}")
    
    # Log some class mapping examples
    for i, cls in enumerate(orig_dataset.classes[:5]):
        print(f"Class {i}: {cls}")
    
    # Initialize results dictionary
    results = {
        "model": "DenseNet-121",
        "datasets": []
    }
    
    # Add original dataset to the list to evaluate
    all_datasets = [ORIGINAL_DATASET_PATH] + ADV_DATASET_PATHS
    
    print("\nTransfer Evaluation Results:")
    print("="*60)
    print(f"{'Dataset':<20}{'Top-1 Accuracy':<20}{'Top-5 Accuracy':<20}")
    print("-"*60)
    
    # Evaluate each dataset
    for i, dataset_path in enumerate(all_datasets):
        dataset_name = DATASET_NAMES[i]
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"{dataset_name:<20}{'N/A':<20}{'N/A':<20} (not found)")
            results["datasets"].append({
                "name": dataset_name,
                "path": dataset_path,
                "exists": False,
                "top1_accuracy": None,
                "top5_accuracy": None
            })
            continue
        
        # Load dataset
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Performance tracking
        correct_top1 = 0
        correct_top5 = 0
        total_images = 0
        
        print(f"\nEvaluating {dataset_name} dataset:")
        print("-" * 40)
        
        # Debug first batch
        first_batch_debug = True
        
        # Create progress bar
        with tqdm(total=len(dataset), desc=f"Evaluating {dataset_name:<12}", ncols=80) as pbar:
            # Evaluate model on dataset
            for images, labels in dataloader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(images)
                    _, top1_preds = torch.max(outputs, 1)
                    _, top5_preds = torch.topk(outputs, 5, dim=1)
                
                # Update performance metrics
                batch_size = images.size(0)
                total_images += batch_size
                
                # Print debug info for the first batch
                if first_batch_debug:
                    print(f"\n{dataset_name.upper()} FIRST BATCH DEBUG:\n")
                    for j in range(min(5, batch_size)):  # First 5 images in batch
                        folder_idx = labels[j].item()
                        folder_name = dataset.classes[folder_idx]
                        
                        # Get ImageNet class index
                        if folder_name in folder_to_imagenet_map:
                            imagenet_idx = folder_to_imagenet_map[folder_name]
                        else:
                            # For original dataset, create mapping
                            if i == 0:
                                # Find corresponding ImageNet class
                                imagenet_idx = None
                                for label in labels_list:
                                    if folder_name.lower() in label.lower():
                                        imagenet_idx = int(label.split(":")[0])
                                        folder_to_imagenet_map[folder_name] = imagenet_idx
                                        break
                                
                                if imagenet_idx is None:
                                    # If we can't find a match, use folder indices with offset
                                    imagenet_idx = 401 + folder_idx
                                    folder_to_imagenet_map[folder_name] = imagenet_idx
                            else:
                                # For adversarial datasets, skip if mapping not found
                                print(f"Warning: Could not find ImageNet mapping for {folder_name}")
                                continue
                        
                        # Get model prediction
                        pred_idx = top1_preds[j].item()
                        pred_score = torch.nn.functional.softmax(outputs[j], dim=0)[pred_idx].item() * 100
                        
                        # Check if prediction is correct
                        is_correct = (pred_idx == imagenet_idx)
                        is_in_top5 = imagenet_idx in top5_preds[j]
                        
                        # Get class names
                        true_class_name = class_idx_to_name.get(imagenet_idx, f"Unknown_{imagenet_idx}")
                        pred_class_name = class_idx_to_name.get(pred_idx, f"Unknown_{pred_idx}")
                        
                        # Print debug info
                        print(f"Image {j}:")
                        print(f"  Folder index: {folder_idx}, Folder name: {folder_name}")
                        print(f"  Mapped ImageNet index: {imagenet_idx}")
                        print(f"  ImageNet class name: {true_class_name}")
                        print(f"  Model prediction: {pred_idx}")
                        print(f"  Model confidence: {pred_score:.2f}%")
                        print(f"  Prediction class name: {pred_class_name}")
                        print(f"  Is correct: {is_correct}")
                        print(f"  Top 5 predictions: {top5_preds[j].tolist()}")
                        print(f"  Is in top 5: {is_in_top5}")
                        print()
                    
                    first_batch_debug = False
                
                # Process the whole batch for metrics
                for j in range(batch_size):
                    folder_idx = labels[j].item()
                    folder_name = dataset.classes[folder_idx]
                    
                    # For original dataset, we need to map folder index to ImageNet index
                    if i == 0:
                        if folder_name not in folder_to_imagenet_map:
                            # Find corresponding ImageNet class
                            imagenet_idx = None
                            for label in labels_list:
                                if folder_name.lower() in label.lower():
                                    imagenet_idx = int(label.split(":")[0])
                                    folder_to_imagenet_map[folder_name] = imagenet_idx
                                    break
                            
                            if imagenet_idx is None:
                                # If we can't find a match, use folder indices with offset
                                imagenet_idx = 401 + folder_idx
                                folder_to_imagenet_map[folder_name] = imagenet_idx
                        
                        imagenet_idx = folder_to_imagenet_map[folder_name]
                    else:
                        # For adversarial datasets, we can reuse the mapping
                        if folder_name not in folder_to_imagenet_map:
                            # Skip without printing warning to avoid terminal clutter
                            continue
                        imagenet_idx = folder_to_imagenet_map[folder_name]
                    
                    # Calculate Top-1 accuracy
                    if top1_preds[j].item() == imagenet_idx:
                        correct_top1 += 1
                    
                    # Calculate Top-5 accuracy
                    if imagenet_idx in top5_preds[j]:
                        correct_top5 += 1
                
                # Update progress bar
                pbar.update(batch_size)
        
        # Calculate accuracy
        top1_acc = 100.0 * correct_top1 / total_images
        top5_acc = 100.0 * correct_top5 / total_images
        
        # Print results in table format
        print(f"{dataset_name:<20}{top1_acc:<10.2f}%{'':<9}{top5_acc:<10.2f}%{'':<9}")
        
        # Add results to dictionary
        results["datasets"].append({
            "name": dataset_name,
            "path": dataset_path,
            "exists": True,
            "top1_accuracy": float(top1_acc),
            "top5_accuracy": float(top5_acc)
        })
    
    print("="*60)
    
    # Save results to JSON
    with open(os.path.join(LOGS_DIR, "task5_transfer_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to logs/task5_transfer_results.json")
    
    # Create visualization of results
    create_accuracy_chart(results)
    
    # Print transferability analysis
    print("\nTransferability Analysis:")
    print("="*60)
    print(f"{'Dataset':<20}{'Transfer Rate':<20}{'Effectiveness':<20}")
    print("-"*60)
    
    # Get baseline accuracies
    baseline_top1 = results["datasets"][0]["top1_accuracy"] if results["datasets"][0]["exists"] else 0
    
    # Summary statistics
    transfer_summary = {
        "excellent": 0,
        "good": 0,
        "moderate": 0,
        "limited": 0
    }
    
    for dataset in results["datasets"]:
        if not dataset["exists"] or dataset["name"] == "Original":
            continue
        
        # Calculate transferability (how well attacks transfer)
        transfer_rate = (baseline_top1 - dataset["top1_accuracy"]) / baseline_top1 * 100
        
        # Determine effectiveness category
        if transfer_rate >= 90:
            effectiveness = "Excellent"
            transfer_summary["excellent"] += 1
        elif transfer_rate >= 70:
            effectiveness = "Good"
            transfer_summary["good"] += 1
        elif transfer_rate >= 40:
            effectiveness = "Moderate"
            transfer_summary["moderate"] += 1
        else:
            effectiveness = "Limited"
            transfer_summary["limited"] += 1
        
        print(f"{dataset['name']:<20}{transfer_rate:<10.1f}%{'':<9}{effectiveness:<20}")
    
    print("="*60)
    
    # Add summary statistics to results
    results["transfer_summary"] = transfer_summary
    results["timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save updated results to JSON
    with open(os.path.join(LOGS_DIR, "task5_transfer_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Print overall findings
    print("\nOverall Findings:")
    print("-"*60)
    print(f"- Total attacks evaluated: {len(results['datasets']) - 1}")
    print(f"- Attacks with excellent transferability: {transfer_summary['excellent']}")
    print(f"- Attacks with good transferability: {transfer_summary['good']}")
    print(f"- Attacks with moderate transferability: {transfer_summary['moderate']}")
    print(f"- Attacks with limited transferability: {transfer_summary['limited']}")
    
    if transfer_summary["excellent"] + transfer_summary["good"] > 0:
        print("\nKey insight: Some attacks transfer well between models, indicating")
        print("a shared vulnerability in the underlying neural network architecture.")
    else:
        print("\nKey insight: The DenseNet-121 model shows resistance to transfer attacks,")
        print("suggesting that model diversity is an effective defense strategy.")
    
    print(f"\nTask 5 completed successfully at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!")


def create_accuracy_chart(results):
    """Create a bar chart comparing accuracies across datasets"""
    # Extract data
    datasets = []
    top1_accuracies = []
    top5_accuracies = []
    
    for dataset in results["datasets"]:
        if dataset["exists"]:
            datasets.append(dataset["name"])
            top1_accuracies.append(dataset["top1_accuracy"])
            top5_accuracies.append(dataset["top5_accuracy"])
    
    # Create bar chart
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, top1_accuracies, width, label='Top-1 Accuracy')
    bars2 = ax.bar(x + width/2, top5_accuracies, width, label='Top-5 Accuracy')
    
    # Add labels and title
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('DenseNet-121 Accuracy on Original and Adversarial Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    # Add exact values on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Adjust bottom margin
    plt.subplots_adjust(bottom=0.15)
    
    # Add grid lines for readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "transfer_results.png"))
    plt.close()
    print(f"Visualization saved to {FIGURES_DIR}/transfer_results.png")


if __name__ == "__main__":
    main() 