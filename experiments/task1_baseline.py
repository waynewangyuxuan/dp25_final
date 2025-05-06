import os
import json
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sys
import datetime

# Create logging directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logging")
os.makedirs(log_dir, exist_ok=True)

# Set up logging to both console and file
log_filename = os.path.join(log_dir, f"task1_baseline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

# Redirect stdout to both terminal and log file
sys.stdout = Logger(log_filename)
print(f"Logging to {log_filename}")

# Set path to the dataset
dataset_path = "./data/TestDataSet"

# Load the model
print("Loading ResNet-34 model...")
pretrained_model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.to(device)
pretrained_model.eval()

# Set up preprocessing as specified in the requirements
print("Creating dataset...")
mean_norms = np.array([0.485, 0.456, 0.406])
std_norms = np.array([0.229, 0.224, 0.225])
plain_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean_norms, std=std_norms)
])

# Load dataset with ImageFolder as specified
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=plain_transforms)
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

# Create dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Function to denormalize images for visualization
def denormalize(tensor):
    tensor = tensor.clone().cpu().detach()
    for t, m, s in zip(tensor, mean_norms, std_norms):
        t.mul_(s).add_(m)
    return tensor

# Function to visualize images and predictions
def visualize_predictions(images, predictions, true_labels, folder_to_imagenet, class_idx_to_name, num_images=5, save_path=None):
    plt.figure(figsize=(15, 3 * num_images))
    
    for i in range(min(num_images, len(images))):
        img = denormalize(images[i])
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Get class names
        folder_idx = true_labels[i].item()
        folder_name = dataset.classes[folder_idx]
        imagenet_idx = folder_to_imagenet.get(folder_name, 401 + folder_idx)
        
        true_class = class_idx_to_name.get(imagenet_idx, f"Unknown ({imagenet_idx})")
        pred_class = "Unknown"
        for idx in predictions[i]:
            if 0 <= idx < 1000:  # Standard ImageNet range
                pred_class = class_idx_to_name.get(idx.item(), f"Unknown ({idx.item()})")
                break
        
        # Plot
        plt.subplot(num_images, 1, i+1)
        plt.imshow(img)
        plt.title(f"True: {true_class}\nPredicted: {pred_class}")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        
    plt.show()

# Evaluate model - Task 1 requirements
print("Evaluating model...")
top1_correct = 0
top5_correct = 0
total = 0

# For visualization
viz_images = []
viz_preds = []
viz_labels = []
folder_to_imagenet_map = {}

for images, labels in tqdm(dataloader):
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = pretrained_model(images)
    
    # Get the top 5 predictions
    _, predicted = outputs.topk(5, 1, True, True)
    
    # Store some samples for visualization
    if len(viz_images) < 5:
        num_to_store = min(5 - len(viz_images), len(images))
        viz_images.extend(images[:num_to_store].cpu())
        viz_preds.extend(predicted[:num_to_store].cpu())
        viz_labels.extend(labels[:num_to_store].cpu())
    
    # Compute top-1 and top-5 accuracy
    batch_size = labels.size(0)
    total += batch_size
    
    # For this test set, we need to map the folder indices to the original ImageNet indices
    # The folder labels (0-99) must be remapped to actual ImageNet classes (401-500)
    
    # Find out what ImageNet indices our folder indices map to
    for i, folder_idx in enumerate(labels):
        folder_idx = folder_idx.item()
        folder_name = dataset.classes[folder_idx]  # Get folder name
        
        # Find corresponding ImageNet class
        imagenet_idx = None
        for label in labels_list:
            if folder_name in label.lower():
                imagenet_idx = int(label.split(":")[0])
                folder_to_imagenet_map[folder_name] = imagenet_idx
                break
        
        if imagenet_idx is None:
            # If we can't find a match, use the pattern that folder indices correspond to
            # ImageNet indices 401-500 in order
            imagenet_idx = 401 + folder_idx
            folder_to_imagenet_map[folder_name] = imagenet_idx
        
        # Check if the model's top prediction matches the actual class
        if predicted[i, 0] == imagenet_idx:
            top1_correct += 1
        
        # Check if the actual class is in the top 5 predictions
        if imagenet_idx in predicted[i]:
            top5_correct += 1

# Calculate accuracies
top1_accuracy = top1_correct / total * 100
top5_accuracy = top5_correct / total * 100

print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

# Visualize sample predictions
print("Generating visualizations...")
figures_dir = "./figures/task1"
os.makedirs(figures_dir, exist_ok=True)
visualize_predictions(
    viz_images, 
    viz_preds, 
    viz_labels, 
    folder_to_imagenet_map, 
    class_idx_to_name,
    save_path=f"{figures_dir}/baseline_predictions.png"
)

# Save class distribution
class_counts = {}
for folder_idx, count in enumerate(dataset.targets.count(folder_idx) for folder_idx in range(len(dataset.classes))):
    folder_name = dataset.classes[folder_idx]
    imagenet_idx = folder_to_imagenet_map.get(folder_name, 401 + folder_idx)
    class_name = class_idx_to_name.get(imagenet_idx, f"Unknown ({imagenet_idx})")
    class_counts[class_name] = count

plt.figure(figsize=(12, 6))
plt.bar(list(class_counts.keys())[:20], list(class_counts.values())[:20])
plt.xticks(rotation=90)
plt.title("Class Distribution (First 20 Classes)")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(f"{figures_dir}/class_distribution.png")
print(f"Class distribution saved to {figures_dir}/class_distribution.png")

# Visualize model confidence
confidences = []
correct_confidences = []
incorrect_confidences = []

print("Analyzing model confidence...")
with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Computing confidences"):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = pretrained_model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get max probability (confidence) for each image
        max_probs, predicted = torch.max(probabilities, dim=1)
        
        # Map to ImageNet classes
        for i, (folder_idx, conf) in enumerate(zip(labels, max_probs)):
            folder_idx = folder_idx.item()
            folder_name = dataset.classes[folder_idx]
            imagenet_idx = folder_to_imagenet_map.get(folder_name, 401 + folder_idx)
            
            # Store confidence value
            conf_val = conf.item()
            confidences.append(conf_val)
            
            # Check if prediction was correct
            if predicted[i] == imagenet_idx:
                correct_confidences.append(conf_val)
            else:
                incorrect_confidences.append(conf_val)

# Plot confidence histogram
plt.figure(figsize=(10, 6))
plt.hist(confidences, bins=20, alpha=0.5, label="All predictions")
plt.hist(correct_confidences, bins=20, alpha=0.5, label="Correct predictions")
plt.hist(incorrect_confidences, bins=20, alpha=0.5, label="Incorrect predictions")
plt.xlabel("Confidence Score")
plt.ylabel("Count")
plt.title("Model Confidence Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{figures_dir}/confidence_distribution.png")
print(f"Confidence distribution saved to {figures_dir}/confidence_distribution.png")

# Visualize the confusion matrix for most confused classes
# Get predictions for a subset of images
num_classes_to_plot = 10  # Top N most frequent classes
top_classes = [folder_idx for folder_idx, _ in sorted(
    [(idx, dataset.targets.count(idx)) for idx in range(len(dataset.classes))],
    key=lambda x: x[1], reverse=True
)][:num_classes_to_plot]

true_labels = []
pred_labels = []

print("Building confusion matrix...")
with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Computing confusion matrix"):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = pretrained_model(images)
        _, predicted = torch.max(outputs, dim=1)
        
        # Map folder indices to ImageNet indices
        for i, folder_idx in enumerate(labels):
            folder_idx = folder_idx.item()
            
            # Only keep top N classes
            if folder_idx not in top_classes:
                continue
                
            folder_name = dataset.classes[folder_idx]
            imagenet_idx = folder_to_imagenet_map.get(folder_name, 401 + folder_idx)
            
            # Store true label
            true_labels.append(folder_idx)
            
            # Store mapped predicted label
            pred_idx = predicted[i].item()
            pred_labels.append(pred_idx)

# Create confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=top_classes)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[dataset.classes[i] for i in top_classes],
            yticklabels=[dataset.classes[i] for i in top_classes])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Top 10 Classes)")
plt.tight_layout()
plt.savefig(f"{figures_dir}/confusion_matrix.png")
print(f"Confusion matrix saved to {figures_dir}/confusion_matrix.png")

# Save results
results = {
    "top1_accuracy": float(top1_accuracy),
    "top5_accuracy": float(top5_accuracy)
}

os.makedirs("logs", exist_ok=True)
with open("logs/baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to logs/baseline_results.json") 