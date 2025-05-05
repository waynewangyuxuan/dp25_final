# Deep Learning Project 3: Jailbreaking Deep Models

This project focuses on crafting adversarial attacks on production-grade image classifiers to degrade their performance while keeping perturbations imperceptible. The target model is a ResNet-34 trained on ImageNet-1K.

## Project Structure

```
deep‑models‑jailbreak/
│
├── README.md                  # Project documentation
├── LICENSE
├── .gitignore                 # Excludes data/, logs/, figures/*.png, checkpoints/
├── requirements.txt           # Python dependencies
├── requirements_gpu.txt
├── Makefile                   # Automation for tasks
├── setup.sh                   # Setup script for environment and dependencies
│
├── src/                       # Core implementation code
│   ├── __init__.py
│   ├── dataset.py             # Dataset loading utilities
│   ├── metrics.py             # Evaluation metrics
│   ├── viz.py                 # Visualization utilities
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── fgsm.py            # Single-step attack implementation
│   │   ├── pgd_full.py        # Iterative full-image attack
│   │   ├── pgd_patch.py       # Iterative 32×32 patch attack
│   │   └── utils.py           # Attack utilities
│
├── experiments/               # Experiment scripts for each task
│   ├── task1_baseline.py      # Evaluate vanilla ResNet-34
│   ├── task2_fgsm.py          # Generates adv_test_set_1
│   ├── task3_pgd_full.py      # Generates adv_test_set_2
│   ├── task4_pgd_patch.py     # Generates adv_test_set_3
│   └── task5_transfer.py      # Cross-model evaluation
│
├── configs/                   # Configuration files
│   ├── fgsm.yaml
│   ├── pgd_full.yaml
│   └── pgd_patch.yaml
│
├── data/                      # Data directory (git-ignored)
│   ├── TestDataSet/           # Original ImageNet subset 
│   ├── adv_test_set_1/        # FGSM adversarial examples
│   ├── adv_test_set_2/        # Full-image PGD adversarial examples
│   └── adv_test_set_3/        # Patch-based adversarial examples
│
├── logs/                      # Logs directory (git-ignored)
│   ├── baseline_results.json  # Baseline evaluation results
│   ├── pgd_full_metrics.csv   # Training metrics
│   └── pgd_patch_metrics.csv  # Training metrics
│
├── figures/                   # Generated visualizations (git-ignored)
│   ├── fgsm_examples.png
│   ├── pgd_full_examples.png
│   ├── pgd_patch_examples.png
│   └── perturbation_histograms.png
│
├── checkpoints/               # Model checkpoints (git-ignored)
│
└── report/                    # AAAI camera-ready
    ├── main.tex
    ├── main.pdf
    ├── figs/
    │   ├── pgd_full_examples.png
    │   ├── pgd_patch_examples.png
    │   └── training_curves_combined.png  # (curve_full + curve_patch composited)
    └── appendix/
```

## Tasks

1. **Task 1: Baseline Evaluation**

### Implementation Details

The baseline evaluation measures the performance of a pretrained ResNet-34 model on a subset of the ImageNet dataset. This evaluation serves as our performance reference before applying adversarial attacks.

#### Key Components

- **Model**: ResNet-34 pretrained on ImageNet-1K
- **Dataset**: 100-class subset of ImageNet (classes 401-500)
- **Metrics**: Top-1 and Top-5 accuracy
- **Implementation**: `experiments/task1_baseline.py`

#### Image Preprocessing

```python
mean_norms = np.array([0.485, 0.456, 0.406])
std_norms = np.array([0.229, 0.224, 0.225])
plain_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean_norms, std=std_norms)
])
```

#### Results

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 76.00% |
| Top-5 Accuracy | 94.20% |

These results indicate the model's strong performance on clean, unmodified images. The high top-5 accuracy suggests that even when the model doesn't predict the exact correct class, it usually includes the correct class among its top five predictions.

### Usage

To reproduce the baseline evaluation:

```bash
source act.sh
python experiments/task1_baseline.py
```

The results will be saved to `logs/baseline_results.json`.

2. **Task 2: FGSM Attacks**

### Implementation Details

The Fast Gradient Sign Method (FGSM) is a simple yet effective adversarial attack technique that perturbs input images to maximize the loss, causing misclassification while keeping perturbations visually imperceptible.

#### Theory

FGSM works by taking a single step in the direction of the gradient of the loss with respect to the input image:

```
x_adv = x + ε * sign(∇_x L(x, y))
```

Where:
- x is the original image
- y is the true label
- L is the cross-entropy loss
- ε is the perturbation budget (0.02 in our implementation)
- sign(∇_x L) produces a tensor with the same shape as x with elements set to +1 or -1 depending on the sign of the gradient

#### Implementation Steps

1. **Input Processing**:
   - Load the images and map folder indices to ImageNet class indices
   - Move tensors to the appropriate device (CPU/GPU)

2. **Gradient Computation**:
   - Set `requires_grad_(True)` on the input images
   - Forward pass through the model
   - Compute the cross-entropy loss between predictions and true labels
   - Backward pass to calculate gradients with respect to input pixels

3. **Perturbation Generation**:
   - Take the sign of the gradients
   - Multiply by epsilon (0.02) to scale the perturbation
   - Add to the original images

4. **Constraint Enforcement**:
   - Clip resulting pixel values to [0,1] range to ensure valid images
   - Explicitly constrain the L∞ distance between original and adversarial images to ε

5. **Save and Evaluate**:
   - Save adversarial examples to "Adversarial Test Set 1"
   - Measure accuracy drop on the adversarial set
   - Visualize successful attacks and perturbations

#### Implementation Challenges

We addressed several key challenges in our implementation:
- Ensuring proper gradient calculation by avoiding `torch.no_grad()` during the attack
- Maintaining consistent batch order by using `shuffle=False` in DataLoader
- Properly clipping perturbations to enforce the strict ε constraint
- Mapping dataset folder indices to actual ImageNet class indices for evaluation

#### Key Components

- **Model**: ResNet-34 pretrained on ImageNet-1K
- **Dataset**: 100-class subset of ImageNet (classes 401-500)
- **Attack Budget**: ε = 0.02 (controls perturbation magnitude)
- **Implementation**: `experiments/task2_fgsm.py` and `src/attacks/fgsm.py`
- **Output**: "Adversarial Test Set 1" saved to `data/adversarial_test_set_1/`

#### Results

| Metric | Original Images | Adversarial Images | Change |
|--------|----------------|-------------------|--------|
| Top-1 Accuracy | 76.00% | 6.80% | -69.20% |
| Top-5 Accuracy | 94.20% | 20.60% | -73.60% |
| Max L∞ Distance | - | 0.02 | - |

These results demonstrate the extreme vulnerability of deep neural networks to carefully crafted perturbations. With perturbations constrained to ε = 0.02 (imperceptible to humans), we achieved a dramatic accuracy drop of 69.20% for top-1 accuracy and 73.60% for top-5 accuracy, far exceeding the target of 50% accuracy reduction.

#### Visualizations

We generated several visualizations to understand the attack:

1. **Successful Attacks**: Original vs adversarial images side by side, showing how imperceptible changes lead to misclassification
2. **Perturbation Visualization**: Highlighting the difference between original and adversarial images (scaled for visibility)
3. **L-infinity Distance Distribution**: Confirming that perturbations stay within the ε constraint

### Usage

To run the FGSM attack:

```bash
source act.sh
python experiments/task2_fgsm.py
```

The script will:
1. Evaluate the model on original images
2. Generate adversarial examples using FGSM
3. Evaluate the model on adversarial images
4. Save visualizations to `figures/task2/`
5. Save adversarial examples to `data/adversarial_test_set_1/`
6. Save results to `logs/fgsm_results.json`

3. **Task 3: Improved Attacks**
   - Implement a stronger attack (e.g., PGD)
   - Same budget constraint: ε = 0.02
   - Save results as "Adversarial Test Set 2"
   - Target: 70% accuracy drop

4. **Task 4: Patch Attacks**
   - Restrict perturbations to a random 32×32 patch
   - Use higher ε value (0.3 - 0.5)
   - Save results as "Adversarial Test Set 3"

5. **Task 5: Transferability**
   - Evaluate a different model (e.g., DenseNet-121)
   - Report accuracy on original and adversarial datasets
   - Discuss findings and mitigation strategies

## Setup and Usage

1. **Setup the environment**:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate the environment**:
   ```
   # To activate the environment and set up paths
   source act.sh
   ```

3. **Run individual tasks**:
   ```
   # Using Make
   make task1
   make task2
   make task3
   make task4
   make task5

   # Or directly
   python experiments/task1_baseline.py
   ```

4. **Run all tasks**:
   ```
   make all
   ```

5. **Clean generated files**:
   ```
   make clean
   ```

## Results

The project aims to demonstrate:
- Baseline performance of ResNet-34 on ImageNet
- Effectiveness of different adversarial attacks
- Importance of perturbation constraints 
- Transferability of attacks across models