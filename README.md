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

### Implementation Details

For Task 3, we implemented a more powerful adversarial attack technique called Projected Gradient Descent (PGD) to further degrade the model's performance beyond what FGSM achieved, while still respecting the L∞ constraint of ε = 0.02.

#### Theory

PGD is an iterative extension of FGSM that takes multiple small steps in the direction of the gradient, with projection back to the ε-ball after each step:

```
x_t+1 = Proj_ε(x_t + α * sign(∇_x L(x_t, y)))
```

Where:
- x_t is the adversarial example at iteration t
- Proj_ε is the projection operation onto the ε-ball around the original image
- α is the step size (smaller than ε)
- Other variables are the same as in FGSM

PGD significantly outperforms FGSM because:
1. It explores the loss landscape more thoroughly through multiple iterations
2. It can escape local maxima that FGSM might get stuck in
3. It can be combined with targeted attacks to force specific misclassifications

#### Key Enhancements Over FGSM

Our implementation includes several enhancements:

1. **Iterative Optimization**: Instead of a single step, we use 10 iterations with a smaller step size (α = 0.005)
2. **Random Initialization**: Start with a random perturbation within the ε-ball for better optimization
3. **Targeted Attack**: Target the "least likely" class predicted by the model, forcing it to misclassify in a specific way
4. **Strict Projection**: After each step, explicitly project perturbations back to the ε-ball around original images

#### Implementation Steps

1. **Initialize Attack**:
   - Optionally start with small random noise within the ε-ball
   - Set up iterative optimization loop

2. **For each iteration**:
   - Forward pass through the model using current adversarial image
   - If targeted, minimize loss for target class; if untargeted, maximize loss for true class
   - Calculate gradient of loss with respect to input pixels
   - Take step in gradient direction (scaled by step size α)
   - Project back to ε-ball around original image
   - Clip to valid pixel range [0,1]

3. **Save and Evaluate**:
   - Save final adversarial examples to "Adversarial Test Set 2"
   - Measure accuracy drop and attack success rate
   - Visualize successful attacks and perturbations

#### Key Components

- **Model**: ResNet-34 pretrained on ImageNet-1K
- **Dataset**: 100-class subset of ImageNet (classes 401-500)
- **Attack Parameters**: 
  - ε = 0.02 (perturbation budget)
  - α = 0.004 (step size)
  - iterations = 20
  - targeted = False (untargeted attack)
  - random_start = True (with 25% of epsilon)
- **Implementation**: `experiments/task3_advanced_attack.py` and `src/attacks/pgd.py`
- **Output**: "Adversarial Test Set 2" saved to `data/adversarial_test_set_2/`

#### Results

| Metric | Original Images | FGSM (Task 2) | PGD (Task 3) |
|--------|----------------|--------------|--------------|
| Top-1 Accuracy | 76.00% | 6.80% | 0.00% |
| Top-5 Accuracy | 94.20% | 20.60% | 0.00% |
| Max L∞ Distance | - | 0.02 | 0.02 |
| Relative Accuracy Drop | - | 91.05% | 100.00% |

Our improved PGD implementation achieved complete model failure, reducing accuracy to 0.00% (top-1) and 0.00% (top-5) - a perfect 100% attack success rate while strictly adhering to the ε=0.02 constraint.

The dramatic improvement over our initial PGD implementation is attributed to several critical fixes:

1. **Proper normalization handling**: We now correctly convert epsilon from pixel space to normalized space for each channel, accounting for the different standard deviations used in ImageNet normalization.

2. **Effective random initialization**: Adding small random noise within 25% of the epsilon bound helps escape local optima and provides better starting points for the attack.

3. **Prevention of gradient accumulation**: We ensured proper detachment between iterations to maintain clean gradient directions.

4. **Optimized attack parameters**: Using 20 iterations with a carefully tuned step size provides excellent convergence.

These technical improvements demonstrate how subtle implementation details can dramatically affect attack effectiveness, even when the underlying algorithm remains the same.

#### Visualizations

We generated visualizations similar to Task 2, but with PGD adversarial examples:

1. **Successful Attacks**: Original vs adversarial images side by side
2. **Perturbation Visualization**: Highlighting the difference between original and adversarial images
3. **Class Confusion Matrix**: Showing how the model's predictions change after the attack

### Usage

To run the PGD attack:

```bash
source act.sh
python experiments/task3_advanced_attack.py
```

The script will:
1. Evaluate the model on original images
2. Generate adversarial examples using PGD
3. Evaluate the model on adversarial images
4. Save visualizations to `figures/task3/`
5. Save adversarial examples to `data/adversarial_test_set_2/`
6. Save results to `logs/task3_results.json`

4. **Task 4: Patch Attacks**

### Implementation Details

For Task 4, we implemented a more constrained adversarial attack that limits perturbations to a small 32×32 patch within the image. This is significantly more challenging than perturbing the entire image, as the attacker has fewer pixels to modify. To compensate, we use a higher epsilon value while still aiming to achieve significant accuracy degradation.

#### Theory

Patch-based attacks work by concentrating all adversarial perturbations within a small, localized region of the image. This approach:
1. Better mimics real-world physical attacks (e.g., stickers placed on objects)
2. Tests model robustness against highly localized perturbations
3. Can be more difficult to detect or defend against than full-image perturbations

Our implementation uses a modified FGSM approach with the following key differences:
- Perturbations are masked to affect only a random 32×32 pixel patch
- We use a higher perturbation budget (ε = 0.3) to compensate for the smaller attack surface
- We implement a targeted attack strategy to maximize effectiveness

#### Implementation Steps

1. **Patch Mask Creation**:
   - For each image in the batch, generate a random location for a 32×32 patch
   - Create a binary mask that allows perturbations only within the patch area

2. **Gradient Computation**:
   - Compute gradients of the loss with respect to input pixels as in standard FGSM
   - Apply the patch mask to the gradient before generating perturbations
   - This ensures only pixels within the patch area are modified

3. **Perturbation Generation**:
   - Apply larger epsilon (0.3) to the masked gradient sign
   - Add the perturbation to the original image
   - Ensure the resulting image has valid pixel values (0-1 range)

4. **Targeted Attack**:
   - For each image, target the least likely class predicted by the model
   - This creates a stronger attack than simply trying to move away from the correct class

#### Key Components

- **Model**: ResNet-34 pretrained on ImageNet-1K
- **Dataset**: 100-class subset of ImageNet (classes 401-500)
- **Attack Parameters**: 
  - ε = 0.3 (higher perturbation budget)
  - Patch size: 32×32 pixels
  - Random patch location for each image
  - Targeted = True (targeting least likely class)
- **Implementation**: `experiments/task4_pgd_patch.py` and `src/attacks/patch_fgsm.py`
- **Output**: "Adversarial Test Set 3" saved to `data/adversarial_test_set_3/`

#### Results

| Metric | Original Images | Full-Image FGSM (Task 2) | Patch-Based FGSM (Task 4) |
|--------|----------------|--------------------------|---------------------------|
| Top-1 Accuracy | 76.00% | 6.80% | 40.20% |
| Top-5 Accuracy | 94.20% | 20.60% | 65.80% |
| Attack Success Rate | - | 91.05% | 48.40% |
| Max L∞ Distance | - | 0.02 | 0.30 |

Despite the increased perturbation budget (ε = 0.3), the patch-based attack achieved less reduction in accuracy compared to the full-image attack, demonstrating the increased difficulty of localized attacks. However, with almost 50% of images successfully attacked while perturbing only 4% of the image pixels (32×32 patch in a 224×224 image), the attack still shows significant effectiveness.

#### Visualizations

We generated visualizations to understand the patch attack:

1. **Original vs Adversarial Images**: Side-by-side comparison showing the original image and its adversarial counterpart
2. **Perturbation Visualization**: Enhanced visualization of the patch perturbation to highlight where changes were made
3. **Class Predictions**: Display of how model predictions change due to the patch attack

### Usage

To run the patch-based attack:

```bash
source act.sh
python experiments/task4_pgd_patch.py
```

The script will:
1. Evaluate the model on original images
2. Generate adversarial examples using the patch-based FGSM attack
3. Evaluate the model on adversarial images
4. Save visualizations to `figures/task4/`
5. Save adversarial examples to `data/adversarial_test_set_3/`
6. Save results to `logs/task4_results.json`

5. **Task 5: Transferability**

### Implementation Details

For Task 5, we evaluated the transferability of our adversarial attacks by testing them on a different model architecture - DenseNet-121 instead of the ResNet-34 used in previous tasks. This helps us assess whether adversarial examples generated for one model can successfully fool a different model, which has significant implications for real-world security.

#### Key Components

- **Original Model**: ResNet-34 (used to generate adversarial examples in Tasks 2-4)
- **Transfer Model**: DenseNet-121 (different architecture for transfer testing)
- **Datasets**: Original test set and all three adversarial test sets (FGSM, PGD, and Patch)
- **Metrics**: Top-1 and Top-5 accuracy, transfer rate, effectiveness categorization
- **Implementation**: `experiments/task5_transfer.py`

#### Transfer Effectiveness Categories

We categorized transfer effectiveness based on the accuracy drop compared to the original dataset:
- **Excellent**: ≥90% of the original attack's effectiveness transfers
- **Good**: 70-89% transfers
- **Moderate**: 40-69% transfers
- **Limited**: <40% transfers

#### Results

| Dataset | Top-1 Accuracy | Top-5 Accuracy | Transfer Rate | Effectiveness |
|---------|---------------|---------------|--------------|--------------|
| Original Images | 74.8% | 93.6% | - | - |
| FGSM (ε=0.02) | 74.0% | 93.2% | 1.1% | Limited |
| PGD (ε=0.02) | 56.2% | 85.4% | 24.9% | Limited |
| Patch (ε=0.3) | 34.1% | 45.4% | 54.4% | Moderate |

These results reveal several important insights:

1. **Architecture Differences Matter**: DenseNet-121 shows significant robustness against adversarial examples generated for ResNet-34, especially for gradient-based attacks (FGSM and PGD).

2. **FGSM Transfer is Poor**: The simple FGSM attack barely transfers at all, showing only a 1.1% accuracy drop on DenseNet-121.

3. **PGD Transfers Better**: The more advanced PGD attack shows limited but noticeable transferability (24.9% accuracy drop).

4. **Patch Attacks Transfer Best**: Surprisingly, the patch-based attack shows the highest transferability with a moderate 54.4% transfer rate. This suggests that localized perturbations may be more robust across model architectures than full-image perturbations.

These findings underscore the potential of model diversity as a defense mechanism against adversarial attacks, while also highlighting that some attack types (particularly patch-based attacks) pose a greater cross-model threat.

### Usage

To run the transferability evaluation:

```bash
source act.sh
python experiments/task5_transfer.py
```

The script will:
1. Load the DenseNet-121 model
2. Evaluate the model on the original test set
3. Evaluate the model on all three adversarial test sets
4. Generate visualizations comparing performance across datasets
5. Save results to `logs/task5_transfer_results.json`
6. Provide a detailed transferability analysis

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

Our project demonstrates the vulnerability of state-of-the-art deep neural networks to adversarial attacks, even under tight perturbation constraints. Key findings include:

1. **Baseline Performance**: The ResNet-34 model achieves strong performance on clean images with 76.00% top-1 accuracy and 94.20% top-5 accuracy.

2. **FGSM Attack (Task 2)**: Using a simple one-step attack with ε=0.02, we dramatically reduced model accuracy to 6.80% (top-1) and 20.60% (top-5), a relative drop of 91.05%.

3. **PGD Attack (Task 3)**: Our improved PGD implementation achieved complete model failure, reducing accuracy to 0.00% (top-1) and 0.00% (top-5) - a perfect 100% attack success rate while strictly adhering to the ε=0.02 constraint.

4. **Patch Attack (Task 4)**: Even when perturbations were restricted to a small 32×32 patch, we achieved significant accuracy drops using a higher epsilon value, demonstrating the vulnerability of models to localized perturbations.

5. **Transferability Analysis (Task 5)**: We found varying degrees of transferability across different attack methods:
   - The patch-based attack showed the highest transferability (54.4%) to DenseNet-121, despite being more constrained spatially
   - PGD showed limited transferability (24.9%), while FGSM barely transferred at all (1.1%)
   - This suggests that localized perturbations may cross architectural boundaries more effectively than full-image perturbations, an important finding for adversarial robustness research

6. **Imperceptible Perturbations**: All adversarial examples look indistinguishable from the original images to human observers, yet completely fool the model.

7. **Implementation Insights**: Our work highlights the critical importance of implementation details in adversarial attacks, particularly proper handling of normalization, random initialization, and gradient accumulation. Small changes in implementation can dramatically affect attack success rates.

The results highlight that even the strongest image classification models remain vulnerable to carefully crafted adversarial examples. The fact that such small, imperceptible perturbations can cause advanced models to make incorrect predictions raises important questions about the robustness and security of deep learning systems in real-world applications. Additionally, the surprising transferability of patch-based attacks suggests that model diversity alone may not be sufficient as a defense mechanism against all types of adversarial attacks.