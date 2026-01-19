<<<<<<< HEAD
# COMP3020---Sperm-Morphology-Analysis-Using-Image-to-Image-Translation-Pix2Pix-

## Synthetic Data Augmentation

The `synthetic_data_augmentation/` directory contains notebooks for generating synthetic sperm images and evaluating classification performance using augmented datasets.

### Contents

#### `synthetic_data_generation.ipynb`
This notebook implements synthetic sperm image generation using U-Net based segmentation models. Key features include:
- **COCO format dataset loading**: Processes sperm annotations with head and tail segmentation
- **Multi-channel mask generation**: Creates separate masks for head and tail components
- **U-Net architecture**: Implements a segmentation model with encoder-decoder structure
- **Synthetic data generation**: Generates new sperm images by combining segmented components
- **Data augmentation pipeline**: Includes horizontal/vertical flips, brightness/contrast adjustments using Albumentations

#### `HuSHem_classification.ipynb`
Classification experiments on the HuSHem dataset with the following capabilities:
- **5-fold cross-validation**: Implements stratified k-fold CV for robust evaluation
- **Transfer learning**: Leverages pre-trained models (ResNet, VGG, etc.) for sperm classification
- **Binary classification**: Normal vs. abnormal sperm morphology
- **Performance metrics**: Calculates accuracy, precision, recall, F1-score, and confusion matrices
- **Data handling**: Includes functions for fold generation, weighted sampling for class imbalance

#### `SMIDS_classification.ipynb`
Similar classification pipeline applied to the SMIDS (Stained Microscopic Images of Sperm) dataset:
- **Cross-validation framework**: Implements k-fold validation with configurable splits
- **Deep learning models**: Tests various CNN architectures on SMIDS data
- **Augmentation evaluation**: Assesses the impact of synthetic data on classification performance
- **Reproducibility**: Supports multiple seeds for statistical significance testing

### Usage

These notebooks are designed to run in environments with GPU support (e.g., Kaggle, Google Colab). They require:
- PyTorch and torchvision
- Albumentations for data augmentation
- Segmentation-models-pytorch for U-Net implementation
- Standard scientific computing libraries (numpy, pandas, matplotlib, scikit-learn)

The workflow typically involves:
1. Generate synthetic data using `synthetic_data_generation.ipynb`
2. Evaluate classification performance with augmented datasets using the classification notebooks
3. Compare results with and without synthetic data augmentation

