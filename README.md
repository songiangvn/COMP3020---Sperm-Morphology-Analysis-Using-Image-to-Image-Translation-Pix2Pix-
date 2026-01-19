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

## Virtual Staining for Holographic Microscopy

The `virtual_stainning/` directory implements image-to-image translation to convert holographic phase contrast microscopy images into virtually stained images, eliminating the need for chemical staining procedures.

### Approaches

This project implements **4 different approaches** for virtual staining:

1. **Pix2Pix (Enhanced)** - Main approach with optimized hyperparameters and enhanced generator architecture
2. **UNet** - Baseline U-Net architecture for direct image translation
3. **cGAN** - Conditional GAN with PatchGAN discriminator
4. **VAE** - Variational Autoencoder for stochastic image generation

### Repository Structure

```
virtual_stainning/
├── train.py                    # Pix2Pix training script
├── train_enhanced.sh           # Optimized training configuration
├── inference_enhanced.py       # Pix2Pix inference
├── preprocessing.py            # Data preprocessing utilities
├── dataset.py                  # Dataset loader for paired images
├── losses.py                   # Custom loss functions
├── classification.ipynb        # Classification evaluation notebook
├── models/                     # Model architectures
│   ├── enhanced_generator.py  # Enhanced Pix2Pix generator
│   ├── pix2pix_modified.py    # Modified Pix2Pix implementation
│   └── baselines.py           # Baseline model implementations
├── holo_stain/                 # Baseline experiments
│   ├── baselines/             # UNet, cGAN, VAE implementations
│   ├── train_baselines.py     # Training script for baselines
│   ├── quick_infer_*.py       # Quick inference scripts per model
│   ├── eval_*_metrics.py      # Evaluation scripts
│   └── outputs_baselines/     # Saved model checkpoints
├── both/                       # Paired dataset (PHASE + STAIN)
├── unstained/                  # Test holographic images
│   └── full_agreement/        # Images with full annotation agreement
│       ├── abnormal/          # Abnormal sperm samples
│       └── normal/            # Normal sperm samples
└── requirements.txt            # Python dependencies
```

### Quick Start

#### Training

**Train Pix2Pix (Main Model):**
```bash
cd virtual_stainning
./train_enhanced.sh
```

**Train Baseline Models:**
```bash
cd holo_stain
python train_baselines.py --model unet --gpu 0
python train_baselines.py --model cgan --gpu 0
python train_baselines.py --model vae --gpu 0
```

#### Inference

**Pix2Pix Inference:**
```bash
python inference_enhanced.py
```

**Baseline Model Inference:**
```bash
cd holo_stain
python quick_infer_unet.py   # U-Net predictions
python quick_infer_cgan.py   # cGAN predictions
python quick_infer_vae.py    # VAE predictions
```

### Dataset

- **Input (PHASE)**: Holographic phase contrast images (256x256 pixels)
- **Target (STAIN)**: Chemically stained ground truth images (256x256 pixels)
- **Split**: 70% training, 15% validation, 15% test (random seed=42)
- **Annotations**: Includes head orientation annotations for quality assessment

### Key Features

- **Enhanced Generator**: Improved architecture with skip connections and attention mechanisms
- **Multiple Loss Functions**: Combines adversarial, L1, perceptual, and SSIM losses
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Comprehensive Evaluation**: PSNR, SSIM, FID metrics for quality assessment
- **Classification Integration**: Evaluates virtual staining quality through downstream classification tasks

### Results

Best visualization results are available in `OLD_augmented_tuned.png`, showing comparisons between:
- Original holographic images
- Virtually stained predictions
- Ground truth stained images

Inference outputs are saved in:
- Pix2Pix: Main outputs directory after training
- Baselines: `holo_stain/inference_{unet,cgan,vae}/`

