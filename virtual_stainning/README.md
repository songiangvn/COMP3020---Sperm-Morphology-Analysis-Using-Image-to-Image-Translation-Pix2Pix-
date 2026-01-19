# Virtual Staining for Holographic Microscopy

**4 Approaches for Virtual Staining:**
1. **Pix2Pix (Enhanced)** - Main approach with optimized hyperparameters
2. **UNet** - Baseline U-Net architecture
3. **cGAN** - Conditional GAN with PatchGAN discriminator
4. **VAE** - Variational Autoencoder

## Repository Structure

```
virtual_stainning/
├── train.py                  # Pix2Pix training
├── train_enhanced.sh         # Training script
├── inference_enhanced.py     # Pix2Pix inference
├── preprocessing.py          # Preprocessing utilities
├── dataset.py               # Dataset loader
├── losses.py                # Loss functions
├── models/                  # Model architectures
│   ├── generator.py         # Pix2Pix generator
│   ├── discriminator.py     # Pix2Pix discriminator
│   └── baselines.py         # Baseline models
├── holo_stain/              # Baseline implementations
│   ├── baselines/           # UNet, cGAN, VAE
│   ├── train_baselines.py   # Training script
│   ├── quick_infer_*.py     # Inference scripts
│   └── outputs_baselines/   # Best checkpoints
├── both/                    # Dataset (PHASE + STAIN)
├── unstained/              # Test images
├── true-code-for-smids.ipynb  # Classification notebook
├── OLD_augmented_tuned.png  # Best result visualization
└── requirements.txt

```

## Quick Start

### 1. Train Pix2Pix (Main Model)
```bash
./train_enhanced.sh
```

### 2. Train Baselines
```bash
cd holo_stain
python train_baselines.py --model unet --gpu 0
python train_baselines.py --model cgan --gpu 0
python train_baselines.py --model vae --gpu 0
```

### 3. Inference
```bash
# Pix2Pix
python inference_enhanced.py

# Baselines
cd holo_stain
python quick_infer_unet.py
python quick_infer_cgan.py
python quick_infer_vae.py
```

## Dataset

- **PHASE images**: Holographic phase contrast (256x256)
- **STAIN images**: Stained ground truth (256x256)
- **Split**: 70% train, 15% val, 15% test (seed=42)

## Results

Best visualization: [OLD_augmented_tuned.png](OLD_augmented_tuned.png)

Inference outputs:
- Pix2Pix: Check outputs directory after training
- Baselines: `holo_stain/inference_{unet,cgan,vae}/`
