# Virtual Staining Model Comparison

## Overview
This project compares 4 different models for holography-to-stained image translation:

1. **Pix2Pix (HoloStain)** - Paper's original model ✅ Currently training
2. **UNet** - Simple encoder-decoder baseline
3. **cGAN** - Standard conditional GAN
4. **VAE** - Variational Autoencoder baseline

## Project Structure

```
holo_stain/
├── baselines/                    # Baseline model architectures
│   ├── unet.py                   # UNet model
│   ├── cgan.py                   # Conditional GAN
│   └── vae.py                    # Variational Autoencoder
│
├── main.py                       # Pix2Pix training (original paper)
├── model.py                      # Pix2Pix architecture
├── train.py                      # Pix2Pix training loop
├── test.py                       # Pix2Pix inference
│
├── train_baselines.py            # Train 3 baseline models
├── inference_all_models.py       # Infer all 4 models
│
├── run_all_experiments.sh        # Complete pipeline (train+infer)
├── train_single_baseline.sh      # Train one baseline
│
├── output/                       # Pix2Pix outputs
│   └── checkpoints/              # Pix2Pix trained weights
│
├── outputs_baselines/            # Baseline model outputs
│   ├── unet/
│   ├── cgan/
│   └── vae/
│
└── inference_results/            # Generated images for comparison
    ├── pix2pix/
    ├── unet/
    ├── cgan/
    └── vae/
```

## Quick Start

### 1. Train All Models (One-Click)

```bash
# Train all 3 baselines + inference all 4 models
./run_all_experiments.sh
```

This will:
- Train UNet, cGAN, VAE (Pix2Pix already trained)
- Generate virtual stained images for all models
- Save results in `inference_results/`

### 2. Train Individual Baseline

```bash
# Train only UNet
./train_single_baseline.sh unet

# Train only cGAN
./train_single_baseline.sh cgan

# Train only VAE
./train_single_baseline.sh vae
```

### 3. Manual Training

```bash
# Train specific model
python train_baselines.py \
    --model unet \
    --data_dir /home/23giang.ns/ML_Project/virtual_stainning/both \
    --output_dir ./outputs_baselines \
    --epochs 120 \
    --batch_size 1 \
    --lr 0.0002 \
    --gpu 7
```

### 4. Inference All Models

```bash
python inference_all_models.py \
    --model all \
    --data_dir /home/23giang.ns/ML_Project/virtual_stainning/both \
    --output_dir ./inference_results \
    --pix2pix_checkpoint ./output/checkpoints \
    --baseline_checkpoint ./outputs_baselines \
    --epochs 120 \
    --gpu 7
```

## Classification Comparison

After inference, run classification on generated images:

1. Open `true-code-for-smids.ipynb`
2. Load each model's generated images:
   - `inference_results/pix2pix/`
   - `inference_results/unet/`
   - `inference_results/cgan/`
   - `inference_results/vae/`
3. Run VGG16 classification
4. Compare accuracy

## Model Details

### Pix2Pix (HoloStain)
- **Architecture**: U-Net generator + PatchGAN discriminator
- **Loss**: Adversarial loss + L1 reconstruction
- **Status**: ✅ Training (epoch 55/120)
- **Performance**: D_loss ~0.65, G_loss decreasing

### UNet
- **Architecture**: Standard encoder-decoder with skip connections
- **Loss**: L1 reconstruction only (no GAN)
- **Pros**: Fast, stable training
- **Cons**: May lack fine details without adversarial loss

### cGAN
- **Architecture**: U-Net generator + PatchGAN discriminator
- **Loss**: Adversarial loss + L1 (λ=100)
- **Pros**: Similar to Pix2Pix, good baseline
- **Cons**: May have mode collapse

### VAE
- **Architecture**: Encoder-decoder with latent space
- **Loss**: Reconstruction + KL divergence
- **Pros**: Probabilistic, handles uncertainty
- **Cons**: May produce blurry images

## Expected Results

| Model | Training Time | Expected Accuracy | Strengths |
|-------|--------------|-------------------|-----------|
| **Pix2Pix** | 2h | Highest | Paper's method, adversarial loss |
| **UNet** | 1.5h | Good | Fast, stable |
| **cGAN** | 2h | Similar to Pix2Pix | Standard baseline |
| **VAE** | 1.5h | Lower | Handles uncertainty |

## Training Status

- ✅ Pix2Pix: Epoch 55/120 (in progress)
- ⏳ UNet: Not started
- ⏳ cGAN: Not started
- ⏳ VAE: Not started

## Hardware Requirements

- **GPU**: NVIDIA RTX 3090 (22GB VRAM)
- **Training time**: ~2 hours per model (120 epochs)
- **Total time**: ~6 hours for all 3 baselines

## Dataset

- **Source**: `/home/23giang.ns/ML_Project/virtual_stainning/both`
- **Format**: Concatenated images (holography | stained)
- **Split**: 80% train, 20% test
- **Size**: 263 images (210 train, 53 test)

## Citation

```
Original Paper: HoloStain - Virtual Staining for Digital Holographic Microscopy
Implementation: Comparison of 4 virtual staining approaches
```

## Contact

For questions or issues, refer to the original paper or contact the authors.
