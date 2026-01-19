#!/bin/bash
# One-click script to run all experiments
# Train baselines + Inference + Classification comparison

set -e  # Exit on error

echo "=================================================="
echo "Virtual Staining Model Comparison Pipeline"
echo "Models: Pix2Pix, UNet, cGAN, VAE"
echo "=================================================="

# Configuration
DATA_DIR="/home/23giang.ns/ML_Project/virtual_stainning/both"
BASELINE_OUTPUT="./outputs_baselines"
INFERENCE_OUTPUT="./inference_results"
EPOCHS=120
GPU=7  # Change to available GPU

# ================================================
# Step 1: Train baseline models
# ================================================
echo ""
echo "Step 1/3: Training baseline models..."
echo "Note: Pix2Pix is already trained, skipping..."

# Train UNet
echo ""
echo "[1/3] Training UNet..."
python train_baselines.py \
    --model unet \
    --data_dir $DATA_DIR \
    --output_dir $BASELINE_OUTPUT \
    --epochs $EPOCHS \
    --batch_size 1 \
    --lr 0.0002 \
    --gpu $GPU

# Train cGAN
echo ""
echo "[2/3] Training cGAN..."
python train_baselines.py \
    --model cgan \
    --data_dir $DATA_DIR \
    --output_dir $BASELINE_OUTPUT \
    --epochs $EPOCHS \
    --batch_size 1 \
    --lr 0.0002 \
    --gpu $GPU

# Train VAE
echo ""
echo "[3/3] Training VAE..."
python train_baselines.py \
    --model vae \
    --data_dir $DATA_DIR \
    --output_dir $BASELINE_OUTPUT \
    --epochs $EPOCHS \
    --batch_size 1 \
    --lr 0.0002 \
    --gpu $GPU

echo ""
echo "✓ Baseline training completed!"

# ================================================
# Step 2: Inference all models
# ================================================
echo ""
echo "Step 2/3: Inference all models on test set..."

python inference_all_models.py \
    --model all \
    --data_dir $DATA_DIR \
    --output_dir $INFERENCE_OUTPUT \
    --pix2pix_checkpoint ./output/checkpoints \
    --baseline_checkpoint $BASELINE_OUTPUT \
    --epochs $EPOCHS \
    --gpu $GPU

echo ""
echo "✓ Inference completed!"

# ================================================
# Step 3: Classification comparison
# ================================================
echo ""
echo "Step 3/3: Running classification on all generated images..."
echo "Please run the classification notebook: true-code-for-smids.ipynb"
echo ""
echo "Classification paths:"
echo "  1. Holography (original): $DATA_DIR"
echo "  2. Pix2Pix (generated):   $INFERENCE_OUTPUT/pix2pix"
echo "  3. UNet (generated):      $INFERENCE_OUTPUT/unet"
echo "  4. cGAN (generated):      $INFERENCE_OUTPUT/cgan"
echo "  5. VAE (generated):       $INFERENCE_OUTPUT/vae"

# ================================================
# Summary
# ================================================
echo ""
echo "=================================================="
echo "All experiments completed successfully!"
echo "=================================================="
echo ""
echo "Results saved in:"
echo "  - Baseline checkpoints: $BASELINE_OUTPUT"
echo "  - Generated images:     $INFERENCE_OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Open true-code-for-smids.ipynb"
echo "  2. Load each generated image folder"
echo "  3. Run VGG16 classification"
echo "  4. Compare accuracy across all 4 models"
echo ""
echo "Expected structure for classification:"
echo "  inference_results/"
echo "    ├── pix2pix/  <- Load here for Pix2Pix"
echo "    ├── unet/     <- Load here for UNet"
echo "    ├── cgan/     <- Load here for cGAN"
echo "    └── vae/      <- Load here for VAE"
echo "=================================================="
