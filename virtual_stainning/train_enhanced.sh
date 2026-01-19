#!/bin/bash
# Training script cho Enhanced Virtual Staining Model
# Optimized cho dataset 334 images

set -e  # Exit on error

echo "================================================"
echo "Enhanced Virtual Staining - Training Script"
echo "================================================"

# Configuration
MODEL="pix2pix_enhanced"
EXP_NAME="enhanced_v1"
DATA_DIR="/home/23giang.ns/ML_Project/virtual_stainning/both"
OUTPUT_DIR="/home/23giang.ns/ML_Project/virtual_stainning/outputs"

# Hyperparameters (optimized for small dataset)
BATCH_SIZE=8
NUM_EPOCHS=400
LR_G=0.0001
LR_D=0.00005
LAMBDA_L1=150.0
NUM_WORKERS=4

echo ""
echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Experiment: ${EXP_NAME}"
echo "  Data: ${DATA_DIR}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning Rate (G): ${LR_G}"
echo "  Learning Rate (D): ${LR_D}"
echo "  Lambda L1: ${LAMBDA_L1}"
echo ""

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

# Count images
NUM_PHASE=$(ls ${DATA_DIR}/*PHASE*.png 2>/dev/null | wc -l)
NUM_STAIN=$(ls ${DATA_DIR}/*STAIN*.png 2>/dev/null | wc -l)

echo "Dataset Info:"
echo "  Phase images: ${NUM_PHASE}"
echo "  Stain images: ${NUM_STAIN}"
echo ""

if [ $NUM_PHASE -eq 0 ] || [ $NUM_STAIN -eq 0 ]; then
    echo "ERROR: No images found in data directory!"
    exit 1
fi

# Check GPU
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || {
    echo "WARNING: nvidia-smi failed. GPU might not be available."
}
echo ""

# Start training
echo "Starting training..."
echo "Logs will be saved to: ${OUTPUT_DIR}/${MODEL}/${EXP_NAME}/logs/"
echo ""

python train.py \
    --model ${MODEL} \
    --exp_name ${EXP_NAME} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr_g ${LR_G} \
    --lr_d ${LR_D} \
    --lambda_l1 ${LAMBDA_L1} \
    --num_workers ${NUM_WORKERS}

echo ""
echo "================================================"
echo "Training completed!"
echo "================================================"
echo "Output directory: ${OUTPUT_DIR}/${MODEL}/${EXP_NAME}/"
echo "Best model: ${OUTPUT_DIR}/${MODEL}/${EXP_NAME}/checkpoints/best_model.pth"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir ${OUTPUT_DIR}/${MODEL}/${EXP_NAME}/logs/"
echo ""
