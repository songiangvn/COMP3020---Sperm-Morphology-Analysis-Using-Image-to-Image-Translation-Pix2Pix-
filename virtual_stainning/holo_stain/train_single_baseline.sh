#!/bin/bash
# Quick start script for training individual baseline models

GPU=7  # Change to available GPU
DATA_DIR="/home/23giang.ns/ML_Project/virtual_stainning/both"
EPOCHS=120

echo "=================================================="
echo "Train Baseline Models Individually"
echo "=================================================="
echo ""
echo "Usage:"
echo "  ./train_single_baseline.sh [model]"
echo ""
echo "Available models:"
echo "  - unet"
echo "  - cgan"
echo "  - vae"
echo ""

if [ $# -eq 0 ]; then
    echo "Error: Please specify a model to train"
    echo "Example: ./train_single_baseline.sh unet"
    exit 1
fi

MODEL=$1

if [ "$MODEL" != "unet" ] && [ "$MODEL" != "cgan" ] && [ "$MODEL" != "vae" ]; then
    echo "Error: Invalid model '$MODEL'"
    echo "Must be one of: unet, cgan, vae"
    exit 1
fi

echo "Training $MODEL on GPU $GPU..."
echo "Data: $DATA_DIR"
echo "Epochs: $EPOCHS"
echo ""

python train_baselines.py \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --output_dir ./outputs_baselines \
    --epochs $EPOCHS \
    --batch_size 1 \
    --lr 0.0002 \
    --gpu $GPU

echo ""
echo "✓ Training completed!"
echo "Checkpoint saved to: ./outputs_baselines/$MODEL/"
