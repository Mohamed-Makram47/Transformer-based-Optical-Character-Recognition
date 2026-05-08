#!/bin/bash

# Exit on error
set -e

# Configuration
export MODEL_NAME=ft_SROIE
export SAVE_PATH=./results/${MODEL_NAME}
export LOG_DIR=./logs/log_${MODEL_NAME}
export DATA=./data/SROIE

# Create directories if they don't exist
mkdir -p ${LOG_DIR}
mkdir -p ${SAVE_PATH}

# Check if data exists
if [ ! -d "$DATA" ]; then
    echo "Error: Data directory $DATA does not exist. Please download and extract SROIE dataset."
    exit 1
fi

# Check if pre-trained model exists
PRETRAINED_MODEL="./pretrained/trocr-large-stage1.pt"
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Error: Pre-trained model $PRETRAINED_MODEL not found. Please download it first."
    exit 1
fi

# Hyperparameters
export BSZ=16
export valid_BSZ=16
export LR=5e-05

echo "Starting TrOCR fine-tuning on SROIE..."
echo "Data: $DATA"
echo "Saving to: $SAVE_PATH"
echo "Logging to: $LOG_DIR"

# Run fairseq-train
# Note: Adjust CUDA_VISIBLE_DEVICES and --nproc_per_node according to your hardware
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
    $(which fairseq-train) \
    --data-type SROIE \
    --user-dir ./src \
    --task text_recognition \
    --input-size 384 \
    --arch trocr_large \
    --seed 1111 \
    --optimizer adam \
    --lr ${LR} \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 \
    --warmup-updates 800 \
    --weight-decay 0.0001 \
    --log-format tqdm \
    --log-interval 10 \
    --batch-size ${BSZ} \
    --batch-size-valid ${valid_BSZ} \
    --save-dir ${SAVE_PATH} \
    --tensorboard-logdir ${LOG_DIR} \
    --max-epoch 300 \
    --patience 10 \
    --ddp-backend legacy_ddp \
    --num-workers 10 \
    --preprocess DA2 \
    --bpe gpt2 \
    --decoder-pretrained roberta2 \
    --update-freq 16 \
    --finetune-from-model ${PRETRAINED_MODEL} \
    --fp16 \
    ${DATA}
