#!/usr/bin/env bash
python main.py \
    --dataset IMAGENET \
    --model-name ResNet50 \
    --img-size 224 \
    --stem 1 \
    --num-workers 16 \
    --lr 0.1 \
    --batch-size 128
