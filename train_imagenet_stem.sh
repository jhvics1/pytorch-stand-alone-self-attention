#!/usr/bin/env bash
python main.py \
    --dataset IMAGENET \
    --model-name ResNet50 \
    --img-size 224 \
    --stem 1 \
    --lr 0.01 \
    --batch-size 48
