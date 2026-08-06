#!/usr/bin/env bash
# Train FireNet-D5 on the ISFire dataset (fixed seed for reproducibility).
#
# Prerequisites:
#   pip install -r requirements.txt
#   python scripts/apply_patches.py
#   Update configs/data.yaml paths to your local dataset copy.
#
# The GA-tuned hyperparameters are already baked into the patched
# ultralytics default.yaml (configs/default_tuned.yaml), so no extra
# hyperparameter flags are needed here.

set -e
yolo task=detect mode=train \
    model=yolov5s.yaml \
    data=configs/data.yaml \
    pretrained=yolov5s.pt \
    epochs=150 \
    imgsz=640 \
    workers=4 \
    seed=42
