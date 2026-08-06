#!/usr/bin/env bash
# Evaluate a trained FireNet-D5 checkpoint on the ISFire test split.
# Usage: bash scripts/validate.sh weights/best.pt

set -e
WEIGHTS=${1:-weights/best.pt}

yolo task=detect mode=val \
    model="$WEIGHTS" \
    data=configs/data.yaml \
    split=test \
    imgsz=640
