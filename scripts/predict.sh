#!/usr/bin/env bash
# Run FireNet-D5 inference on an image, folder, or video.
# Usage: bash scripts/predict.sh weights/best.pt path/to/image_or_video

set -e
WEIGHTS=${1:-weights/best.pt}
SOURCE=${2:?Usage: predict.sh <weights> <image|folder|video>}

yolo task=detect mode=predict \
    model="$WEIGHTS" \
    source="$SOURCE" \
    imgsz=640 \
    conf=0.25 \
    save=True
