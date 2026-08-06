#!/usr/bin/env bash
# Genetic-algorithm hyperparameter search (stage 1 of the FireNet-D5 pipeline).
#
# This stage uses the official YOLOv5 repository's built-in genetic
# evolution (--evolve). It was run for 10 generations at 100 epochs per
# generation with the custom SE+BiFPN model definition. The best
# hyperparameters found (generation 6) are recorded in
# docs/ga_evolution_log.txt and baked into configs/default_tuned.yaml,
# which is used for the final 150-epoch training run.
#
# Usage:
#   git clone https://github.com/ultralytics/yolov5
#   cd yolov5 && pip install -r requirements.txt
#   cp ../configs/firenet_d5_model.yaml models/custom_yolov5s.yaml
#   bash ../scripts/evolve_hyperparams.sh /path/to/ISFire/data.yaml

set -e
DATA_YAML=${1:?Usage: evolve_hyperparams.sh /path/to/data.yaml}

python train.py \
    --img 640 \
    --batch 8 \
    --epochs 100 \
    --data "$DATA_YAML" \
    --cfg ./models/custom_yolov5s.yaml \
    --weights '' \
    --cache \
    --noautoanchor \
    --evolve 10
