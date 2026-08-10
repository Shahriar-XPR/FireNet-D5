# FireNet-D5

**A Genetic Algorithm-Optimized Architecture with BiFPN and SE Attention for Multi-Class Indoor Fire and Smoke Detection**

This repository contains all author-generated code underpinning the findings of the manuscript *FireNet-D5* (PLOS ONE). The code, trained weights, and dataset are released without restrictions to support full reproducibility and reuse, in line with the [PLOS ONE code sharing policy](https://journals.plos.org/plosone/s/materials-and-software-sharing#loc-sharing-code).

## Overview

FireNet-D5 is a modified YOLOv5s detector for four indoor fire/smoke classes:

| Class ID | Class name |
|---|---|
| 0 | black smoke |
| 1 | blue fire |
| 2 | white-gray smoke |
| 3 | yellow-orange fire |

Three modifications to the YOLOv5s baseline:

1. **SE attention** — a Squeeze-and-Excitation channel-attention block inserted in the backbone immediately before SPPF (layer 9 in `configs/firenet_d5_model.yaml`).
2. **BiFPN fusion** — the plain `Concat` operations in the neck are replaced with learnable weighted `BiFPN_Concat2` fusion nodes.
3. **Genetic-algorithm hyperparameter optimization** — YOLOv5's built-in genetic evolution (10 generations × 100 epochs per candidate) is used to tune the training hyperparameters; the best configuration is provided in `configs/default_tuned.yaml` and used for the final 150-epoch training run.

## Repository layout

```
firenet-d5-code/
├── README.md
├── LICENSE                      # AGPL-3.0 (inherited from Ultralytics)
├── CITATION.cff
├── requirements.txt
├── configs/
│   ├── firenet_d5_model.yaml    # FireNet-D5 architecture (SE + BiFPN)
│   ├── default_tuned.yaml       # Ultralytics default.yaml with GA-tuned hyperparameters
│   └── data.yaml                # ISFire dataset configuration (edit paths)
├── patched_ultralytics/         # Patched files installed into ultralytics==8.3.33
│   └── nn/
│       ├── tasks.py             # Registers SqueezeExcitation & BiFPN modules in the parser
│       └── modules/
│           ├── __init__.py
│           ├── block.py
│           └── conv.py          # Defines SqueezeExcitation, BiFPN_Concat2/3
├── scripts/
│   ├── apply_patches.py         # Installs the patched files into the ultralytics package
│   ├── download_dataset.py      # Downloads the ISFire dataset from Hugging Face
│   ├── evolve_hyperparams.sh    # Stage 1: GA hyperparameter search (YOLOv5 --evolve)
│   ├── train.sh                 # Stage 2: final training (seed 42, 150 epochs)
│   ├── validate.sh              # Evaluation on the test split
│   ├── predict.sh               # Inference on images/videos
│   └── gradcam.py               # Grad-CAM explainability visualizations
└── weights/
    ├── best.pt                  # Best checkpoint (used for all reported results)
    └── last.pt                  # Final-epoch checkpoint
```

## Dataset

The ISFire dataset (Fire_Smoke_4C) is publicly available on Hugging Face under a CC-BY-4.0 license:

**https://huggingface.co/datasets/shahriar-5/Fire_Smoke_4C**

Download and extract it with:

```bash
python scripts/download_dataset.py --output ./ISFire
```

Then update the `train` / `val` / `test` paths in `configs/data.yaml` to point to the extracted image folders.

## Reproducing the results

### 1. Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Developed on Python 3.12 with `ultralytics==8.3.33`. A CUDA-capable GPU is strongly recommended.

### 2. Apply the architecture patches

The SE and BiFPN modules must be registered inside the installed Ultralytics package:

```bash
python scripts/apply_patches.py
python -c "from ultralytics.nn.modules import SqueezeExcitation, BiFPN_Concat2; print('OK')"
```

This also installs the GA-tuned hyperparameters (`configs/default_tuned.yaml`) as the Ultralytics defaults, and the FireNet-D5 architecture as the `yolov5.yaml` model definition. Originals are backed up with a `.orig` suffix.

### 3. (Optional) Re-run the GA hyperparameter search

This is stage 1 of the pipeline and is computationally expensive (10 generations × 100 epochs). Its output is already provided in `configs/default_tuned.yaml`, so it can be skipped when reproducing the final model. To re-run it:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt
cp ../configs/firenet_d5_model.yaml models/custom_yolov5s.yaml
bash ../scripts/evolve_hyperparams.sh /path/to/ISFire/data.yaml
```

Search settings: **10 generations**, **100 epochs** per candidate, image size 640, batch 8, trained from scratch (`--weights ''`), auto-anchor disabled, fitness = weighted combination of mAP@0.5 and mAP@0.5:0.95 (YOLOv5 default).

### 4. Train the final model

```bash
bash scripts/train.sh
```

This runs `yolo task=detect mode=train model=yolov5s.yaml data=configs/data.yaml pretrained=yolov5s.pt epochs=150 imgsz=640 workers=4 seed=42`. Because of the applied patches, `yolov5s.yaml` resolves to the FireNet-D5 architecture and the GA-tuned hyperparameters are used automatically. The fixed seed (42) with `deterministic: True` makes the run reproducible up to CUDA nondeterminism.

### 5. Evaluate

```bash
bash scripts/validate.sh weights/best.pt
```

### 6. Inference

```bash
bash scripts/predict.sh weights/best.pt path/to/image_or_video
```

### 7. Grad-CAM visualizations

Grad-CAM figures were generated with the [pourmand1376/yolov5](https://github.com/pourmand1376/yolov5) fork (branch `add_gradcam`):

```bash
git clone https://github.com/pourmand1376/yolov5
cd yolov5 && git checkout add_gradcam
pip install -r requirements.txt && pip install grad-cam==1.4.6
python /path/to/scripts/gradcam.py --weights /path/to/weights/best.pt --source image.jpg
```

## Third-party code

- [Ultralytics](https://github.com/ultralytics/ultralytics) v8.3.33 (AGPL-3.0) — the files in `patched_ultralytics/` are modified copies of Ultralytics source files; modifications are limited to adding/registering the `SqueezeExcitation`, `BiFPN_Concat2`, and `BiFPN_Concat3` modules.
- [YOLOv5](https://github.com/ultralytics/yolov5) (AGPL-3.0) — used for genetic-algorithm hyperparameter evolution.
- [pourmand1376/yolov5](https://github.com/pourmand1376/yolov5), branch `add_gradcam` (AGPL-3.0) — used for Grad-CAM visualizations.
- [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0) — attention-layer helper functions.

## License

Because this repository contains modified Ultralytics source files, it is distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**, the same license as the upstream project. See `LICENSE`.

## Citation

If you use this code or the ISFire dataset, please cite the manuscript:

> FireNet-D5: A Genetic Algorithm-Optimized Architecture with BiFPN and SE Attention for Multi-Class Indoor Fire and Smoke Detection. PLOS ONE.
