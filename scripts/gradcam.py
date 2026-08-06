#!/usr/bin/env python3
"""Grad-CAM visualization for FireNet-D5 predictions (post-hoc XAI).

Uses the YOLOv5 Grad-CAM fork by pourmand1376, as in the manuscript.

Setup:
    git clone https://github.com/pourmand1376/yolov5
    cd yolov5
    git checkout add_gradcam
    pip install -r requirements.txt
    pip install grad-cam==1.4.6

Usage (run from inside the cloned yolov5 directory):
    python /path/to/scripts/gradcam.py \
        --weights /path/to/weights/best.pt \
        --source /path/to/image.jpg \
        --output gradcam_output.jpg
"""

import argparse

CLASS_NAMES = ["black smoke", "blue fire", "white-gray smoke", "yellow-orange fire"]


def main() -> None:
    parser = argparse.ArgumentParser(description="FireNet-D5 Grad-CAM")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--output", default="gradcam_output.jpg", help="Output image path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--method", default="GradCAM",
                        help="CAM method, e.g. GradCAM, GradCAMPlusPlus, EigenCAM")
    parser.add_argument("--layer", type=int, default=-2,
                        help="Model layer index used for the CAM")
    args = parser.parse_args()

    # Imported here so the script fails with a clear message when not run
    # from inside the pourmand1376/yolov5 (add_gradcam branch) repository.
    try:
        from explainer.explainer import run
    except ImportError:
        raise SystemExit(
            "Could not import 'explainer'. Run this script from inside the "
            "pourmand1376/yolov5 repository (branch: add_gradcam)."
        )
    from PIL import Image

    image = run(
        weights=args.weights,
        source=args.source,
        imgsz=args.imgsz,
        method=args.method,
        layer=args.layer,
        class_names=CLASS_NAMES,
    )
    Image.fromarray(image).save(args.output)
    print(f"Grad-CAM visualization saved to {args.output}")


if __name__ == "__main__":
    main()
