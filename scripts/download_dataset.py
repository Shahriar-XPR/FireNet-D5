#!/usr/bin/env python3
"""Download the ISFire (Fire_Smoke_4C) dataset from Hugging Face.

Dataset: https://huggingface.co/datasets/shahriar-5/Fire_Smoke_4C
License: CC-BY-4.0

Usage:
    python scripts/download_dataset.py --output ./ISFire
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = (
    "https://huggingface.co/datasets/shahriar-5/Fire_Smoke_4C/"
    "resolve/main/dataset.zip"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the ISFire dataset")
    parser.add_argument("--output", default="./ISFire",
                        help="Directory to extract the dataset into")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep dataset.zip after extraction")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "dataset.zip"

    print(f"Downloading {DATASET_URL}\n  -> {zip_path} (~267 MB)")
    with urllib.request.urlopen(DATASET_URL) as resp, open(zip_path, "wb") as f:
        shutil.copyfileobj(resp, f)

    print(f"Extracting to {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    if not args.keep_zip:
        zip_path.unlink()

    print("Done. Update the train/val/test paths in configs/data.yaml to the")
    print(f"extracted image folders under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
