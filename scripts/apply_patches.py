#!/usr/bin/env python3
"""Apply the FireNet-D5 patches to an installed ultralytics==8.3.33 package.

The FireNet-D5 architecture adds two custom modules to Ultralytics:
  * SqueezeExcitation (SE channel-attention block, inserted before SPPF)
  * BiFPN_Concat2 / BiFPN_Concat3 (learnable weighted feature fusion in the neck)

Because these modules must be visible to the Ultralytics model parser, four
files inside the installed package are replaced with patched versions, and the
default hyperparameter file is replaced with the GA-tuned configuration:

  patched_ultralytics/nn/tasks.py              -> ultralytics/nn/tasks.py
  patched_ultralytics/nn/modules/__init__.py   -> ultralytics/nn/modules/__init__.py
  patched_ultralytics/nn/modules/block.py      -> ultralytics/nn/modules/block.py
  patched_ultralytics/nn/modules/conv.py       -> ultralytics/nn/modules/conv.py
  configs/default_tuned.yaml                   -> ultralytics/cfg/default.yaml
  configs/firenet_d5_model.yaml                -> ultralytics/cfg/models/v5/yolov5.yaml

Backups of the original files are stored next to them with a ``.orig`` suffix
the first time this script runs.

Usage:
    pip install ultralytics==8.3.33 timm
    python scripts/apply_patches.py
"""

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_ultralytics() -> Path:
    try:
        import ultralytics
    except ImportError:
        sys.exit("ultralytics is not installed. Run: pip install ultralytics==8.3.33")
    version = getattr(ultralytics, "__version__", "unknown")
    if version != "8.3.33":
        print(f"WARNING: patches were developed against ultralytics==8.3.33, "
              f"but version {version} is installed. Proceeding anyway.")
    return Path(ultralytics.__file__).parent


def main() -> None:
    pkg = find_ultralytics()
    print(f"Patching ultralytics at: {pkg}")

    replacements = {
        REPO_ROOT / "patched_ultralytics/nn/tasks.py": pkg / "nn/tasks.py",
        REPO_ROOT / "patched_ultralytics/nn/modules/__init__.py": pkg / "nn/modules/__init__.py",
        REPO_ROOT / "patched_ultralytics/nn/modules/block.py": pkg / "nn/modules/block.py",
        REPO_ROOT / "patched_ultralytics/nn/modules/conv.py": pkg / "nn/modules/conv.py",
        REPO_ROOT / "configs/default_tuned.yaml": pkg / "cfg/default.yaml",
        REPO_ROOT / "configs/firenet_d5_model.yaml": pkg / "cfg/models/v5/yolov5.yaml",
    }

    for src, dst in replacements.items():
        if not src.exists():
            sys.exit(f"Missing source file: {src}")
        backup = dst.with_suffix(dst.suffix + ".orig")
        if dst.exists() and not backup.exists():
            shutil.copy2(dst, backup)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src.relative_to(REPO_ROOT)} -> {dst}")

    print("\nDone. Verify with:")
    print("  python -c \"from ultralytics.nn.modules import SqueezeExcitation, BiFPN_Concat2; print('OK')\"")


if __name__ == "__main__":
    main()
