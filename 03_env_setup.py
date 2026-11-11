
# ============================================================
# STAGE 1: PREPARE Vi-SAFE ENVIRONMENT
# ============================================================
# Installs and verifies:
#   1. PyTorch compatibility
#   2. MMEngine
#   3. MMCV
#   4. MMAction2 prerequisites
#   5. Ultralytics
#   6. Project and dataset paths
#
# Run:
#   !python stage1_environment.py
# ============================================================

import sys
import subprocess
import importlib
from pathlib import Path

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path("/content/Vi-SAFE-3B42-Implementation")
DATASET_ROOT = Path("/content/RWF-2000")
YOLO_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

print("=" * 60)
print("Vi-SAFE | ENVIRONMENT SETUP")
print("=" * 60)

# ------------------------------------------------------------
# 2. VERIFY PROJECT FILES
# ------------------------------------------------------------

required_paths = {
    "Project source": PROJECT_ROOT,
    "RWF-2000 dataset": DATASET_ROOT,
    "YOLOv8 weights": YOLO_WEIGHTS,
}

for name, path in required_paths.items():
    print(f"{name}: {'FOUND' if path.exists() else 'MISSING'}")
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

# ------------------------------------------------------------
# 3. CHECK EXISTING PYTORCH
# ------------------------------------------------------------

import torch

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    raise RuntimeError(
        "GPU not detected. Enable GPU in Colab: "
        "Runtime > Change runtime type > T4 GPU."
    )

# ------------------------------------------------------------
# 4. INSTALL COMPATIBLE DEPENDENCIES
# ------------------------------------------------------------

packages = [
    "mmengine==0.10.5",
    "mmcv-lite==2.1.0",
    "ultralytics",
    "transformers==4.47.1",
    "tokenizers>=0.21,<0.22",
    "opencv-python-headless",
    "decord",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "tqdm",
]

print("\nInstalling dependencies...")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    *packages
])

# ------------------------------------------------------------
# 5. VERIFY INSTALLATIONS
# ------------------------------------------------------------

modules = [
    "mmengine",
    "mmcv",
    "ultralytics",
    "transformers",
    "cv2",
    "decord",
    "sklearn",
    "pandas",
    "matplotlib",
]

print("\nDependency verification:")

for module_name in modules:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "Installed")
        print(f"[OK] {module_name}: {version}")
    except Exception as error:
        raise RuntimeError(
            f"Failed to import {module_name}: {error}"
        ) from error

# ------------------------------------------------------------
# 6. VERIFY PROJECT DATASET STRUCTURE
# ------------------------------------------------------------

print("\nDataset verification:")

expected = {
    "train/Fight": 800,
    "train/NonFight": 800,
    "val/Fight": 200,
    "val/NonFight": 200,
}

video_extensions = {".avi", ".mp4", ".mov", ".mkv"}

for relative_path, expected_count in expected.items():
    folder = DATASET_ROOT / relative_path

    if not folder.exists():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")

    videos = [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in video_extensions
    ]

    print(f"{relative_path}: {len(videos)} / {expected_count}")

    if len(videos) != expected_count:
        raise RuntimeError(
            f"Unexpected video count in {relative_path}"
        )

# ------------------------------------------------------------
# 7. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Project:", PROJECT_ROOT)
print("Dataset:", DATASET_ROOT)
print("YOLO weights:", YOLO_WEIGHTS)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("Total videos: 2000")
print("=" * 60)
