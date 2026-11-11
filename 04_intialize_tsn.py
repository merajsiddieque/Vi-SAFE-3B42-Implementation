
# ============================================================
# STAGE 2: INITIALIZE ORIGINAL Vi-SAFE TSN
# File: 04_intialize_tsn.py
#
# Architecture:
#   TSN Recognizer2D + ResNet-50 backbone
#
# Pretraining:
#   ImageNet-pretrained ResNet-50
#
# No temporal attention is added.
# ============================================================

import sys
import re
import logging
import urllib.request
from pathlib import Path

import torch


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/content/Vi-SAFE-3B42-Implementation")
MMACTION_ROOT = Path("/content/mmaction2_full_v1_2_0")

CONFIG_PATH = (
    MMACTION_ROOT
    / "configs"
    / "recognition"
    / "tsn"
    / "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"
)

DRIVE_ROOT = Path("/content/drive/MyDrive/Vi-SAFE-3B42")

OUTPUT_ROOT = DRIVE_ROOT / "outputs"
LOG_DIR = OUTPUT_ROOT / "logs"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

LOG_FILE = LOG_DIR / "stage2_tsn_initialization.log"

RESNET_CHECKPOINT = (
    CHECKPOINT_DIR / "resnet50-11ad3fa6.pth"
)

TSN_CHECKPOINT = (
    CHECKPOINT_DIR / "tsn_initialized.pth"
)

RESNET_URL = (
    "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
)


# ============================================================
# 2. VERIFY DIRECTORIES
# ============================================================

if not PROJECT_ROOT.is_dir():
    raise FileNotFoundError(
        f"Project directory not found: {PROJECT_ROOT}"
    )

if not MMACTION_ROOT.is_dir():
    raise FileNotFoundError(
        f"MMAction2 directory not found: {MMACTION_ROOT}"
    )

if not CONFIG_PATH.is_file():
    raise FileNotFoundError(
        f"TSN configuration not found: {CONFIG_PATH}"
    )

if not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError(
        "Google Drive is not mounted. "
        "Mount it in a separate Colab notebook cell."
    )

LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)

logger = logging.getLogger("ViSAFE-Stage2")

logger.info("=" * 65)
logger.info("STAGE 2: TSN INITIALIZATION")
logger.info("=" * 65)


# ============================================================
# 4. DEVICE INFORMATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Device: {DEVICE}")

if torch.cuda.is_available():
    logger.info(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# 5. APPLY ADAFACTOR REGISTRY COMPATIBILITY PATCH
# ============================================================

# IMPORTANT:
# This must happen before importing mmaction.
#
# MMEngine can register torch.optim.Adafactor before
# Transformers attempts to register its own Adafactor.
# This causes a duplicate registry-name error.

sys.path.insert(0, str(MMACTION_ROOT))

from mmengine.registry import Registry

if not getattr(Registry, "_visafe_adafactor_patch", False):

    original_register_module = Registry._register_module

    def compatible_register_module(
        self,
        module,
        module_name=None,
        force=False
    ):

        if (
            module_name == "Adafactor"
            and module.__module__.startswith("transformers")
        ):

            existing = self.module_dict.get("Adafactor")

            if (
                existing is not None
                and existing.__module__.startswith("torch.optim")
            ):
                self._module_dict.pop("Adafactor")

        return original_register_module(
            self,
            module,
            module_name=module_name,
            force=force
        )

    Registry._register_module = compatible_register_module
    Registry._visafe_adafactor_patch = True

logger.info(
    "Adafactor registry compatibility patch applied."
)


# ============================================================
# 6. IMPORT MMACTION2
# ============================================================

try:

    from mmengine.config import Config

    from mmaction.utils import register_all_modules
    from mmaction.registry import MODELS

    register_all_modules(init_default_scope=True)

    logger.info(
        "MMAction2 modules imported successfully."
    )

except Exception:

    logger.exception("MMAction2 import failed.")
    raise


# ============================================================
# 7. LOAD ORIGINAL TSN CONFIGURATION
# ============================================================

logger.info("Loading TSN configuration...")

cfg = Config.fromfile(str(CONFIG_PATH))

logger.info(f"Model type: {cfg.model.type}")
logger.info(
    f"Backbone type: {cfg.model.backbone.type}"
)
logger.info(
    f"Backbone depth: {cfg.model.backbone.depth}"
)

if cfg.model.type != "Recognizer2D":
    raise ValueError("Expected Recognizer2D model.")

if cfg.model.backbone.type != "ResNet":
    raise ValueError("Expected ResNet backbone.")

if cfg.model.backbone.depth != 50:
    raise ValueError("Expected ResNet-50 backbone.")


# ============================================================
# 8. DISABLE AUTOMATIC PRETRAINED LOADING
# ============================================================

# Load the weights manually after converting torchvision
# parameter names to the MMAction2 ResNet naming convention.

cfg.model.backbone.pretrained = None

# Preserve the original TSN architecture and configuration.
# No temporal attention or additional modules are introduced.


# ============================================================
# 9. DOWNLOAD RESNET-50 PRETRAINED CHECKPOINT
# ============================================================

def download_resnet_checkpoint():

    if RESNET_CHECKPOINT.exists():
        logger.info(
            f"Using existing checkpoint: {RESNET_CHECKPOINT}"
        )
        return

    logger.info(
        "Downloading ImageNet-pretrained ResNet-50..."
    )

    temporary_path = RESNET_CHECKPOINT.with_suffix(".tmp")

    try:

        urllib.request.urlretrieve(
            RESNET_URL,
            temporary_path
        )

        temporary_path.replace(RESNET_CHECKPOINT)

        logger.info(
            "ResNet-50 checkpoint downloaded successfully."
        )

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        logger.exception(
            "ResNet-50 checkpoint download failed."
        )
        raise


download_resnet_checkpoint()


# ============================================================
# 10. BUILD TSN MODEL
# ============================================================

logger.info("Building original TSN model...")

try:

    tsn_model = MODELS.build(cfg.model)

    tsn_model.init_weights()

    logger.info(
        "Original TSN model constructed successfully."
    )

except Exception:

    logger.exception("TSN model construction failed.")
    raise


# ============================================================
# 11. LOAD TORCHVISION CHECKPOINT
# ============================================================

logger.info("Loading ResNet-50 checkpoint...")

try:

    try:
        source_state = torch.load(
            RESNET_CHECKPOINT,
            map_location="cpu",
            weights_only=True
        )

    except TypeError:
        source_state = torch.load(
            RESNET_CHECKPOINT,
            map_location="cpu"
        )

    if "state_dict" in source_state:
        source_state = source_state["state_dict"]

    logger.info(
        f"Source tensors: {len(source_state)}"
    )

except Exception:

    logger.exception(
        "Unable to load ResNet-50 checkpoint."
    )
    raise


# ============================================================
# 12. CONVERT TORCHVISION KEYS TO MMCV KEYS
# ============================================================

def convert_resnet_key(key):

    # Initial convolution
    if key == "conv1.weight":
        return "conv1.conv.weight"

    # Initial batch normalization
    if key.startswith("bn1."):
        return key.replace(
            "bn1.",
            "conv1.bn.",
            1
        )

    # ResNet bottleneck layers
    match = re.match(
        r"^(layer\d+\.\d+)\.(conv[123]|bn[123])\.(.+)$",
        key
    )

    if match:

        block, layer_name, parameter = match.groups()
        number = layer_name[-1]

        if layer_name.startswith("conv"):

            return (
                f"{block}.conv{number}.conv.{parameter}"
            )

        return (
            f"{block}.conv{number}.bn.{parameter}"
        )

    # Downsample layers
    match = re.match(
        r"^(layer\d+\.\d+)\.downsample\.(0|1)\.(.+)$",
        key
    )

    if match:

        block, index, parameter = match.groups()

        if index == "0":

            return (
                f"{block}.downsample.conv.{parameter}"
            )

        return (
            f"{block}.downsample.bn.{parameter}"
        )

    return key


logger.info(
    "Converting torchvision ResNet parameter names..."
)

converted_state = {}

for key, value in source_state.items():

    # Ignore ImageNet classification head.
    if key.startswith("fc."):
        continue

    converted_key = convert_resnet_key(key)

    converted_state[converted_key] = value


logger.info(
    f"Converted tensors: {len(converted_state)}"
)


# ============================================================
# 13. VALIDATE ALL BACKBONE PARAMETERS
# ============================================================

backbone = tsn_model.backbone

model_state = backbone.state_dict()

model_keys = set(model_state.keys())
checkpoint_keys = set(converted_state.keys())

missing_keys = sorted(
    model_keys - checkpoint_keys
)

unexpected_keys = sorted(
    checkpoint_keys - model_keys
)

shape_mismatches = []

for key in model_keys.intersection(checkpoint_keys):

    expected_shape = tuple(model_state[key].shape)
    actual_shape = tuple(converted_state[key].shape)

    if expected_shape != actual_shape:

        shape_mismatches.append(
            (
                key,
                expected_shape,
                actual_shape
            )
        )


logger.info(
    f"Model tensors: {len(model_keys)}"
)

logger.info(
    f"Missing keys: {len(missing_keys)}"
)

logger.info(
    f"Unexpected keys: {len(unexpected_keys)}"
)

logger.info(
    f"Shape mismatches: {len(shape_mismatches)}"
)

if missing_keys:
    logger.error(
        f"Missing keys: {missing_keys[:20]}"
    )

if unexpected_keys:
    logger.error(
        f"Unexpected keys: {unexpected_keys[:20]}"
    )

if shape_mismatches:
    logger.error(
        f"Shape mismatches: {shape_mismatches[:20]}"
    )

if missing_keys or unexpected_keys or shape_mismatches:

    raise RuntimeError(
        "Checkpoint validation failed. "
        "Pretrained weights were not loaded."
    )


# ============================================================
# 14. LOAD WEIGHTS STRICTLY
# ============================================================

logger.info(
    "Loading pretrained weights with strict=True..."
)

backbone.load_state_dict(
    converted_state,
    strict=True
)

logger.info(
    "All ResNet-50 backbone weights loaded successfully."
)


# ============================================================
# 15. MOVE MODEL TO DEVICE
# ============================================================

tsn_model = tsn_model.to(DEVICE)
tsn_model.eval()

logger.info(
    f"TSN model moved to {DEVICE}."
)


# ============================================================
# 16. SAVE INITIALIZED TSN CHECKPOINT
# ============================================================

logger.info(
    "Saving initialized TSN checkpoint to Google Drive..."
)

checkpoint_data = {

    "state_dict": {
        key: value.detach().cpu()
        for key, value in tsn_model.state_dict().items()
    },

    "meta": {

        "stage": 2,
        "architecture": "Original Vi-SAFE TSN",
        "recognizer": "Recognizer2D",
        "backbone": "ResNet-50",

        "pretrained_weights": (
            "Torchvision ImageNet-pretrained ResNet-50"
        ),

        "config_path": str(CONFIG_PATH),
        "device_used": DEVICE,
    }
}

temporary_checkpoint = TSN_CHECKPOINT.with_suffix(".tmp")

torch.save(
    checkpoint_data,
    temporary_checkpoint
)

temporary_checkpoint.replace(TSN_CHECKPOINT)

logger.info(
    f"Checkpoint saved: {TSN_CHECKPOINT}"
)


# ============================================================
# 17. FINAL REPORT
# ============================================================

logger.info("=" * 65)
logger.info("STAGE 2 COMPLETED")
logger.info("=" * 65)

print("\n" + "=" * 65)
print("STAGE 2: TSN INITIALIZATION REPORT")
print("=" * 65)

print("Architecture       : Original TSN")
print("Recognizer         : Recognizer2D")
print("Backbone           : ResNet-50")
print(f"Device             : {DEVICE}")

print("\nCheckpoint validation:")
print(f"Missing keys       : {len(missing_keys)}")
print(f"Unexpected keys    : {len(unexpected_keys)}")
print(f"Shape mismatches   : {len(shape_mismatches)}")

print("\nSaved files:")
print(f"TSN checkpoint     : {TSN_CHECKPOINT}")
print(f"ResNet checkpoint  : {RESNET_CHECKPOINT}")
print(f"Log file           : {LOG_FILE}")

print("\nSTATUS: STAGE 2 COMPLETED SUCCESSFULLY")
print("=" * 65)
