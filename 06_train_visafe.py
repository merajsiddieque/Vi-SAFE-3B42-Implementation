#!/usr/bin/env python3

from pathlib import Path
import os
import sys
import csv
import time
import random
import argparse
import logging

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_CANDIDATES = [
    Path("/content/Vi-SAFE-3B42-Implementation"),
    Path("/content/Vi-SAFE-3B42"),
]

PROJECT_ROOT = next(
    (p for p in PROJECT_CANDIDATES if (p / "app" / "main.py").is_file()),
    None
)

if PROJECT_ROOT is None:
    raise FileNotFoundError("Project directory not found.")

sys.path.insert(0, str(PROJECT_ROOT))

DATA_CANDIDATES = [
    Path("/content/RWF-2000"),
    PROJECT_ROOT / "RWF-2000",
    Path("/content/drive/MyDrive/Vi-SAFE-3B42/RWF-2000"),
]

DATA_ROOT = next(
    (
        p for p in DATA_CANDIDATES
        if (p / "train").is_dir() and (p / "val").is_dir()
    ),
    None
)

if DATA_ROOT is None:
    raise FileNotFoundError("RWF-2000 dataset not found.")

MMACTION_ROOT = Path("/content/mmaction2_full_v1_2_0")

if not MMACTION_ROOT.is_dir():
    MMACTION_ROOT = PROJECT_ROOT / "mmaction2"

if not MMACTION_ROOT.is_dir():
    raise FileNotFoundError("MMAction2 source not found.")

sys.path.insert(0, str(MMACTION_ROOT))

CONFIG_PATH = (
    MMACTION_ROOT
    / "configs/recognition/tsn/"
    "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"
)

YOLO_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

DRIVE_ROOT = Path("/content/drive/MyDrive")

if not DRIVE_ROOT.is_dir():
    raise RuntimeError("Google Drive is not mounted.")

OUTPUT_DIR = (
    DRIVE_ROOT
    / "Vi-SAFE-3B42"
    / "outputs"
    / "full_training"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. TRAINING CONFIGURATION
# ============================================================

EPOCHS = 100
BATCH_SIZE = 2
NUM_SEGMENTS = 8
NUM_WORKERS = 2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

EARLY_STOP_PATIENCE = 20

SAVE_EVERY = 5

SEED = 42
DEVICE = "cuda"

RUN_NAME = "visafe-rwf2000-baseline"


# ============================================================
# 3. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--no-resume",
    action="store_true",
    help="Start a fresh training run instead of restoring latest checkpoint."
)

parser.add_argument(
    "--epochs",
    type=int,
    default=EPOCHS
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=BATCH_SIZE
)

parser.add_argument(
    "--num_segments",
    type=int,
    default=NUM_SEGMENTS
)

parser.add_argument(
    "--lr",
    type=float,
    default=LEARNING_RATE
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=NUM_WORKERS
)

parser.add_argument(
    "--seed",
    type=int,
    default=SEED
)

args = parser.parse_args()


# ============================================================
# 4. LOGGING
# ============================================================

LOG_PATH = OUTPUT_DIR / "training.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger("ViSAFE")


# ============================================================
# 5. ENVIRONMENT VALIDATION
# ============================================================

assert CONFIG_PATH.is_file(), f"Missing config: {CONFIG_PATH}"
assert YOLO_WEIGHTS.is_file(), f"Missing YOLO weights: {YOLO_WEIGHTS}"

for filename in [
    "encoder_yolo.py",
    "yolo_tsn.py",
    "dataset_rwf2000.py"
]:
    assert (PROJECT_ROOT / "app" / filename).is_file()

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Training requires a GPU."
    )

torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0")

logger.info("=" * 65)
logger.info("Vi-SAFE BASELINE TRAINING")
logger.info("=" * 65)

logger.info("Project: %s", PROJECT_ROOT)
logger.info("Dataset: %s", DATA_ROOT)
logger.info("MMAction2: %s", MMACTION_ROOT)
logger.info("GPU: %s", torch.cuda.get_device_name(0))
logger.info("CUDA: %s", torch.version.cuda)
logger.info("PyTorch: %s", torch.__version__)
logger.info(
    "GPU memory: %.2f GB",
    torch.cuda.get_device_properties(0).total_memory / 1e9
)
logger.info("Output: %s", OUTPUT_DIR)

logger.info(
    "Configuration: epochs=%d batch=%d segments=%d",
    args.epochs,
    args.batch_size,
    args.num_segments
)


# ============================================================
# 6. RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)


# ============================================================
# 7. MMENGINE COMPATIBILITY
# ============================================================

from mmengine.registry import Registry, DefaultScope

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
            and getattr(module, "__module__", "").startswith("transformers")
        ):

            existing = self.module_dict.get("Adafactor")

            if (
                existing is not None
                and getattr(existing, "__module__", "").startswith("torch.optim")
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


DefaultScope.get_instance(
    "mmaction",
    scope_name="mmaction"
)

from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)


# ============================================================
# 8. IMPORT PROJECT MODULES
# ============================================================

from app.dataset_rwf2000 import RWF2000Dataset
from app.encoder_yolo import YOLOEncoder


# ============================================================
# 9. BASELINE TSN + YOLO MODEL
# ============================================================

from mmengine import Config
from mmaction.registry import MODELS


class ViSAFEModel(nn.Module):

    def __init__(
        self,
        config_path,
        yolo_encoding_dim=128,
        freeze_tsn_backbone=False
    ):

        super().__init__()

        config = Config.fromfile(str(config_path))

        if "train_cfg" not in config.model:
            config.model["train_cfg"] = config.get(
                "train_cfg", None
            )

        if "test_cfg" not in config.model:
            config.model["test_cfg"] = config.get(
                "test_cfg", None
            )

        # RWF-2000 is a binary classification task.
        config.model.cls_head.num_classes = 2

        self.tsn = MODELS.build(config.model)

        if freeze_tsn_backbone:

            for parameter in self.tsn.backbone.parameters():
                parameter.requires_grad = False

        original_fc = self.tsn.cls_head.fc_cls

        tsn_dim = original_fc.in_features
        output_dim = 2

        self.fusion_fc = nn.Linear(
            tsn_dim + yolo_encoding_dim,
            output_dim
        )

        # Initialize the TSN feature portion from its original classifier.
        with torch.no_grad():

            self.fusion_fc.weight[:, :tsn_dim].copy_(
                original_fc.weight
            )

            self.fusion_fc.bias.copy_(
                original_fc.bias
            )

    def forward(self, frames, yolo_encodings):

        # frames: [B, T, 3, H, W]

        B, T, C, H, W = frames.shape

        frames = frames.reshape(
            B * T,
            C,
            H,
            W
        )

        features = self.tsn.backbone(frames)

        features = self.tsn.cls_head.avg_pool(features)

        features = features.flatten(1)

        features = features.reshape(B, T, -1)

        # Original baseline: uniform temporal mean pooling.
        tsn_features = features.mean(dim=1)

        # Uniform temporal aggregation of YOLO features.
        yolo_features = yolo_encodings.mean(dim=1)

        fused_features = torch.cat(
            [tsn_features, yolo_features],
            dim=1
        )

        logits = self.fusion_fc(fused_features)

        return logits


# ============================================================
# 10. DATASET
# ============================================================

train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    T.ToTensor()
])

val_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])

train_dataset = RWF2000Dataset(
    root_dir=str(DATA_ROOT),
    mode="train",
    num_segments=args.num_segments,
    transform=train_transform
)

val_dataset = RWF2000Dataset(
    root_dir=str(DATA_ROOT),
    mode="val",
    num_segments=args.num_segments,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(args.num_workers > 0)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(args.num_workers > 0)
)

logger.info("Training videos: %d", len(train_dataset))
logger.info("Validation videos: %d", len(val_dataset))


# ============================================================
# 11. INITIALIZE YOLO AND TSN
# ============================================================

yolo_encoder = YOLOEncoder(
    yolo_weights_path=str(YOLO_WEIGHTS),
    device=device,
    freeze_yolo=False
).to(device)

model = ViSAFEModel(
    config_path=CONFIG_PATH,
    yolo_encoding_dim=128,
    freeze_tsn_backbone=False
).to(device)

logger.info("TSN device: %s", next(model.parameters()).device)
logger.info("YOLO device: %s", next(yolo_encoder.parameters()).device)

logger.info("Model initialized with 2 output classes.")


# ============================================================
# 12. OPTIMIZER AND SCHEDULER
# ============================================================

parameters = [
    p for p in (
        list(model.parameters()) +
        list(yolo_encoder.parameters())
    )
    if p.requires_grad
]

optimizer = optim.Adam(
    parameters,
    lr=args.lr,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# 13. CHECKPOINT PATHS
# ============================================================

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pt"

BEST_MODEL = OUTPUT_DIR / "best_model.pt"

METRICS_PATH = OUTPUT_DIR / "metrics.csv"


# ============================================================
# 14. ATOMIC CHECKPOINT SAVING
# ============================================================

def atomic_save(checkpoint, destination):

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    torch.save(checkpoint, temporary_path)

    os.replace(
        temporary_path,
        destination
    )


def make_checkpoint(epoch, best_acc, patience_counter):

    return {
        "epoch": epoch,
        "best_val_acc": best_acc,
        "patience_counter": patience_counter,

        "model_state_dict": model.state_dict(),
        "yolo_encoder_state_dict": yolo_encoder.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),

        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),

        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_segments": args.num_segments,
            "lr": args.lr,
            "weight_decay": WEIGHT_DECAY,
            "seed": args.seed
        }
    }


# ============================================================
# 15. RESUME TRAINING
# ============================================================

start_epoch = 1
best_val_acc = -1.0
patience_counter = 0

if LATEST_CHECKPOINT.exists() and not args.no_resume:

    logger.info("=" * 65)
    logger.info("RESUMING FROM CHECKPOINT")
    logger.info("=" * 65)

    # Load checkpoint on CPU.
    # This prevents RNG state tensors from being moved to CUDA.
    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    saved_config = checkpoint.get("config", {})

    if (
        saved_config.get("batch_size") != args.batch_size
        or saved_config.get("num_segments") != args.num_segments
        or saved_config.get("lr") != args.lr
    ):
        raise ValueError(
            "Training configuration differs from the saved checkpoint. "
            "Use the original batch size, segments and learning rate to resume."
        )

    # Restore model weights.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    yolo_encoder.load_state_dict(
        checkpoint["yolo_encoder_state_dict"]
    )

    # Restore optimizer and scheduler.
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    # Restore training progress.
    start_epoch = checkpoint["epoch"] + 1

    best_val_acc = checkpoint["best_val_acc"]

    patience_counter = checkpoint["patience_counter"]

    # Restore Python and NumPy random states.
    random.setstate(
        checkpoint["python_rng_state"]
    )

    np.random.set_state(
        checkpoint["numpy_rng_state"]
    )

    # Restore CPU PyTorch RNG state.
    torch.set_rng_state(
        checkpoint["torch_rng_state"].cpu()
    )

    # Restore CUDA RNG states.
    torch.cuda.set_rng_state_all(
        [state.cpu() for state in checkpoint["cuda_rng_state"]]
    )

    logger.info(
        "Restored epoch %d. Next epoch: %d",
        checkpoint["epoch"],
        start_epoch
    )

    logger.info(
        "Best validation accuracy: %.4f",
        best_val_acc
    )

    logger.info("Checkpoint restored successfully.")

else:

    logger.info("Starting a new training run.")


# ============================================================
# 16. YOLO FEATURE EXTRACTION
# ============================================================

def extract_yolo_features(frames):

    encoded_batch = []

    for index in range(frames.shape[0]):

        encoded = yolo_encoder(frames[index])

        encoded_batch.append(encoded)

    return torch.stack(
        encoded_batch,
        dim=0
    )


# ============================================================
# 17. TRAINING FUNCTION
# ============================================================

def train_one_epoch(epoch):

    model.train()
    yolo_encoder.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{args.epochs} [Train]",
        bar_format="{desc}: {percentage:3.0f}%",
        leave=True,
        dynamic_ncols=True
    )

    for frames, labels in progress:

        frames = frames.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        ).long()

        yolo_features = extract_yolo_features(frames)

        logits = model(
            frames,
            yolo_features
        )

        loss = criterion(
            logits,
            labels
        )

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        progress.set_postfix_str(
            f"loss={total_loss / total_samples:.4f} "
            f"acc={total_correct / total_samples:.4f}"
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ============================================================
# 18. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate(epoch):

    model.eval()
    yolo_encoder.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        val_loader,
        desc=f"Epoch {epoch}/{args.epochs} [Val]",
        bar_format="{desc}: {percentage:3.0f}%",
        leave=True,
        dynamic_ncols=True
    )

    for frames, labels in progress:

        frames = frames.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        ).long()

        yolo_features = extract_yolo_features(frames)

        logits = model(
            frames,
            yolo_features
        )

        loss = criterion(
            logits,
            labels
        )

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        progress.set_postfix_str(
            f"loss={total_loss / total_samples:.4f} "
            f"acc={total_correct / total_samples:.4f}"
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ============================================================
# 19. METRICS CSV
# ============================================================

if not METRICS_PATH.exists() or args.no_resume:

    with METRICS_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
            "time_seconds"
        ])


# ============================================================
# 20. MAIN TRAINING LOOP
# ============================================================

logger.info("=" * 65)
logger.info("STARTING TRAINING")
logger.info("Starting epoch: %d", start_epoch)
logger.info("Total epochs: %d", args.epochs)
logger.info("=" * 65)

for epoch in range(start_epoch, args.epochs + 1):

    epoch_start = time.time()

    train_loss, train_acc = train_one_epoch(epoch)

    val_loss, val_acc = validate(epoch)

    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]

    epoch_time = time.time() - epoch_start

    logger.info(
        "[Epoch %d/%d] "
        "Train Loss: %.4f | Train Acc: %.4f | "
        "Val Loss: %.4f | Val Acc: %.4f | "
        "LR: %.7f | Time: %.2f min",
        epoch,
        args.epochs,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        current_lr,
        epoch_time / 60
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    with METRICS_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            epoch_time
        ])

    # --------------------------------------------------------
    # UPDATE BEST MODEL
    # --------------------------------------------------------

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        patience_counter = 0

        best_checkpoint = make_checkpoint(
            epoch,
            best_val_acc,
            patience_counter
        )

        atomic_save(
            best_checkpoint,
            BEST_MODEL
        )

        logger.info(
            "BEST MODEL SAVED | Epoch: %d | Val Acc: %.4f",
            epoch,
            best_val_acc
        )

    else:

        patience_counter += 1

    # --------------------------------------------------------
    # SAVE LATEST CHECKPOINT EVERY EPOCH
    # --------------------------------------------------------

    checkpoint = make_checkpoint(
        epoch,
        best_val_acc,
        patience_counter
    )

    atomic_save(
        checkpoint,
        LATEST_CHECKPOINT
    )

    logger.info(
        "Latest checkpoint saved after epoch %d.",
        epoch
    )

    # --------------------------------------------------------
    # PERIODIC CHECKPOINT EVERY 5 EPOCHS
    # --------------------------------------------------------

    if epoch % SAVE_EVERY == 0:

        periodic_path = (
            CHECKPOINT_DIR
            / f"checkpoint_epoch_{epoch:03d}.pt"
        )

        atomic_save(
            checkpoint,
            periodic_path
        )

        logger.info(
            "Periodic checkpoint saved: %s",
            periodic_path.name
        )

    # --------------------------------------------------------
    # GPU MEMORY REPORT
    # --------------------------------------------------------

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

    logger.info(
        "GPU memory allocated: %.2f GB | Reserved: %.2f GB",
        allocated,
        reserved
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        EARLY_STOP_PATIENCE > 0
        and patience_counter >= EARLY_STOP_PATIENCE
    ):

        logger.info(
            "EARLY STOPPING at epoch %d.",
            epoch
        )

        break


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

logger.info("=" * 65)
logger.info("TRAINING FINISHED")
logger.info("=" * 65)

logger.info(
    "Best validation accuracy: %.4f",
    best_val_acc
)

logger.info(
    "Best model: %s",
    BEST_MODEL
)

logger.info(
    "Latest checkpoint: %s",
    LATEST_CHECKPOINT
)

logger.info(
    "Metrics: %s",
    METRICS_PATH
)

logger.info(
    "Checkpoint directory: %s",
    CHECKPOINT_DIR
)

logger.info("=" * 65)