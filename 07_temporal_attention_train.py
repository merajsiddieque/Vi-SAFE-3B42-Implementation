
#!/usr/bin/env python3
"""
07_temporal_attention_train.py

Vi-SAFE-3B42: Temporal Attention Experiment

Architecture:
    Video frames
        |
        +---- TSN ResNet50 --------> 2048-D feature per segment
        |
        +---- YOLOEncoder ---------> 128-D feature per segment
                                      |
                              Concatenate (2176-D)
                                      |
                              Temporal Attention
                                      |
                               Weighted Sum
                                      |
                              Binary Classifier
                                      |
                               Fight / NonFight

Dataset labels:
    Fight    = 1
    NonFight = 0

This script does NOT modify any existing project files.
"""

from pathlib import Path
import argparse
import csv
import logging
import os
import random
import sys
import time

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
    (
        p for p in PROJECT_CANDIDATES
        if (p / "app" / "encoder_yolo.py").is_file()
    ),
    None,
)

if PROJECT_ROOT is None:
    raise FileNotFoundError("Vi-SAFE project directory not found.")

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
    None,
)

if DATA_ROOT is None:
    raise FileNotFoundError("RWF-2000 dataset not found.")

MMACTION_ROOT = Path("/content/mmaction2_full_v1_2_0")

if not MMACTION_ROOT.is_dir():
    MMACTION_ROOT = PROJECT_ROOT / "mmaction2"

if not MMACTION_ROOT.is_dir():
    raise FileNotFoundError("MMAction2 source directory not found.")

sys.path.insert(0, str(MMACTION_ROOT))

CONFIG_PATH = (
    MMACTION_ROOT
    / "configs/recognition/tsn/"
    / "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"
)

YOLO_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

DRIVE_ROOT = Path("/content/drive/MyDrive")

if not DRIVE_ROOT.is_dir():
    raise RuntimeError("Google Drive is not mounted.")

# Separate directory: baseline results remain untouched.
OUTPUT_DIR = (
    DRIVE_ROOT
    / "Vi-SAFE-3B42"
    / "outputs"
    / "temporal_attention"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. TRAINING CONFIGURATION
# ============================================================

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_SEGMENTS = 8
DEFAULT_NUM_WORKERS = 2

DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5

DEFAULT_EARLY_STOP_PATIENCE = 20
SAVE_EVERY = 5

YOLO_FEATURE_DIM = 128
ATTENTION_HIDDEN_DIM = 256


# ============================================================
# 3. COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Vi-SAFE Temporal Attention Training"
)

parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
parser.add_argument("--num_segments", type=int, default=DEFAULT_NUM_SEGMENTS)
parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)

parser.add_argument("--lr", type=float, default=DEFAULT_LR)
parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)

parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=DEFAULT_EARLY_STOP_PATIENCE,
)

parser.add_argument(
    "--freeze_yolo",
    action="store_true",
)

parser.add_argument(
    "--freeze_tsn_backbone",
    action="store_true",
)

parser.add_argument(
    "--no-resume",
    action="store_true",
    help="Start a fresh run and clear only temporal-attention checkpoints.",
)

args = parser.parse_args()

if args.epochs < 1:
    raise ValueError("Epochs must be greater than zero.")

if args.batch_size < 1 or args.num_segments < 1:
    raise ValueError("Batch size and number of segments must be positive.")

if args.num_workers < 0:
    raise ValueError("num_workers cannot be negative.")


# ============================================================
# 4. LOGGING
# ============================================================

LOG_PATH = OUTPUT_DIR / "training.log"
METRICS_PATH = OUTPUT_DIR / "metrics.csv"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pt"
BEST_MODEL = CHECKPOINT_DIR / "best_model.pt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)

logger = logging.getLogger("ViSAFE-TemporalAttention")


# ============================================================
# 5. ENVIRONMENT AND GPU VALIDATION
# ============================================================

if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Missing TSN config: {CONFIG_PATH}")

if not YOLO_WEIGHTS.is_file():
    raise FileNotFoundError(f"Missing YOLO weights: {YOLO_WEIGHTS}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Enable GPU in Google Colab."
    )

device = torch.device("cuda:0")

torch.backends.cudnn.benchmark = True

logger.info("=" * 70)
logger.info("Vi-SAFE TEMPORAL ATTENTION EXPERIMENT")
logger.info("=" * 70)

logger.info("Project: %s", PROJECT_ROOT)
logger.info("Dataset: %s", DATA_ROOT)
logger.info("MMAction2: %s", MMACTION_ROOT)

logger.info("GPU: %s", torch.cuda.get_device_name(0))
logger.info("CUDA: %s", torch.version.cuda)
logger.info("PyTorch: %s", torch.__version__)

logger.info(
    "GPU memory: %.2f GB",
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
)

logger.info("Output directory: %s", OUTPUT_DIR)

logger.info(
    "Configuration: epochs=%d, batch=%d, segments=%d, lr=%g",
    args.epochs,
    args.batch_size,
    args.num_segments,
    args.lr,
)

logger.info("Labels: Fight=1, NonFight=0")


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
        force=False,
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
            force=force,
        )

    Registry._register_module = compatible_register_module
    Registry._visafe_adafactor_patch = True


DefaultScope.get_instance(
    "mmaction",
    scope_name="mmaction",
)

from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)

from mmengine import Config
from mmaction.registry import MODELS

from app.dataset_rwf2000 import RWF2000Dataset
from app.encoder_yolo import YOLOEncoder


# ============================================================
# 8. TEMPORAL ATTENTION MODULE
# ============================================================

class TemporalAttention(nn.Module):
    """
    Learns the importance of each temporal segment.

    Input:
        [B, T, D]

    Output:
        video_features:    [B, D]
        attention_weights: [B, T]
    """

    def __init__(
        self,
        feature_dim,
        hidden_dim=256,
    ):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                feature_dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                1,
                bias=False,
            ),
        )

    def forward(self, segment_features):

        # Calculate one score for every segment.
        scores = self.attention(
            segment_features
        ).squeeze(-1)

        # Normalize segment importance.
        weights = torch.softmax(
            scores,
            dim=1,
        )

        # Weighted temporal aggregation.
        video_features = torch.sum(
            segment_features * weights.unsqueeze(-1),
            dim=1,
        )

        return video_features, weights


# ============================================================
# 9. Vi-SAFE TEMPORAL ATTENTION MODEL
# ============================================================

class ViSAFETemporalAttention(nn.Module):

    def __init__(
        self,
        config_path,
        yolo_encoding_dim=128,
        freeze_tsn_backbone=False,
    ):

        super().__init__()

        config = Config.fromfile(
            str(config_path)
        )

        if "train_cfg" not in config.model:
            config.model["train_cfg"] = config.get(
                "train_cfg",
                None,
            )

        if "test_cfg" not in config.model:
            config.model["test_cfg"] = config.get(
                "test_cfg",
                None,
            )

        # Binary classification for RWF-2000.
        config.model.cls_head.num_classes = 2

        self.tsn = MODELS.build(
            config.model
        )

        if freeze_tsn_backbone:

            for parameter in self.tsn.backbone.parameters():
                parameter.requires_grad = False

        original_fc = self.tsn.cls_head.fc_cls

        self.tsn_feature_dim = original_fc.in_features

        self.yolo_feature_dim = yolo_encoding_dim

        self.fused_feature_dim = (
            self.tsn_feature_dim + self.yolo_feature_dim
        )

        self.temporal_attention = TemporalAttention(
            feature_dim=self.fused_feature_dim,
            hidden_dim=ATTENTION_HIDDEN_DIM,
        )

        self.classifier = nn.Linear(
            self.fused_feature_dim,
            2,
        )

        # Initialize TSN portion from the TSN classification head.
        with torch.no_grad():

            self.classifier.weight[:, :self.tsn_feature_dim].copy_(
                original_fc.weight
            )

            self.classifier.bias.copy_(
                original_fc.bias
            )

    def forward(
        self,
        frames,
        yolo_encodings,
        return_attention=False,
    ):

        # frames: [B, T, 3, H, W]
        # yolo_encodings: [B, T, 128]

        batch_size, segments, channels, height, width = frames.shape

        if yolo_encodings.shape[:2] != (batch_size, segments):
            raise ValueError(
                "TSN and YOLO segment dimensions do not match."
            )

        # Process all sampled frames through TSN backbone.
        flattened_frames = frames.reshape(
            batch_size * segments,
            channels,
            height,
            width,
        )

        tsn_features = self.tsn.backbone(
            flattened_frames
        )

        if isinstance(tsn_features, (tuple, list)):
            tsn_features = tsn_features[-1]

        tsn_features = self.tsn.cls_head.avg_pool(
            tsn_features
        )

        tsn_features = tsn_features.flatten(1)

        # Restore temporal dimension.
        tsn_features = tsn_features.reshape(
            batch_size,
            segments,
            self.tsn_feature_dim,
        )

        # IMPORTANT:
        # Do not mean-pool the temporal dimension.
        # Keep individual segment features for attention.
        fused_segments = torch.cat(
            [
                tsn_features,
                yolo_encodings,
            ],
            dim=-1,
        )

        # Learned temporal attention.
        video_features, attention_weights = self.temporal_attention(
            fused_segments
        )

        logits = self.classifier(
            video_features
        )

        if return_attention:
            return logits, attention_weights

        return logits


# ============================================================
# 10. DATASET AND DATALOADERS
# ============================================================

train_transform = T.Compose([

    T.ToPILImage(),

    T.RandomHorizontalFlip(p=0.5),

    T.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1,
    ),

    T.ToTensor(),
])

val_transform = T.Compose([

    T.ToPILImage(),

    T.ToTensor(),
])


train_dataset = RWF2000Dataset(
    root_dir=str(DATA_ROOT),
    mode="train",
    num_segments=args.num_segments,
    transform=train_transform,
    random_seed=args.seed,
)

val_dataset = RWF2000Dataset(
    root_dir=str(DATA_ROOT),
    mode="val",
    num_segments=args.num_segments,
    transform=val_transform,
    random_seed=args.seed,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(args.num_workers > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(args.num_workers > 0),
)


logger.info("Training videos: %d", len(train_dataset))
logger.info("Validation videos: %d", len(val_dataset))


# ============================================================
# 11. INITIALIZE YOLO AND TSN
# ============================================================

yolo_encoder = YOLOEncoder(
    yolo_weights_path=str(YOLO_WEIGHTS),
    device=device,
    freeze_yolo=args.freeze_yolo,
).to(device)

model = ViSAFETemporalAttention(
    config_path=CONFIG_PATH,
    yolo_encoding_dim=YOLO_FEATURE_DIM,
    freeze_tsn_backbone=args.freeze_tsn_backbone,
).to(device)

logger.info("Model initialized.")
logger.info("TSN device: %s", next(model.parameters()).device)
logger.info("YOLO device: %s", next(yolo_encoder.parameters()).device)

logger.info("TSN feature dimension: %d", model.tsn_feature_dim)
logger.info("YOLO feature dimension: %d", model.yolo_feature_dim)
logger.info("Fused segment dimension: %d", model.fused_feature_dim)


# ============================================================
# 12. OPTIMIZER AND SCHEDULER
# ============================================================

trainable_parameters = [

    parameter

    for parameter in (
        list(model.parameters())
        + list(yolo_encoder.parameters())
    )

    if parameter.requires_grad
]

optimizer = optim.Adam(
    trainable_parameters,
    lr=args.lr,
    weight_decay=args.weight_decay,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# 13. CHECKPOINT UTILITIES
# ============================================================

def atomic_save(checkpoint, destination):

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    os.replace(
        temporary_path,
        destination,
    )


def make_checkpoint(
    epoch,
    best_val_acc,
    patience_counter,
):

    return {

        "experiment": "visafe_temporal_attention",

        "epoch": epoch,

        "best_val_acc": best_val_acc,

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

            "weight_decay": args.weight_decay,

            "seed": args.seed,

            "attention_hidden_dim": ATTENTION_HIDDEN_DIM,

            "yolo_feature_dim": YOLO_FEATURE_DIM,
        },
    }


# ============================================================
# 14. RESUME TRAINING
# ============================================================

start_epoch = 1
best_val_acc = -1.0
patience_counter = 0

if args.no_resume:

    logger.info("Fresh run requested.")

    # Only clear temporal-attention experiment files.
    for checkpoint_file in CHECKPOINT_DIR.glob("*.pt"):
        checkpoint_file.unlink()

    if METRICS_PATH.exists():
        METRICS_PATH.unlink()


elif LATEST_CHECKPOINT.exists():

    logger.info("Resuming temporal-attention experiment.")

    # IMPORTANT:
    # Load checkpoint on CPU.
    # This keeps RNG state tensors as CPU ByteTensors.
    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("experiment") != "visafe_temporal_attention":
        raise RuntimeError(
            "Checkpoint does not belong to the temporal-attention experiment."
        )

    saved_config = checkpoint.get("config", {})

    required_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_segments": args.num_segments,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "attention_hidden_dim": ATTENTION_HIDDEN_DIM,
        "yolo_feature_dim": YOLO_FEATURE_DIM,
    }

    mismatches = [
        key
        for key, value in required_config.items()
        if saved_config.get(key) != value
    ]

    if mismatches:
        raise ValueError(
            "Cannot resume because these settings differ from the checkpoint: "
            + ", ".join(mismatches)
            + ". Use the original settings or --no-resume."
        )

    # Restore model weights.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    yolo_encoder.load_state_dict(
        checkpoint["yolo_encoder_state_dict"]
    )

    # Restore optimizer.
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # Restore scheduler.
    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    # Restore training progress.
    start_epoch = checkpoint["epoch"] + 1

    best_val_acc = checkpoint["best_val_acc"]

    patience_counter = checkpoint["patience_counter"]

    # Restore Python RNG.
    random.setstate(
        checkpoint["python_rng_state"]
    )

    # Restore NumPy RNG.
    np.random.set_state(
        checkpoint["numpy_rng_state"]
    )

    # Restore CPU PyTorch RNG.
    torch.set_rng_state(
        checkpoint["torch_rng_state"].cpu()
    )

    # Restore CUDA RNG states.
    torch.cuda.set_rng_state_all(
        [state.cpu() for state in checkpoint["cuda_rng_state"]]
    )

    logger.info(
        "Restored after epoch %d. Next epoch: %d",
        checkpoint["epoch"],
        start_epoch,
    )

    logger.info(
        "Best validation accuracy: %.4f",
        best_val_acc,
    )

    logger.info("Checkpoint restored successfully.")

else:

    logger.info("No previous checkpoint found. Starting from epoch 1.")


# ============================================================
# 15. YOLO FEATURE EXTRACTION
# ============================================================

def extract_yolo_features(frames):
    """
    Input:
        [B, T, 3, H, W]

    Output:
        [B, T, 128]
    """

    encoded_batch = []

    for batch_index in range(frames.shape[0]):

        encoded = yolo_encoder(
            frames[batch_index]
        )

        encoded_batch.append(encoded)

    return torch.stack(
        encoded_batch,
        dim=0,
    )


# ============================================================
# 16. TRAINING FUNCTION
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
        dynamic_ncols=True,
    )

    for frames, labels in progress:

        frames = frames.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).long()

        yolo_features = extract_yolo_features(
            frames
        )

        logits = model(
            frames,
            yolo_features,
        )

        loss = criterion(
            logits,
            labels,
        )

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        batch_count = labels.size(0)

        total_loss += loss.item() * batch_count

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_count

    return (
        total_loss / max(total_samples, 1),
        total_correct / max(total_samples, 1),
    )


# ============================================================
# 17. VALIDATION FUNCTION
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
        dynamic_ncols=True,
    )

    for frames, labels in progress:

        frames = frames.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).long()

        yolo_features = extract_yolo_features(
            frames
        )

        logits = model(
            frames,
            yolo_features,
        )

        loss = criterion(
            logits,
            labels,
        )

        predictions = logits.argmax(dim=1)

        batch_count = labels.size(0)

        total_loss += loss.item() * batch_count

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_count

    return (
        total_loss / max(total_samples, 1),
        total_correct / max(total_samples, 1),
    )


# ============================================================
# 18. METRICS CSV
# ============================================================

if not METRICS_PATH.exists():

    with METRICS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
            "time_seconds",
        ])


# ============================================================
# 19. MAIN TRAINING LOOP
# ============================================================

logger.info("=" * 70)
logger.info("STARTING TEMPORAL ATTENTION TRAINING")
logger.info("Starting epoch: %d", start_epoch)
logger.info("Total epochs: %d", args.epochs)
logger.info("Latest checkpoint: saved after every epoch")
logger.info("Periodic checkpoint: every %d epochs", SAVE_EVERY)
logger.info("=" * 70)

if start_epoch > args.epochs:

    logger.info(
        "Checkpoint already reached the requested epoch count."
    )


for epoch in range(start_epoch, args.epochs + 1):

    epoch_start = time.time()

    # -------------------------
    # Training
    # -------------------------

    train_loss, train_acc = train_one_epoch(
        epoch
    )

    # -------------------------
    # Validation
    # -------------------------

    val_loss, val_acc = validate(
        epoch
    )

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
        epoch_time / 60,
    )

    # -------------------------
    # Save metrics
    # -------------------------

    with METRICS_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            epoch_time,
        ])

    # -------------------------
    # Update best model
    # -------------------------

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        patience_counter = 0

        best_checkpoint = make_checkpoint(
            epoch,
            best_val_acc,
            patience_counter,
        )

        atomic_save(
            best_checkpoint,
            BEST_MODEL,
        )

        logger.info(
            "BEST MODEL SAVED | Epoch: %d | Val Acc: %.4f",
            epoch,
            best_val_acc,
        )

    else:

        patience_counter += 1

    # -------------------------
    # Save latest checkpoint
    # Every completed epoch
    # -------------------------

    checkpoint = make_checkpoint(
        epoch,
        best_val_acc,
        patience_counter,
    )

    atomic_save(
        checkpoint,
        LATEST_CHECKPOINT,
    )

    logger.info(
        "Latest checkpoint saved after epoch %d.",
        epoch,
    )

    # -------------------------
    # Periodic checkpoint
    # Every 5 epochs
    # -------------------------

    if epoch % SAVE_EVERY == 0:

        periodic_path = (
            CHECKPOINT_DIR
            / f"checkpoint_epoch_{epoch:03d}.pt"
        )

        atomic_save(
            checkpoint,
            periodic_path,
        )

        logger.info(
            "Periodic checkpoint saved: %s",
            periodic_path.name,
        )

    # -------------------------
    # GPU memory report
    # -------------------------

    allocated = (
        torch.cuda.memory_allocated(0)
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved(0)
        / (1024 ** 3)
    )

    logger.info(
        "GPU memory allocated: %.2f GB | Reserved: %.2f GB",
        allocated,
        reserved,
    )

    # -------------------------
    # Early stopping
    # -------------------------

    if (
        args.early_stop_patience > 0
        and patience_counter >= args.early_stop_patience
    ):

        logger.info(
            "EARLY STOPPING at epoch %d.",
            epoch,
        )

        break


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

logger.info("=" * 70)
logger.info("TEMPORAL ATTENTION TRAINING FINISHED")
logger.info("=" * 70)

logger.info(
    "Best validation accuracy: %.4f",
    best_val_acc,
)

logger.info(
    "Best model: %s",
    BEST_MODEL,
)

logger.info(
    "Latest checkpoint: %s",
    LATEST_CHECKPOINT,
)

logger.info(
    "Metrics: %s",
    METRICS_PATH,
)

logger.info(
    "Training log: %s",
    LOG_PATH,
)

logger.info(
    "Checkpoint directory: %s",
    CHECKPOINT_DIR,
)

logger.info("=" * 70)
