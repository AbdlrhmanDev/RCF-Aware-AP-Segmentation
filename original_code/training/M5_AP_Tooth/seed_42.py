# ORIGINAL SOURCE CELL 44 FROM USER TRAINING NOTEBOOK; no scientific logic modified.
# ============================================================
# MODEL 5 — DEEPLABV3+ (AP + TOOTH)
# RCF-Aware AP Segmentation Research
#
# SAME PROTOCOL AS M2, DIFFERENT ARCHITECTURE
# ============================================================

import os
import gc
import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from pathlib import Path
from torch.utils.data import (
    Dataset, DataLoader, WeightedRandomSampler
)
from tqdm.auto import tqdm

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
MAX_EPOCHS = 25
PATIENCE = 6
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
TOOTH_WEIGHT = 0.30
POSITIVE_SAMPLING_FRACTION = 0.40

DEVICE = torch.device("cuda")

assert torch.cuda.is_available(), "GPU required."
assert "REAL_MYDRIVE" in globals()
assert "train_dataset" in globals()
assert "val_dataset" in globals()
assert "val_loader" in globals()

assert len(train_dataset) == 3466
assert len(val_dataset) == 743

assert int(train_dataset.df["AP"].sum()) == 460
assert int(val_dataset.df["AP"].sum()) == 98

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False

# ============================================================
# 2. INDEPENDENT OUTPUT DIRECTORY
# ============================================================

ROOT = (
    Path(REAL_MYDRIVE)
    / "RCF_Aware_AP_Segmentation_Research"
)

M5_DIR = (
    ROOT / "04_Models"
    / "M05_DeepLabV3Plus_AP_Tooth_Seed42"
)

M5_DIR.mkdir(parents=True, exist_ok=True)

M5_BEST = M5_DIR / "best_model.pt"
M5_LAST = M5_DIR / "last_checkpoint.pt"
M5_HISTORY = M5_DIR / "training_history.csv"
M5_CONFIG_PATH = M5_DIR / "training_config.json"

M5_CONFIG = {
    "model": "DeepLabV3Plus",
    "encoder": "resnet18",
    "encoder_weights": "imagenet",
    "encoder_output_stride": 16,
    "tasks": ["AP", "Tooth"],
    "output_channels": 2,
    "AP_channel": 0,
    "Tooth_channel": 1,
    "image_size": 512,
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "learning_rate": LEARNING_RATE,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "tooth_weight": TOOTH_WEIGHT,
    "positive_sampling_fraction":
        POSITIVE_SAMPLING_FRACTION,
    "loss": "BCE_plus_positive_image_Dice_per_task",
    "checkpoint_selection": "minimum_AP_validation_loss",
    "validation_threshold": 0.5,
    "split_version": "v1"
}

if M5_CONFIG_PATH.exists():

    with open(M5_CONFIG_PATH) as f:
        existing_config = json.load(f)

    assert existing_config == M5_CONFIG, (
        "Existing M5 configuration differs."
    )

else:

    if M5_BEST.exists() or M5_LAST.exists():
        raise RuntimeError(
            "M5 checkpoint exists without configuration. "
            "Review files before continuing."
        )

    with open(M5_CONFIG_PATH, "w") as f:
        json.dump(M5_CONFIG, f, indent=2)

if M5_BEST.exists() != M5_LAST.exists():
    raise RuntimeError(
        "Incomplete M5 checkpoint pair. "
        "Review saved files before continuing."
    )

# ============================================================
# 3. RELEASE PREVIOUS GPU MODEL OBJECTS
# Saved model checkpoints remain untouched.
# ============================================================

for name in [
    "model_m4",
    "optimizer_m4",
    "scheduler_m4",
    "scaler_m4",
    "model_m3",
    "optimizer_m3",
    "scheduler_m3",
    "scaler_m3",
    "model_m2",
    "optimizer_m2",
    "scheduler_m2",
    "scaler_m2"
]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 4. SAME AUGMENTATION AS M2
# ============================================================

class M5AugmentedDataset(Dataset):

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):

        sample = self.base[index]

        image = sample["image"].clone()
        ap = sample["AP_mask"].clone()
        tooth = sample["Tooth_mask"].clone()

        # Same horizontal flip as M2
        if random.random() < 0.5:

            image = torch.flip(image, dims=[2])
            ap = torch.flip(ap, dims=[2])
            tooth = torch.flip(tooth, dims=[2])

        # Same mild intensity augmentation as M2
        if random.random() < 0.5:

            contrast = random.uniform(0.95, 1.05)
            brightness = random.uniform(-0.08, 0.08)

            image = torch.clamp(
                image * contrast + brightness,
                -3.0, 3.0
            )

        return {
            "image": image,
            "AP_mask": ap,
            "Tooth_mask": tooth
        }

m5_train_dataset = M5AugmentedDataset(train_dataset)

# ============================================================
# 5. SAME AP-AWARE SAMPLER AS M2
# ============================================================

m5_labels = (
    train_dataset.df["AP"]
    .astype(int)
    .to_numpy()
)

n_positive = int((m5_labels == 1).sum())
n_negative = int((m5_labels == 0).sum())

assert n_positive == 460
assert n_negative == 3006

m5_sample_weights = np.where(
    m5_labels == 1,
    POSITIVE_SAMPLING_FRACTION / n_positive,
    (1.0 - POSITIVE_SAMPLING_FRACTION) / n_negative
)

m5_sample_weights = torch.as_tensor(
    m5_sample_weights,
    dtype=torch.double
)

def create_m5_train_loader(epoch):

    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)

    sampler = WeightedRandomSampler(
        weights=m5_sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator
    )

    return DataLoader(
        m5_train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

# ============================================================
# 6. MODEL
# ============================================================

# Pretrained weights only needed when training from scratch.
# Resume loads all learned weights from the checkpoint.

encoder_weights = (
    None if M5_LAST.exists() else "imagenet"
)

model_m5 = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights=encoder_weights,
    encoder_output_stride=16,
    in_channels=3,
    classes=2,
    activation=None
).to(DEVICE)

optimizer_m5 = torch.optim.AdamW(
    model_m5.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler_m5 = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_m5,
        mode="min",
        factor=0.5,
        patience=2
    )
)

scaler_m5 = torch.amp.GradScaler("cuda")

# ============================================================
# 7. LOSS — IDENTICAL DEFINITION TO M2
# ============================================================

def m5_loss_components(logits, targets):

    logits = logits.float()
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    ).mean(dim=(1, 2, 3))

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    predicted_area = probabilities.sum(
        dim=(1, 2, 3)
    )

    target_area = targets.sum(
        dim=(1, 2, 3)
    )

    soft_dice_loss = 1.0 - (
        (2.0 * intersection + 1.0)
        / (predicted_area + target_area + 1.0)
    )

    positive = target_area > 0

    return bce, soft_dice_loss, positive


def m5_task_loss(logits, targets):

    bce, soft_dice_loss, positive = (
        m5_loss_components(logits, targets)
    )

    loss = bce.mean()

    if positive.any():
        loss = loss + soft_dice_loss[positive].mean()

    return loss


def m5_training_loss(outputs, ap_target, tooth_target):

    ap_logits = outputs[:, 0:1]
    tooth_logits = outputs[:, 1:2]

    ap_loss = m5_task_loss(
        ap_logits, ap_target
    )

    tooth_loss = m5_task_loss(
        tooth_logits, tooth_target
    )

    total_loss = (
        ap_loss + TOOTH_WEIGHT * tooth_loss
    )

    return total_loss, ap_loss, tooth_loss

# ============================================================
# 8. VALIDATION — SAME METRICS AS M2
# No fixed assumption about Tooth mask count.
# ============================================================

@torch.no_grad()
def validate_m5(model, loader):

    model.eval()

    total_bce = 0.0
    total_soft_dice = 0.0
    total_images = 0

    ap_dice_scores = []
    ap_iou_scores = []
    tooth_dice_scores = []

    ap_positive_count = 0
    ap_negative_count = 0
    ap_negative_fp_count = 0

    for batch in loader:

        images = batch["image"].to(
            DEVICE, non_blocking=True
        )

        ap_targets = batch["AP_mask"].to(
            DEVICE, non_blocking=True
        )

        tooth_targets = batch["Tooth_mask"].to(
            DEVICE, non_blocking=True
        )

        with torch.amp.autocast("cuda"):
            outputs = model(images)

        ap_logits = outputs[:, 0:1].float()
        tooth_logits = outputs[:, 1:2].float()

        bce, soft_dice, positive = m5_loss_components(
            ap_logits, ap_targets
        )

        total_bce += bce.sum().item()

        if positive.any():
            total_soft_dice += (
                soft_dice[positive].sum().item()
            )

        ap_predictions = ap_logits >= 0
        tooth_predictions = tooth_logits >= 0

        ap_gt = ap_targets >= 0.5
        tooth_gt = tooth_targets >= 0.5

        for i in range(images.shape[0]):

            # ---------------------------
            # AP segmentation
            # ---------------------------

            gt = ap_gt[i]
            pred = ap_predictions[i]

            gt_pixels = int(gt.sum().item())
            pred_pixels = int(pred.sum().item())

            if gt_pixels > 0:

                intersection = int(
                    (gt & pred).sum().item()
                )

                dice = (
                    2 * intersection
                    / (gt_pixels + pred_pixels)
                )

                union = (
                    gt_pixels + pred_pixels
                    - intersection
                )

                iou = (
                    intersection / union
                    if union > 0 else 0.0
                )

                ap_dice_scores.append(dice)
                ap_iou_scores.append(iou)

                ap_positive_count += 1

            else:

                ap_negative_count += 1

                if pred_pixels > 0:
                    ap_negative_fp_count += 1

            # ---------------------------
            # Tooth segmentation
            # ---------------------------

            t_gt = tooth_gt[i]
            t_pred = tooth_predictions[i]

            t_gt_pixels = int(t_gt.sum().item())
            t_pred_pixels = int(t_pred.sum().item())

            if t_gt_pixels > 0:

                t_intersection = int(
                    (t_gt & t_pred).sum().item()
                )

                tooth_dice_scores.append(
                    2 * t_intersection
                    / (t_gt_pixels + t_pred_pixels)
                )

        total_images += images.shape[0]

    assert total_images == 743
    assert ap_positive_count == 98
    assert ap_negative_count == 645

    if not tooth_dice_scores:
        raise RuntimeError(
            "No positive Tooth masks in validation."
        )

    val_ap_loss = (
        total_bce / total_images
        + total_soft_dice / ap_positive_count
    )

    return {
        "val_loss": float(val_ap_loss),
        "val_AP_Dice": float(
            np.mean(ap_dice_scores)
        ),
        "val_AP_IoU": float(
            np.mean(ap_iou_scores)
        ),
        "val_Tooth_Dice": float(
            np.mean(tooth_dice_scores)
        ),
        "val_negative_FP_rate": (
            ap_negative_fp_count / ap_negative_count
        ),
        "val_positive_images": ap_positive_count,
        "val_negative_images": ap_negative_count,
        "val_Tooth_positive_images": len(
            tooth_dice_scores
        )
    }

# ============================================================
# 9. ATOMIC CHECKPOINT SAVING
# ============================================================

def save_m5_checkpoint(state, destination):

    temporary = destination.with_suffix(".tmp")

    torch.save(state, temporary)
    os.replace(temporary, destination)

# ============================================================
# 10. RESTORE PREVIOUS STATE IF AVAILABLE
# ============================================================

start_epoch = 1
best_val_loss = float("inf")
best_epoch = 0
bad_epochs = 0
history = []

if M5_LAST.exists():

    print("Existing M5 checkpoint found.")

    checkpoint = torch.load(
        M5_LAST,
        map_location="cpu",
        weights_only=True
    )

    assert checkpoint["config"] == M5_CONFIG

    model_m5.load_state_dict(
        checkpoint["model_state"]
    )

    optimizer_m5.load_state_dict(
        checkpoint["optimizer_state"]
    )

    scheduler_m5.load_state_dict(
        checkpoint["scheduler_state"]
    )

    scaler_m5.load_state_dict(
        checkpoint["scaler_state"]
    )

    best_val_loss = float(
        checkpoint["best_val_loss"]
    )

    best_epoch = int(
        checkpoint["best_epoch"]
    )

    bad_epochs = int(
        checkpoint["bad_epochs"]
    )

    history = list(checkpoint["history"])

    start_epoch = int(checkpoint["epoch"]) + 1

    assert [
        int(record["epoch"])
        for record in history
    ] == list(range(1, start_epoch))

# ============================================================
# 11. PRE-TRAINING SAFETY CHECK
# ============================================================

print("\n" + "=" * 65)
print("MODEL 5 — PRE-TRAINING SAFETY CHECK")
print("=" * 65)

print("GPU:", torch.cuda.get_device_name(0))
print("Architecture: DeepLabV3+")
print("Encoder: ResNet18")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Start epoch:", start_epoch)
print("Tooth weight:", TOOTH_WEIGHT)

# Test full batch forward/backward without updating weights.
# Use eval mode so batch-normalization statistics
# remain unchanged during this probe.

model_m5.eval()

probe_batch = next(iter(create_m5_train_loader(
    start_epoch
)))

probe_images = probe_batch["image"].to(DEVICE)
probe_ap = probe_batch["AP_mask"].to(DEVICE)
probe_tooth = probe_batch["Tooth_mask"].to(DEVICE)

assert probe_images.shape == (4, 3, 512, 512)

optimizer_m5.zero_grad(set_to_none=True)

with torch.amp.autocast("cuda"):

    probe_outputs = model_m5(probe_images)

    probe_loss, _, _ = m5_training_loss(
        probe_outputs,
        probe_ap,
        probe_tooth
    )

assert torch.isfinite(probe_loss).item()

probe_loss.backward()

optimizer_m5.zero_grad(set_to_none=True)

del (
    probe_batch,
    probe_images,
    probe_ap,
    probe_tooth,
    probe_outputs,
    probe_loss
)

gc.collect()
torch.cuda.empty_cache()

print("Batch-4 forward/backward: PASSED")

# Check full validation before training.
# This catches potential mask-count errors early.

preflight_metrics = validate_m5(
    model_m5,
    val_loader
)

assert np.isfinite(preflight_metrics["val_loss"])

print("Full validation preflight: PASSED")
print(
    "AP-positive validation images:",
    preflight_metrics["val_positive_images"]
)
print(
    "Tooth-positive resized masks:",
    preflight_metrics["val_Tooth_positive_images"]
)

print("\nM5 PRE-TRAINING CHECK: PASSED")

# ============================================================
# 12. TRAINING LOOP
# ============================================================

print("\n" + "=" * 65)
print("MODEL 5 — DEEPLABV3+ AP + TOOTH TRAINING")
print("=" * 65)

print("GPU:", torch.cuda.get_device_name(0))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

print("Maximum epochs:", MAX_EPOCHS)
print("Starting epoch:", start_epoch)
print("Learning rate:", LEARNING_RATE)
print("Tooth auxiliary weight:", TOOTH_WEIGHT)
print("Output directory:", M5_DIR)

for epoch in range(start_epoch, MAX_EPOCHS + 1):

    if bad_epochs >= PATIENCE:
        print("Early stopping already reached.")
        break

    epoch_start_time = time.time()

    # Same epoch-level seed protocol as M2
    random.seed(SEED + epoch)
    np.random.seed(SEED + epoch)
    torch.manual_seed(SEED + epoch)

    train_loader_epoch = create_m5_train_loader(
        epoch
    )

    model_m5.train()

    total_training_loss = 0.0
    total_ap_loss = 0.0
    total_tooth_loss = 0.0
    seen_images = 0

    progress = tqdm(
        train_loader_epoch,
        desc=f"M5 Epoch {epoch}/{MAX_EPOCHS}"
    )

    for batch in progress:

        images = batch["image"].to(
            DEVICE, non_blocking=True
        )

        ap_targets = batch["AP_mask"].to(
            DEVICE, non_blocking=True
        )

        tooth_targets = batch["Tooth_mask"].to(
            DEVICE, non_blocking=True
        )

        optimizer_m5.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast("cuda"):

            outputs = model_m5(images)

            (
                loss,
                ap_loss,
                tooth_loss
            ) = m5_training_loss(
                outputs,
                ap_targets,
                tooth_targets
            )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )

        scaler_m5.scale(loss).backward()

        scaler_m5.unscale_(optimizer_m5)

        torch.nn.utils.clip_grad_norm_(
            model_m5.parameters(),
            max_norm=1.0
        )

        scaler_m5.step(optimizer_m5)
        scaler_m5.update()

        bs = images.shape[0]

        total_training_loss += loss.item() * bs
        total_ap_loss += ap_loss.item() * bs
        total_tooth_loss += tooth_loss.item() * bs

        seen_images += bs

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    assert seen_images == 3466

    train_loss = (
        total_training_loss / seen_images
    )

    train_ap_loss = (
        total_ap_loss / seen_images
    )

    train_tooth_loss = (
        total_tooth_loss / seen_images
    )

    # --------------------------------------------------------
    # VALIDATION: Original distribution, no augmentation
    # --------------------------------------------------------

    metrics = validate_m5(
        model_m5,
        val_loader
    )

    assert np.isfinite(metrics["val_loss"])

    scheduler_m5.step(
        metrics["val_loss"]
    )

    current_lr = (
        optimizer_m5.param_groups[0]["lr"]
    )

    epoch_minutes = (
        time.time() - epoch_start_time
    ) / 60.0

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_AP_loss": train_ap_loss,
        "train_Tooth_loss": train_tooth_loss,
        **metrics,
        "learning_rate": current_lr,
        "epoch_minutes": epoch_minutes
    }

    # --------------------------------------------------------
    # BEST CHECKPOINT: AP validation loss only
    # --------------------------------------------------------

    improved = (
        metrics["val_loss"]
        < best_val_loss - 1e-4
    )

    if improved:

        best_val_loss = metrics["val_loss"]
        best_epoch = epoch
        bad_epochs = 0

        save_m5_checkpoint(
            {
                "epoch": epoch,
                "model_state": model_m5.state_dict(),
                "val_loss": best_val_loss,
                "config": M5_CONFIG
            },
            M5_BEST
        )

    else:

        bad_epochs += 1

    history.append(record)

    # --------------------------------------------------------
    # SAVE RESUMABLE CHECKPOINT
    # --------------------------------------------------------

    save_m5_checkpoint(
        {
            "epoch": epoch,
            "model_state": model_m5.state_dict(),
            "optimizer_state":
                optimizer_m5.state_dict(),
            "scheduler_state":
                scheduler_m5.state_dict(),
            "scaler_state":
                scaler_m5.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
            "config": M5_CONFIG
        },
        M5_LAST
    )

    pd.DataFrame(history).to_csv(
        M5_HISTORY,
        index=False
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print(f"M5 EPOCH {epoch}/{MAX_EPOCHS}")
    print("-" * 65)

    print(f"Train Total Loss: {train_loss:.4f}")
    print(f"Train AP Loss: {train_ap_loss:.4f}")
    print(f"Train Tooth Loss: {train_tooth_loss:.4f}")

    print(f"Val AP Loss: {metrics['val_loss']:.4f}")
    print(f"Val AP Dice: {metrics['val_AP_Dice']:.4f}")
    print(f"Val AP IoU: {metrics['val_AP_IoU']:.4f}")
    print(
        "Val Tooth Dice:",
        f"{metrics['val_Tooth_Dice']:.4f}"
    )
    print(
        "Val AP Negative FP Rate:",
        f"{metrics['val_negative_FP_rate']:.4f}"
    )

    print(f"Learning Rate: {current_lr:.7f}")
    print(f"Epoch Time: {epoch_minutes:.2f} minutes")
    print(f"Best Epoch: {best_epoch}")

    if improved:
        print("BEST MODEL UPDATED")

    print("CHECKPOINT SAVED")

    if bad_epochs >= PATIENCE:
        print("\nEARLY STOPPING ACTIVATED")
        break

# ============================================================
# 13. FINAL REPORT
# ============================================================

assert M5_BEST.is_file()
assert M5_LAST.is_file()
assert M5_HISTORY.is_file()

best_record = next(
    record for record in history
    if record["epoch"] == best_epoch
)

print("\n" + "=" * 65)
print("MODEL 5 TRAINING FINISHED")
print("=" * 65)

print("Epochs completed:", len(history))
print("Best epoch:", best_epoch)
print(
    "Best AP validation loss:",
    round(best_val_loss, 5)
)

print("\nBEST CHECKPOINT METRICS")

print(
    "Val AP Dice:",
    round(best_record["val_AP_Dice"], 4)
)

print(
    "Val AP IoU:",
    round(best_record["val_AP_IoU"], 4)
)

print(
    "Val Tooth Dice:",
    round(best_record["val_Tooth_Dice"], 4)
)

print(
    "Val AP Negative FP Rate:",
    round(best_record["val_negative_FP_rate"], 4)
)

print("\nSaved files:")
print("Best model:", M5_BEST)
print("Last checkpoint:", M5_LAST)
print("Training history:", M5_HISTORY)

print("\nOriginal PRAD dataset: NOT MODIFIED")
print("M1-M4 checkpoints: NOT MODIFIED")
print("Held-out test set: NOT USED")