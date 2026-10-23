# ORIGINAL SOURCE CELL 47 FROM USER TRAINING NOTEBOOK; no scientific logic modified.
# ============================================================
# MODEL 6 — DEEPLABV3+ | AP + TOOTH + RCF
# COMPLETE TRAINING WITH SAFE RESUME
# NO TEST ACCESS
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
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ============================================================
# 1. RESEARCH CONFIGURATION
# ============================================================

assert torch.cuda.is_available(), "Enable T4 GPU first."

for name in ["REAL_MYDRIVE", "train_dataset", "val_dataset", "val_loader"]:
    assert name in globals(), f"Missing {name}. Restore dataset environment."

DEVICE = torch.device("cuda")

SEED = 42
BATCH_SIZE = 4
MAX_EPOCHS = 25
PATIENCE = 6
LR = 1e-4
WEIGHT_DECAY = 1e-4
TOOTH_WEIGHT = 0.30
RCF_WEIGHT = 0.30
AP_SAMPLE_FRACTION = 0.40

assert len(train_dataset) == 3466
assert len(val_dataset) == 743
assert int(train_dataset.df["AP"].sum()) == 460
assert int(val_dataset.df["AP"].sum()) == 98
assert int(val_dataset.df["RCF"].sum()) == 288

M6_CONFIG = {
    "model": "DeepLabV3Plus",
    "encoder": "resnet18",
    "encoder_weights": "imagenet",
    "encoder_output_stride": 16,
    "tasks": ["AP", "Tooth", "RCF"],
    "output_channels": 3,
    "AP_channel": 0,
    "Tooth_channel": 1,
    "RCF_channel": 2,
    "image_size": 512,
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "learning_rate": LR,
    "optimizer": "AdamW",
    "weight_decay": WEIGHT_DECAY,
    "tooth_weight": TOOTH_WEIGHT,
    "rcf_weight": RCF_WEIGHT,
    "positive_sampling_fraction": AP_SAMPLE_FRACTION,
    "loss": "BCE_plus_positive_image_Dice_per_task",
    "checkpoint_selection": "minimum_AP_validation_loss",
    "validation_threshold": 0.5,
    "split_version": "v1"
}

# ============================================================
# 2. SEPARATE M6 STORAGE
# ============================================================

ROOT = (
    Path(REAL_MYDRIVE)
    / "RCF_Aware_AP_Segmentation_Research"
)

M6_DIR = (
    ROOT / "04_Models"
    / "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed42"
)

M6_DIR.mkdir(parents=True, exist_ok=True)

M6_BEST = M6_DIR / "best_model.pt"
M6_LAST = M6_DIR / "last_checkpoint.pt"
M6_HISTORY = M6_DIR / "training_history.csv"
M6_CONFIG_PATH = M6_DIR / "training_config.json"

# Prevent accidental mismatch or overwrite.
if M6_CONFIG_PATH.exists():

    with open(M6_CONFIG_PATH) as f:
        existing_config = json.load(f)

    assert existing_config == M6_CONFIG, (
        "M6 configuration differs from the saved experiment."
    )

else:

    assert not M6_BEST.exists() and not M6_LAST.exists(), (
        "Checkpoint exists without configuration. Review files."
    )

    with open(M6_CONFIG_PATH, "w") as f:
        json.dump(M6_CONFIG, f, indent=2)

if M6_BEST.exists() != M6_LAST.exists():
    raise RuntimeError(
        "Only one M6 checkpoint exists. "
        "Stop and inspect the saved state before resuming."
    )

# ============================================================
# 3. RELEASE UNUSED MODELS FROM GPU
# ============================================================

for name in [
    "model_m5", "optimizer_m5", "scheduler_m5", "scaler_m5",
    "model_m4", "optimizer_m4", "scheduler_m4", "scaler_m4",
    "model_m6", "optimizer_m6", "scheduler_m6", "scaler_m6",
    "model_eval"
]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

# ============================================================
# 4. SAME TRAIN AUGMENTATION AS M4/M5
# ============================================================

class M6AugmentedDataset(Dataset):

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):

        sample = self.base[index]

        image = sample["image"].clone()
        ap = sample["AP_mask"].clone()
        tooth = sample["Tooth_mask"].clone()
        rcf = sample["RCF_mask"].clone()

        # Horizontal flip
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            ap = torch.flip(ap, dims=[2])
            tooth = torch.flip(tooth, dims=[2])
            rcf = torch.flip(rcf, dims=[2])

        # Mild image intensity augmentation
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
            "Tooth_mask": tooth,
            "RCF_mask": rcf
        }

m6_train_dataset = M6AugmentedDataset(train_dataset)

# ============================================================
# 5. AP-AWARE WEIGHTED SAMPLING
# ============================================================

train_ap_labels = (
    train_dataset.df["AP"].astype(int).to_numpy()
)

n_pos = int((train_ap_labels == 1).sum())
n_neg = int((train_ap_labels == 0).sum())

assert n_pos == 460
assert n_neg == 3006

sample_weights = np.where(
    train_ap_labels == 1,
    AP_SAMPLE_FRACTION / n_pos,
    (1 - AP_SAMPLE_FRACTION) / n_neg
)

sample_weights = torch.tensor(
    sample_weights, dtype=torch.double
)

def create_m6_train_loader(epoch):

    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator
    )

    return DataLoader(
        m6_train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

# ============================================================
# 6. BUILD DEEPLABV3+ MODEL
# ============================================================

# A resumed model uses checkpoint weights.
# A new model uses ImageNet initialization.

encoder_weights = (
    None if M6_LAST.exists() else "imagenet"
)

model_m6 = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights=encoder_weights,
    encoder_output_stride=16,
    in_channels=3,
    classes=3,
    activation=None
).to(DEVICE)

optimizer_m6 = torch.optim.AdamW(
    model_m6.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler_m6 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_m6,
    mode="min",
    factor=0.5,
    patience=2
)

scaler_m6 = torch.amp.GradScaler("cuda")

# ============================================================
# 7. LOSS FUNCTIONS
# ============================================================

def m6_loss_components(logits, targets):

    logits = logits.float()
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).mean(dim=(1, 2, 3))

    probs = torch.sigmoid(logits)

    intersection = (
        probs * targets
    ).sum(dim=(1, 2, 3))

    pred_area = probs.sum(dim=(1, 2, 3))
    gt_area = targets.sum(dim=(1, 2, 3))

    dice_loss = 1.0 - (
        (2.0 * intersection + 1.0)
        / (pred_area + gt_area + 1.0)
    )

    positive = gt_area > 0

    return bce, dice_loss, positive


def m6_task_loss(logits, targets):

    bce, dice_loss, positive = m6_loss_components(
        logits, targets
    )

    loss = bce.mean()

    if positive.any():
        loss = loss + dice_loss[positive].mean()

    return loss


def m6_training_loss(outputs, ap, tooth, rcf):

    ap_loss = m6_task_loss(
        outputs[:, 0:1], ap
    )

    tooth_loss = m6_task_loss(
        outputs[:, 1:2], tooth
    )

    rcf_loss = m6_task_loss(
        outputs[:, 2:3], rcf
    )

    total_loss = (
        ap_loss
        + TOOTH_WEIGHT * tooth_loss
        + RCF_WEIGHT * rcf_loss
    )

    return total_loss, ap_loss, tooth_loss, rcf_loss

# ============================================================
# 8. VALIDATION — CORRECTED RCF MASK COUNT
# ============================================================

@torch.inference_mode()
def validate_m6(model, loader):

    model.eval()

    total_bce = 0.0
    total_soft_dice = 0.0
    total_images = 0

    ap_dices = []
    ap_ious = []
    tooth_dices = []
    rcf_dices = []

    ap_positive_count = 0
    ap_negative_count = 0
    ap_false_positive_images = 0

    rcf_negative_count = 0
    rcf_false_positive_images = 0

    for batch in loader:

        images = batch["image"].to(DEVICE)
        ap_gt = batch["AP_mask"].to(DEVICE)
        tooth_gt = batch["Tooth_mask"].to(DEVICE)
        rcf_gt = batch["RCF_mask"].to(DEVICE)

        with torch.amp.autocast("cuda"):
            outputs = model(images)

        ap_logits = outputs[:, 0:1].float()

        bce, dice_loss, positive = m6_loss_components(
            ap_logits, ap_gt
        )

        total_bce += bce.sum().item()
        total_soft_dice += dice_loss[positive].sum().item()

        preds = outputs.float() >= 0

        targets = [
            ap_gt >= 0.5,
            tooth_gt >= 0.5,
            rcf_gt >= 0.5
        ]

        for i in range(images.shape[0]):

            for task in range(3):

                pred = preds[i, task]
                gt = targets[task][i, 0]

                gt_pixels = int(gt.sum().item())
                pred_pixels = int(pred.sum().item())

                if gt_pixels > 0:

                    intersection = int(
                        (pred & gt).sum().item()
                    )

                    dice = (
                        2 * intersection
                        / (gt_pixels + pred_pixels)
                    )

                    if task == 0:

                        union = (
                            gt_pixels + pred_pixels
                            - intersection
                        )

                        ap_dices.append(dice)
                        ap_ious.append(
                            intersection / union
                        )
                        ap_positive_count += 1

                    elif task == 1:
                        tooth_dices.append(dice)

                    else:
                        rcf_dices.append(dice)

                else:

                    if task == 0:
                        ap_negative_count += 1
                        ap_false_positive_images += int(
                            pred_pixels > 0
                        )

                    elif task == 2:
                        rcf_negative_count += 1
                        rcf_false_positive_images += int(
                            pred_pixels > 0
                        )

        total_images += images.shape[0]

    # Dataset integrity checks
    assert total_images == 743
    assert ap_positive_count == 98
    assert ap_negative_count == 645

    # 742 positive Tooth masks after preprocessing.
    assert len(tooth_dices) == 742

    # 288 in original metadata, 287 after 512px resizing.
    assert len(rcf_dices) == 287
    assert rcf_negative_count == 456

    val_ap_loss = (
        total_bce / total_images
        + total_soft_dice / ap_positive_count
    )

    return {
        "val_loss": float(val_ap_loss),
        "val_AP_Dice": float(np.mean(ap_dices)),
        "val_AP_IoU": float(np.mean(ap_ious)),
        "val_Tooth_Dice": float(np.mean(tooth_dices)),
        "val_RCF_Dice": float(np.mean(rcf_dices)),
        "val_negative_FP_rate": (
            ap_false_positive_images / ap_negative_count
        ),
        "val_RCF_negative_FP_rate": (
            rcf_false_positive_images / rcf_negative_count
        ),
        "val_positive_images": ap_positive_count,
        "val_negative_images": ap_negative_count,
        "val_Tooth_positive_images": len(tooth_dices),
        "val_RCF_positive_images": len(rcf_dices)
    }

# ============================================================
# 9. ATOMIC CHECKPOINT SAVING
# ============================================================

def save_m6_checkpoint(state, destination):

    temp_path = destination.with_suffix(".tmp")

    torch.save(state, temp_path)
    os.replace(temp_path, destination)

# ============================================================
# 10. RESTORE CHECKPOINT IF M6 ALREADY STARTED
# ============================================================

start_epoch = 1
best_epoch = 0
best_val_loss = float("inf")
bad_epochs = 0
history = []

if M6_LAST.exists():

    saved = torch.load(
        M6_LAST,
        map_location="cpu",
        weights_only=True
    )

    assert saved["config"] == M6_CONFIG

    model_m6.load_state_dict(saved["model_state"])
    optimizer_m6.load_state_dict(saved["optimizer_state"])
    scheduler_m6.load_state_dict(saved["scheduler_state"])
    scaler_m6.load_state_dict(saved["scaler_state"])

    best_epoch = int(saved["best_epoch"])
    best_val_loss = float(saved["best_val_loss"])
    bad_epochs = int(saved["bad_epochs"])
    history = list(saved["history"])
    start_epoch = int(saved["epoch"]) + 1

    assert [
        int(row["epoch"]) for row in history
    ] == list(range(1, start_epoch))

    print("Existing M6 checkpoint restored.")

# ============================================================
# 11. PRE-TRAINING SAFETY CHECK
# ============================================================

print("\n" + "=" * 65)
print("MODEL 6 — PRE-TRAINING SAFETY CHECK")
print("=" * 65)

print("GPU:", torch.cuda.get_device_name(0))
print("Architecture: DeepLabV3+")
print("Encoder: ResNet18")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Starting epoch:", start_epoch)

# Test batch=4 forward and backward.
# Eval mode protects BatchNorm statistics.
# No optimizer step is performed.

model_m6.eval()

probe = next(iter(create_m6_train_loader(start_epoch)))

probe_images = probe["image"].to(DEVICE)
probe_ap = probe["AP_mask"].to(DEVICE)
probe_tooth = probe["Tooth_mask"].to(DEVICE)
probe_rcf = probe["RCF_mask"].to(DEVICE)

assert probe_images.shape == (4, 3, 512, 512)

optimizer_m6.zero_grad(set_to_none=True)

with torch.amp.autocast("cuda"):

    probe_outputs = model_m6(probe_images)

    probe_loss, _, _, _ = m6_training_loss(
        probe_outputs,
        probe_ap,
        probe_tooth,
        probe_rcf
    )

assert probe_outputs.shape == (4, 3, 512, 512)
assert torch.isfinite(probe_loss).item()

probe_loss.backward()

optimizer_m6.zero_grad(set_to_none=True)

del probe, probe_images, probe_ap
del probe_tooth, probe_rcf, probe_outputs, probe_loss

gc.collect()
torch.cuda.empty_cache()

print("Batch-4 forward/backward: PASSED")

# Validate all 743 images before training.
preflight = validate_m6(model_m6, val_loader)

assert np.isfinite(preflight["val_loss"])

print("Full validation preflight: PASSED")
print("AP-positive images:", preflight["val_positive_images"])
print(
    "Tooth-positive resized masks:",
    preflight["val_Tooth_positive_images"]
)
print(
    "RCF-positive resized masks:",
    preflight["val_RCF_positive_images"]
)

print("\nM6 PRE-TRAINING CHECK: PASSED")

# ============================================================
# 12. TRAINING LOOP
# ============================================================

print("\n" + "=" * 65)
print("MODEL 6 — DEEPLABV3+ AP + TOOTH + RCF TRAINING")
print("=" * 65)

print("Starting epoch:", start_epoch)
print("Maximum epochs:", MAX_EPOCHS)
print("Best previous epoch:", best_epoch)
print("Best previous val loss:", best_val_loss)
print("Early stopping counter:", bad_epochs)
print("Output directory:", M6_DIR)

for epoch in range(start_epoch, MAX_EPOCHS + 1):

    if bad_epochs >= PATIENCE:
        print("Early stopping already reached.")
        break

    epoch_start = time.time()

    # Consistent per-epoch seeding
    random.seed(SEED + epoch)
    np.random.seed(SEED + epoch)
    torch.manual_seed(SEED + epoch)

    epoch_loader = create_m6_train_loader(epoch)

    model_m6.train()

    total_train_loss = 0.0
    total_ap_loss = 0.0
    total_tooth_loss = 0.0
    total_rcf_loss = 0.0
    seen_images = 0

    progress = tqdm(
        epoch_loader,
        desc=f"M6 Epoch {epoch}/{MAX_EPOCHS}"
    )

    for batch in progress:

        images = batch["image"].to(
            DEVICE, non_blocking=True
        )

        ap = batch["AP_mask"].to(
            DEVICE, non_blocking=True
        )

        tooth = batch["Tooth_mask"].to(
            DEVICE, non_blocking=True
        )

        rcf = batch["RCF_mask"].to(
            DEVICE, non_blocking=True
        )

        optimizer_m6.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):

            outputs = model_m6(images)

            loss, ap_loss, tooth_loss, rcf_loss = (
                m6_training_loss(outputs, ap, tooth, rcf)
            )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss during Epoch {epoch}"
            )

        scaler_m6.scale(loss).backward()

        scaler_m6.unscale_(optimizer_m6)

        torch.nn.utils.clip_grad_norm_(
            model_m6.parameters(),
            max_norm=1.0
        )

        scaler_m6.step(optimizer_m6)
        scaler_m6.update()

        bs = images.shape[0]

        total_train_loss += loss.item() * bs
        total_ap_loss += ap_loss.item() * bs
        total_tooth_loss += tooth_loss.item() * bs
        total_rcf_loss += rcf_loss.item() * bs

        seen_images += bs

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    assert seen_images == 3466

    train_loss = total_train_loss / seen_images
    train_ap_loss = total_ap_loss / seen_images
    train_tooth_loss = total_tooth_loss / seen_images
    train_rcf_loss = total_rcf_loss / seen_images

    # ========================================================
    # 13. VALIDATION
    # ========================================================

    metrics = validate_m6(model_m6, val_loader)

    assert metrics["val_positive_images"] == 98
    assert metrics["val_negative_images"] == 645
    assert metrics["val_RCF_positive_images"] == 287
    assert np.isfinite(metrics["val_loss"])

    scheduler_m6.step(metrics["val_loss"])

    current_lr = optimizer_m6.param_groups[0]["lr"]

    epoch_minutes = (time.time() - epoch_start) / 60

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_AP_loss": train_ap_loss,
        "train_Tooth_loss": train_tooth_loss,
        "train_RCF_loss": train_rcf_loss,
        **metrics,
        "learning_rate": current_lr,
        "epoch_minutes": epoch_minutes
    }

    # ========================================================
    # 14. BEST MODEL SELECTION
    # ========================================================

    improved = (
        metrics["val_loss"] < best_val_loss - 1e-4
    )

    if improved:

        best_val_loss = metrics["val_loss"]
        best_epoch = epoch
        bad_epochs = 0

        save_m6_checkpoint(
            {
                "epoch": epoch,
                "model_state": model_m6.state_dict(),
                "val_loss": best_val_loss,
                "config": M6_CONFIG
            },
            M6_BEST
        )

    else:
        bad_epochs += 1

    history.append(record)

    # ========================================================
    # 15. SAVE FULL RESUMABLE CHECKPOINT
    # ========================================================

    save_m6_checkpoint(
        {
            "epoch": epoch,
            "model_state": model_m6.state_dict(),
            "optimizer_state": optimizer_m6.state_dict(),
            "scheduler_state": scheduler_m6.state_dict(),
            "scaler_state": scaler_m6.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
            "config": M6_CONFIG
        },
        M6_LAST
    )

    history_temp = M6_HISTORY.with_suffix(".tmp")

    pd.DataFrame(history).to_csv(
        history_temp,
        index=False
    )

    os.replace(history_temp, M6_HISTORY)

    # ========================================================
    # 16. EPOCH REPORT
    # ========================================================

    print("\n" + "-" * 65)
    print(f"M6 EPOCH {epoch}/{MAX_EPOCHS}")
    print("-" * 65)

    print(f"Train Total Loss: {train_loss:.4f}")
    print(f"Train AP Loss: {train_ap_loss:.4f}")
    print(f"Train Tooth Loss: {train_tooth_loss:.4f}")
    print(f"Train RCF Loss: {train_rcf_loss:.4f}")

    print(f"Val AP Loss: {metrics['val_loss']:.4f}")
    print(f"Val AP Dice: {metrics['val_AP_Dice']:.4f}")
    print(f"Val AP IoU: {metrics['val_AP_IoU']:.4f}")
    print(f"Val Tooth Dice: {metrics['val_Tooth_Dice']:.4f}")
    print(f"Val RCF Dice: {metrics['val_RCF_Dice']:.4f}")

    print(
        "Val AP Negative FP Rate:",
        f"{metrics['val_negative_FP_rate']:.4f}"
    )

    print(
        "Val RCF Negative FP Rate:",
        f"{metrics['val_RCF_negative_FP_rate']:.4f}"
    )

    print(f"Learning Rate: {current_lr:.7f}")
    print(f"Epoch Time: {epoch_minutes:.2f} minutes")
    print(f"Best Epoch: {best_epoch}")
    print(f"Early Stopping Counter: {bad_epochs}/{PATIENCE}")

    if improved:
        print("BEST MODEL UPDATED")

    print("CHECKPOINT SAVED")
    print("-" * 65)

    if bad_epochs >= PATIENCE:
        print("\nEARLY STOPPING ACTIVATED")
        break

# ============================================================
# 17. FINAL INTEGRITY CHECK
# ============================================================

assert M6_BEST.is_file()
assert M6_LAST.is_file()
assert M6_HISTORY.is_file()

final_state = torch.load(
    M6_LAST,
    map_location="cpu",
    weights_only=True
)

assert final_state["config"] == M6_CONFIG
assert int(final_state["epoch"]) == int(history[-1]["epoch"])
assert len(final_state["history"]) == len(history)

best_record = next(
    row for row in history
    if int(row["epoch"]) == best_epoch
)

# ============================================================
# 18. FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("MODEL 6 TRAINING FINISHED")
print("=" * 65)

print("Epochs completed:", len(history))
print("Best epoch:", best_epoch)
print("Best AP validation loss:", round(best_val_loss, 5))

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
    "Val RCF Dice:",
    round(best_record["val_RCF_Dice"], 4)
)

print(
    "Val AP Negative FP Rate:",
    round(best_record["val_negative_FP_rate"], 4)
)

print("\nSaved files:")
print("Best model:", M6_BEST)
print("Last checkpoint:", M6_LAST)
print("Training history:", M6_HISTORY)

print("\nOriginal PRAD dataset: NOT MODIFIED")
print("M1-M5 checkpoints: NOT MODIFIED")
print("Held-out test set: NOT USED")