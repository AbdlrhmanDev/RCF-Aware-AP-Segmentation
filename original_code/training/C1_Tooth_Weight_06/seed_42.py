# ORIGINAL SOURCE CELL 73 FROM USER TRAINING NOTEBOOK; no scientific logic modified.
# ============================================================
# CONTROL C1 — LOSS-MATCHED AUXILIARY SUPERVISION
#
# DeepLabV3+ | AP + Tooth
# Tooth loss weight = 0.60
# Seed = 42
#
# Compare against:
# M5: AP + 0.30 Tooth
# M6: AP + 0.30 Tooth + 0.30 RCF
#
# Independent ImageNet initialization
# Safe epoch-level resume
# Validation only — NO TEST
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
    Dataset, DataLoader, WeightedRandomSampler,
    SequentialSampler
)

# ============================================================
# 1. ENVIRONMENT
# ============================================================

for name in [
    "REAL_MYDRIVE", "train_dataset",
    "val_dataset", "val_loader"
]:
    assert name in globals(), f"Missing: {name}"

assert torch.cuda.is_available()
assert smp.__version__ == "0.5.0"

assert len(train_dataset) == 3466
assert len(val_dataset) == 743
assert int(train_dataset.df["AP"].sum()) == 460
assert int(val_dataset.df["AP"].sum()) == 98
assert isinstance(val_loader.sampler, SequentialSampler)

DEVICE = torch.device("cuda")
SEED = 42

ROOT = (
    Path(REAL_MYDRIVE)
    / "RCF_Aware_AP_Segmentation_Research"
)

MODELS_DIR = ROOT / "04_Models"

SOURCE_DIR = (
    MODELS_DIR
    / "M05_DeepLabV3Plus_AP_Tooth_Seed42"
)

RUN_DIR = (
    MODELS_DIR
    / "C01_DeepLabV3Plus_AP_Tooth_AuxW06_Seed42"
)

# ============================================================
# 2. CONFIGURATION
# ============================================================

with open(SOURCE_DIR / "training_config.json") as f:
    original = json.load(f)

assert int(original["seed"]) == 42
assert original["model"] == "DeepLabV3Plus"
assert original["encoder"] == "resnet18"
assert original["encoder_weights"] == "imagenet"
assert original["tasks"] == ["AP", "Tooth"]
assert int(original["output_channels"]) == 2
assert int(original["encoder_output_stride"]) == 16

CONFIG = dict(original)

CONFIG["tooth_weight"] = 0.6
CONFIG["control_type"] = "nominal_auxiliary_weight_matched"
CONFIG["control_reference"] = "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed42"

assert all(
    CONFIG[k] == original[k]
    for k in original
    if k != "tooth_weight"
)

BATCH_SIZE = int(CONFIG["batch_size"])
MAX_EPOCHS = int(CONFIG["max_epochs"])
PATIENCE = int(CONFIG["patience"])

LR = float(CONFIG["learning_rate"])
WD = float(CONFIG["weight_decay"])
TOOTH_WEIGHT = float(CONFIG["tooth_weight"])
AP_FRACTION = float(
    CONFIG["positive_sampling_fraction"]
)

assert BATCH_SIZE == 4
assert MAX_EPOCHS == 25
assert PATIENCE == 6
assert LR == 1e-4
assert WD == 1e-4
assert TOOTH_WEIGHT == 0.6
assert AP_FRACTION == 0.4

# ============================================================
# 3. SAFE STORAGE
# ============================================================

RUN_DIR.mkdir(parents=True, exist_ok=True)

BEST = RUN_DIR / "best_model.pt"
LAST = RUN_DIR / "last_checkpoint.pt"
HISTORY_FILE = RUN_DIR / "training_history.csv"
CONFIG_FILE = RUN_DIR / "training_config.json"

if CONFIG_FILE.exists():

    with open(CONFIG_FILE) as f:
        assert json.load(f) == CONFIG

else:

    assert not any(
        x.exists()
        for x in [BEST, LAST, HISTORY_FILE]
    )

    temp = CONFIG_FILE.with_suffix(".tmp")

    with open(temp, "w") as f:
        json.dump(CONFIG, f, indent=2)

    os.replace(temp, CONFIG_FILE)

if BEST.exists() != LAST.exists():
    raise RuntimeError("Incomplete checkpoint pair.")

if HISTORY_FILE.exists() and not LAST.exists():
    raise RuntimeError("History without checkpoint.")

# ============================================================
# 4. CLEAN GPU AND SET SEED
# ============================================================

for name in [
    "model", "optimizer", "scheduler", "scaler",
    "images", "outputs", "ap", "tooth", "rcf",
    "loss", "probe", "probe_loss"
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
# 5. DATA AUGMENTATION
# ============================================================

class ControlC1Dataset(Dataset):

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):

        sample = self.base[index]

        image = sample["image"].clone()
        ap = sample["AP_mask"].clone()
        tooth = sample["Tooth_mask"].clone()

        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            ap = torch.flip(ap, dims=[2])
            tooth = torch.flip(tooth, dims=[2])

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


augmented_dataset = ControlC1Dataset(train_dataset)

# ============================================================
# 6. AP-AWARE SAMPLER
# ============================================================

labels = (
    train_dataset.df["AP"]
    .astype(int).to_numpy()
)

n_pos = int((labels == 1).sum())
n_neg = int((labels == 0).sum())

assert n_pos == 460
assert n_neg == 3006

weights = np.where(
    labels == 1,
    AP_FRACTION / n_pos,
    (1 - AP_FRACTION) / n_neg
)

weights = torch.tensor(
    weights, dtype=torch.double
)


def build_control_loader(epoch):

    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator
    )

    return DataLoader(
        augmented_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

# ============================================================
# 7. INDEPENDENT MODEL
# ============================================================

model = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights=None if LAST.exists() else "imagenet",
    encoder_output_stride=16,
    in_channels=3,
    classes=2,
    activation=None
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WD
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

scaler = torch.amp.GradScaler("cuda")

# ============================================================
# 8. IDENTICAL LOSS DEFINITION
# ============================================================

def control_loss_parts(logits, targets):

    logits = logits.float()
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).mean(dim=(1, 2, 3))

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    pred_area = probabilities.sum(dim=(1, 2, 3))
    gt_area = targets.sum(dim=(1, 2, 3))

    dice_loss = 1.0 - (
        (2 * intersection + 1)
        / (pred_area + gt_area + 1)
    )

    return bce, dice_loss, gt_area > 0


def control_task_loss(logits, targets):

    bce, dice_loss, positive = control_loss_parts(
        logits, targets
    )

    result = bce.mean()

    if positive.any():
        result = result + dice_loss[positive].mean()

    return result


def control_training_loss(outputs, ap, tooth):

    ap_loss = control_task_loss(
        outputs[:, 0:1], ap
    )

    tooth_loss = control_task_loss(
        outputs[:, 1:2], tooth
    )

    total = ap_loss + 0.6 * tooth_loss

    return total, ap_loss, tooth_loss

# ============================================================
# 9. VALIDATION
# ============================================================

@torch.inference_mode()
def validate_control(model, loader):

    model.eval()

    bce_sum = 0.0
    soft_dice_sum = 0.0

    image_count = 0
    ap_positive = 0
    ap_negative = 0
    ap_negative_fp = 0

    ap_dices = []
    ap_ious = []
    tooth_dices = []

    for batch in loader:

        images = batch["image"].to(DEVICE)
        ap_gt = batch["AP_mask"].to(DEVICE)
        tooth_gt = batch["Tooth_mask"].to(DEVICE)

        with torch.amp.autocast("cuda"):
            outputs = model(images)

        ap_logits = outputs[:, 0:1].float()
        tooth_logits = outputs[:, 1:2].float()

        bce, soft_dice, positive = control_loss_parts(
            ap_logits, ap_gt
        )

        bce_sum += bce.sum().item()
        soft_dice_sum += soft_dice[positive].sum().item()

        ap_pred = ap_logits >= 0
        tooth_pred = tooth_logits >= 0

        for i in range(images.shape[0]):

            gt = ap_gt[i, 0] >= 0.5
            pred = ap_pred[i, 0]

            gt_pixels = int(gt.sum().item())
            pred_pixels = int(pred.sum().item())

            if gt_pixels:

                intersection = int(
                    (gt & pred).sum().item()
                )

                union = (
                    gt_pixels + pred_pixels
                    - intersection
                )

                ap_dices.append(
                    2 * intersection
                    / (gt_pixels + pred_pixels)
                )

                ap_ious.append(
                    intersection / union
                )

                ap_positive += 1

            else:

                ap_negative += 1
                ap_negative_fp += int(pred_pixels > 0)

            t_gt = tooth_gt[i, 0] >= 0.5
            t_pred = tooth_pred[i, 0]

            t_gt_pixels = int(t_gt.sum().item())

            if t_gt_pixels:

                t_pred_pixels = int(t_pred.sum().item())

                t_intersection = int(
                    (t_gt & t_pred).sum().item()
                )

                tooth_dices.append(
                    2 * t_intersection
                    / (t_gt_pixels + t_pred_pixels)
                )

        image_count += images.shape[0]

    assert image_count == 743
    assert ap_positive == 98
    assert ap_negative == 645
    assert len(tooth_dices) == 742

    val_loss = (
        bce_sum / 743
        + soft_dice_sum / 98
    )

    return {
        "val_loss": float(val_loss),
        "val_AP_Dice": float(np.mean(ap_dices)),
        "val_AP_IoU": float(np.mean(ap_ious)),
        "val_Tooth_Dice": float(np.mean(tooth_dices)),
        "val_negative_FP_rate": (
            ap_negative_fp / ap_negative
        ),
        "val_positive_images": ap_positive,
        "val_negative_images": ap_negative,
        "val_Tooth_positive_images": len(tooth_dices)
    }

# ============================================================
# 10. ATOMIC SAVING
# ============================================================

def control_save_checkpoint(state, path):

    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def control_save_history(records):

    temp = HISTORY_FILE.with_suffix(".tmp")

    pd.DataFrame(records).to_csv(
        temp, index=False
    )

    os.replace(temp, HISTORY_FILE)

# ============================================================
# 11. RESUME
# ============================================================

start_epoch = 1
best_epoch = 0
best_val_loss = float("inf")
bad_epochs = 0
history = []

if LAST.exists():

    saved = torch.load(
        LAST,
        map_location="cpu",
        weights_only=True
    )

    assert saved["config"] == CONFIG

    model.load_state_dict(
        saved["model_state"],
        strict=True
    )

    optimizer.load_state_dict(saved["optimizer_state"])
    scheduler.load_state_dict(saved["scheduler_state"])
    scaler.load_state_dict(saved["scaler_state"])

    best_epoch = int(saved["best_epoch"])
    best_val_loss = float(saved["best_val_loss"])
    bad_epochs = int(saved["bad_epochs"])
    history = list(saved["history"])

    start_epoch = int(saved["epoch"]) + 1

    assert [
        int(row["epoch"]) for row in history
    ] == list(range(1, start_epoch))

    best_saved = torch.load(
        BEST,
        map_location="cpu",
        weights_only=True
    )

    assert best_saved["config"] == CONFIG
    assert int(best_saved["epoch"]) == best_epoch

    del saved, best_saved

    print("Control C1 Seed42 checkpoint restored.")

else:

    print("New independent Control C1 Seed42 experiment.")

# ============================================================
# 12. PREFLIGHT
# ============================================================

print("\n" + "=" * 65)
print("CONTROL C1 SEED42 — PRE-TRAINING CHECK")
print("=" * 65)

print("GPU:", torch.cuda.get_device_name(0))
print("Model: DeepLabV3+ ResNet18")
print("Tasks: AP + Tooth")
print("Auxiliary Tooth weight:", TOOTH_WEIGHT)
print("Starting epoch:", start_epoch)

model.eval()

probe = next(iter(build_control_loader(start_epoch)))

x = probe["image"].to(DEVICE)
a = probe["AP_mask"].to(DEVICE)
t = probe["Tooth_mask"].to(DEVICE)

optimizer.zero_grad(set_to_none=True)

with torch.amp.autocast("cuda"):

    y = model(x)
    probe_loss, _, _ = control_training_loss(y, a, t)

assert y.shape == (4, 2, 512, 512)
assert torch.isfinite(probe_loss).item()

probe_loss.backward()
optimizer.zero_grad(set_to_none=True)

del probe, x, a, t, y, probe_loss

gc.collect()
torch.cuda.empty_cache()

print("Batch-4 forward/backward: PASSED")

preflight = validate_control(model, val_loader)

assert np.isfinite(preflight["val_loss"])

print("Full validation preflight: PASSED")
print("AP-positive:", preflight["val_positive_images"])
print("Tooth-positive:", preflight["val_Tooth_positive_images"])

if history:

    saved_loss = float(history[-1]["val_loss"])

    assert abs(
        preflight["val_loss"] - saved_loss
    ) < 0.002, "Resumed validation mismatch."

    print("Restored checkpoint validation: PASSED")

print("CONTROL C1 SEED42 PRE-TRAINING CHECK: PASSED")

# ============================================================
# 13. TRAINING LOOP
# ============================================================

print("\n" + "=" * 65)
print("CONTROL C1 | DEEPLABV3+ | SEED42 TRAINING")
print("=" * 65)

print("Maximum epochs:", MAX_EPOCHS)
print("Starting epoch:", start_epoch)
print("Output directory:", RUN_DIR)

for epoch in range(start_epoch, MAX_EPOCHS + 1):

    if bad_epochs >= PATIENCE:
        print("Early stopping already reached.")
        break

    epoch_start = time.time()

    random.seed(SEED + epoch)
    np.random.seed(SEED + epoch)
    torch.manual_seed(SEED + epoch)

    loader = build_control_loader(epoch)

    model.train()

    running_total = 0.0
    running_ap = 0.0
    running_tooth = 0.0
    seen = 0

    print(
        f"\nStarting Control C1 Seed42 Epoch "
        f"{epoch}/{MAX_EPOCHS}",
        flush=True
    )

    for batch_index, batch in enumerate(
        loader, start=1
    ):

        images = batch["image"].to(DEVICE)
        ap = batch["AP_mask"].to(DEVICE)
        tooth = batch["Tooth_mask"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            loss, ap_loss, tooth_loss = (
                control_training_loss(
                    outputs, ap, tooth
                )
            )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        bs = images.shape[0]
        seen += bs

        running_total += loss.item() * bs
        running_ap += ap_loss.item() * bs
        running_tooth += tooth_loss.item() * bs

        if (
            batch_index % 250 == 0
            or batch_index == len(loader)
        ):

            print(
                f"Epoch {epoch}: "
                f"Batch {batch_index}/{len(loader)}",
                flush=True
            )

    assert seen == 3466

    metrics = validate_control(
        model, val_loader
    )

    assert np.isfinite(metrics["val_loss"])

    scheduler.step(metrics["val_loss"])

    current_lr = optimizer.param_groups[0]["lr"]

    record = {
        "epoch": epoch,
        "train_loss": running_total / seen,
        "train_AP_loss": running_ap / seen,
        "train_Tooth_loss": running_tooth / seen,
        **metrics,
        "learning_rate": current_lr,
        "epoch_minutes": (
            time.time() - epoch_start
        ) / 60
    }

    # Best checkpoint chosen by AP validation loss
    improved = (
        metrics["val_loss"]
        < best_val_loss - 1e-4
    )

    if improved:

        best_val_loss = float(metrics["val_loss"])
        best_epoch = epoch
        bad_epochs = 0

        control_save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "config": CONFIG
            },
            BEST
        )

    else:

        bad_epochs += 1

    history.append(record)

    control_save_checkpoint(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
            "config": CONFIG
        },
        LAST
    )

    control_save_history(history)

    print("\n" + "-" * 65)
    print(f"CONTROL C1 SEED42 — EPOCH {epoch}/{MAX_EPOCHS}")
    print("-" * 65)

    print(f"Train Total Loss: {record['train_loss']:.4f}")
    print(f"Val AP Loss: {metrics['val_loss']:.4f}")
    print(f"Val AP Dice: {metrics['val_AP_Dice']:.4f}")
    print(f"Val AP IoU: {metrics['val_AP_IoU']:.4f}")
    print(f"Val Tooth Dice: {metrics['val_Tooth_Dice']:.4f}")

    print(
        "Val AP Negative FP Rate:",
        f"{metrics['val_negative_FP_rate']:.4f}"
    )

    print(f"Learning Rate: {current_lr:.7f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Early Stopping: {bad_epochs}/{PATIENCE}")

    if improved:
        print("BEST MODEL UPDATED")

    print("CHECKPOINT SAVED")
    print("-" * 65)

    if bad_epochs >= PATIENCE:
        print("\nEARLY STOPPING ACTIVATED")
        break

# ============================================================
# 14. FINAL REPORT
# ============================================================

assert BEST.is_file()
assert LAST.is_file()
assert HISTORY_FILE.is_file()

last_saved = torch.load(
    LAST,
    map_location="cpu",
    weights_only=True
)

best_saved = torch.load(
    BEST,
    map_location="cpu",
    weights_only=True
)

assert last_saved["config"] == CONFIG
assert best_saved["config"] == CONFIG
assert int(best_saved["epoch"]) == best_epoch
assert int(last_saved["epoch"]) == history[-1]["epoch"]

best_record = next(
    r for r in history
    if int(r["epoch"]) == best_epoch
)

print("\n" + "=" * 65)
print("CONTROL C1 SEED42 TRAINING FINISHED")
print("=" * 65)

print("Seed:", SEED)
print("Tooth loss weight:", TOOTH_WEIGHT)
print("Epochs completed:", len(history))
print("Best epoch:", best_epoch)
print("Best AP validation loss:", round(best_val_loss, 5))

print("\nBEST CHECKPOINT METRICS")
print("Val AP Dice:", round(best_record["val_AP_Dice"], 4))
print("Val AP IoU:", round(best_record["val_AP_IoU"], 4))
print("Val Tooth Dice:", round(best_record["val_Tooth_Dice"], 4))
print(
    "Val AP Negative FP Rate:",
    round(best_record["val_negative_FP_rate"], 4)
)

print("\nSaved files:")
print("Best:", BEST)
print("Last:", LAST)
print("History:", HISTORY_FILE)

print("\nOriginal M2/M4/M5/M6: NOT MODIFIED")
print("Original PRAD dataset: NOT MODIFIED")
print("Held-out Test set: NOT USED")

gc.collect()
torch.cuda.empty_cache()