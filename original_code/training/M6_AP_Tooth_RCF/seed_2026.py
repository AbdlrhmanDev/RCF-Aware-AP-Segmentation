# ORIGINAL SOURCE CELL 69 FROM USER TRAINING NOTEBOOK; no scientific logic modified.
# ============================================================
# STAGE 3 — MULTI-SEED REPLICATION
# M6: DeepLabV3+ ResNet18
# Tasks: AP + Tooth + RCF
# Seed: 2026
#
# Full Colab training + safe checkpoint resume
# Fixed validation protocol
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
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    SequentialSampler
)

# ============================================================
# 1. ENVIRONMENT VERIFICATION
# ============================================================

required = [
    "REAL_MYDRIVE",
    "train_dataset",
    "val_dataset",
    "val_loader"
]

missing = [x for x in required if x not in globals()]

if missing:
    raise RuntimeError(
        f"Missing variables: {missing}. "
        "Restore dataset environment first."
    )

assert torch.cuda.is_available(), "Enable Colab GPU."
assert smp.__version__ == "0.5.0"

assert len(train_dataset) == 3466
assert len(val_dataset) == 743

assert int(train_dataset.df["AP"].sum()) == 460
assert int(val_dataset.df["AP"].sum()) == 98
assert int(val_dataset.df["RCF"].sum()) == 288

assert isinstance(
    val_loader.sampler,
    SequentialSampler
)

DEVICE = torch.device("cuda")
SEED = 2026

print("=" * 65)
print("M6 SEED2026 — ENVIRONMENT VERIFIED")
print("=" * 65)

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("SMP:", smp.__version__)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ============================================================
# 2. LOAD FROZEN M6 CONFIGURATION
# ============================================================

ROOT = (
    Path(REAL_MYDRIVE)
    / "RCF_Aware_AP_Segmentation_Research"
)

MODELS_DIR = ROOT / "04_Models"

SOURCE_DIR = (
    MODELS_DIR
    / "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed42"
)

RUN_DIR = (
    MODELS_DIR
    / "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed2026"
)

SOURCE_CONFIG = SOURCE_DIR / "training_config.json"

assert SOURCE_CONFIG.is_file()

with open(SOURCE_CONFIG) as f:
    original_config = json.load(f)

assert int(original_config["seed"]) == 42
assert original_config["model"] == "DeepLabV3Plus"
assert original_config["encoder"] == "resnet18"
assert original_config["encoder_weights"] == "imagenet"

assert original_config["tasks"] == [
    "AP", "Tooth", "RCF"
]

assert int(original_config["output_channels"]) == 3
assert int(original_config["image_size"]) == 512
assert int(original_config["encoder_output_stride"]) == 16

assert int(original_config["AP_channel"]) == 0
assert int(original_config["Tooth_channel"]) == 1
assert int(original_config["RCF_channel"]) == 2

CONFIG = dict(original_config)
CONFIG["seed"] = SEED

assert {
    k: v for k, v in CONFIG.items()
    if k != "seed"
} == {
    k: v for k, v in original_config.items()
    if k != "seed"
}

BATCH_SIZE = int(CONFIG["batch_size"])
MAX_EPOCHS = int(CONFIG["max_epochs"])
PATIENCE = int(CONFIG["patience"])

LR = float(CONFIG["learning_rate"])
WD = float(CONFIG["weight_decay"])

TOOTH_WEIGHT = float(CONFIG["tooth_weight"])
RCF_WEIGHT = float(CONFIG["rcf_weight"])

AP_FRACTION = float(
    CONFIG["positive_sampling_fraction"]
)

assert BATCH_SIZE == 4
assert MAX_EPOCHS == 25
assert PATIENCE == 6

assert LR == 1e-4
assert WD == 1e-4

assert TOOTH_WEIGHT == 0.3
assert RCF_WEIGHT == 0.3
assert AP_FRACTION == 0.4

# ============================================================
# 3. SAFE EXPERIMENT STORAGE
# ============================================================

RUN_DIR.mkdir(parents=True, exist_ok=True)

BEST = RUN_DIR / "best_model.pt"
LAST = RUN_DIR / "last_checkpoint.pt"

HISTORY_FILE = RUN_DIR / "training_history.csv"
CONFIG_FILE = RUN_DIR / "training_config.json"

if CONFIG_FILE.exists():

    with open(CONFIG_FILE) as f:
        existing_config = json.load(f)

    assert existing_config == CONFIG, (
        "Experiment configuration mismatch."
    )

else:

    assert not any(
        p.exists()
        for p in [BEST, LAST, HISTORY_FILE]
    ), "Existing files found without configuration."

    temp = CONFIG_FILE.with_suffix(".tmp")

    with open(temp, "w") as f:
        json.dump(CONFIG, f, indent=2)

    os.replace(temp, CONFIG_FILE)

if BEST.exists() != LAST.exists():

    raise RuntimeError(
        "Incomplete checkpoint pair. "
        "Inspect the experiment before resuming."
    )

if HISTORY_FILE.exists() and not LAST.exists():

    raise RuntimeError(
        "Training history exists without checkpoint."
    )

# ============================================================
# 4. CLEAN PREVIOUS GPU OBJECTS
# ============================================================

for name in [
    "model", "optimizer", "scheduler", "scaler",
    "images", "outputs", "ap", "tooth", "rcf",
    "loss", "ap_loss", "tooth_loss", "rcf_loss",
    "probe", "probe_loss"
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
# 5. FROZEN TRAINING AUGMENTATION
# ============================================================

class M6Seed2026Dataset(Dataset):

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

        # Mild intensity augmentation
        if random.random() < 0.5:

            contrast = random.uniform(0.95, 1.05)
            brightness = random.uniform(-0.08, 0.08)

            image = torch.clamp(
                image * contrast + brightness,
                -3.0,
                3.0
            )

        return {
            "image": image,
            "AP_mask": ap,
            "Tooth_mask": tooth,
            "RCF_mask": rcf
        }


augmented_dataset = M6Seed2026Dataset(
    train_dataset
)

# ============================================================
# 6. AP-AWARE SAMPLING
# ============================================================

labels = (
    train_dataset.df["AP"]
    .astype(int)
    .to_numpy()
)

n_pos = int((labels == 1).sum())
n_neg = int((labels == 0).sum())

assert n_pos == 460
assert n_neg == 3006

sample_weights = np.where(
    labels == 1,
    AP_FRACTION / n_pos,
    (1.0 - AP_FRACTION) / n_neg
)

sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.double
)


def build_train_loader(epoch):

    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
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
# 7. CREATE MODEL
# ============================================================

# New experiment:
# ResNet18 ImageNet initialization.
#
# Resume:
# Restore M6 Seed2026 weights from its checkpoint.
#
# Never load trained Seed42/Seed123 checkpoints.

encoder_weights = (
    None if LAST.exists() else "imagenet"
)

model = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights=encoder_weights,
    encoder_output_stride=16,
    in_channels=3,
    classes=3,
    activation=None
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WD
)

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )
)

scaler = torch.amp.GradScaler("cuda")

# ============================================================
# 8. ORIGINAL LOSS FUNCTION
# ============================================================

def loss_parts(logits, targets):

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

    dice_loss = 1.0 - (
        (2.0 * intersection + 1.0)
        / (predicted_area + target_area + 1.0)
    )

    positive = target_area > 0

    return bce, dice_loss, positive


def task_loss(logits, targets):

    bce, dice_loss, positive = loss_parts(
        logits,
        targets
    )

    result = bce.mean()

    if positive.any():

        result = (
            result
            + dice_loss[positive].mean()
        )

    return result


def training_loss(outputs, ap, tooth, rcf):

    ap_loss = task_loss(
        outputs[:, 0:1], ap
    )

    tooth_loss = task_loss(
        outputs[:, 1:2], tooth
    )

    rcf_loss = task_loss(
        outputs[:, 2:3], rcf
    )

    total = (
        ap_loss
        + TOOTH_WEIGHT * tooth_loss
        + RCF_WEIGHT * rcf_loss
    )

    return total, ap_loss, tooth_loss, rcf_loss

# ============================================================
# 9. FULL VALIDATION — 743 IMAGES
# ============================================================

@torch.inference_mode()
def validate_m6_2026(model, loader):

    model.eval()

    bce_sum = 0.0
    soft_dice_sum = 0.0

    image_count = 0

    ap_positive = 0
    ap_negative = 0
    ap_negative_fp = 0

    rcf_negative = 0

    ap_dices = []
    ap_ious = []
    tooth_dices = []
    rcf_dices = []

    for batch in loader:

        images = batch["image"].to(DEVICE)

        ap_gt = batch["AP_mask"].to(DEVICE)
        tooth_gt = batch["Tooth_mask"].to(DEVICE)
        rcf_gt = batch["RCF_mask"].to(DEVICE)

        with torch.amp.autocast("cuda"):
            outputs = model(images)

        ap_logits = outputs[:, 0:1].float()

        bce, soft_dice, positive = loss_parts(
            ap_logits,
            ap_gt
        )

        bce_sum += bce.sum().item()
        soft_dice_sum += (
            soft_dice[positive].sum().item()
        )

        predictions = outputs.float() >= 0

        targets = [
            ap_gt >= 0.5,
            tooth_gt >= 0.5,
            rcf_gt >= 0.5
        ]

        for i in range(images.shape[0]):

            for task in range(3):

                pred = predictions[i, task]
                gt = targets[task][i, 0]

                gt_pixels = int(gt.sum().item())
                pred_pixels = int(pred.sum().item())

                if gt_pixels > 0:

                    intersection = int(
                        (pred & gt).sum().item()
                    )

                    dice = (
                        2.0 * intersection
                        / (gt_pixels + pred_pixels)
                    )

                    if task == 0:

                        union = (
                            gt_pixels
                            + pred_pixels
                            - intersection
                        )

                        ap_dices.append(dice)

                        ap_ious.append(
                            intersection / union
                        )

                        ap_positive += 1

                    elif task == 1:

                        tooth_dices.append(dice)

                    else:

                        rcf_dices.append(dice)

                else:

                    if task == 0:

                        ap_negative += 1
                        ap_negative_fp += int(
                            pred_pixels > 0
                        )

                    elif task == 2:

                        rcf_negative += 1

        image_count += images.shape[0]

    # --------------------------------------------------------
    # Validate dataset distribution
    # --------------------------------------------------------

    assert image_count == 743
    assert ap_positive == 98
    assert ap_negative == 645

    assert len(tooth_dices) == 742
    assert len(rcf_dices) == 287
    assert rcf_negative == 456

    # Frozen checkpoint criterion:
    # AP BCE across all validation images
    # + AP soft Dice loss on positive images

    val_loss = (
        bce_sum / image_count
        + soft_dice_sum / ap_positive
    )

    return {
        "val_loss": float(val_loss),

        "val_AP_Dice": float(
            np.mean(ap_dices)
        ),

        "val_AP_IoU": float(
            np.mean(ap_ious)
        ),

        "val_Tooth_Dice": float(
            np.mean(tooth_dices)
        ),

        "val_RCF_Dice": float(
            np.mean(rcf_dices)
        ),

        "val_negative_FP_rate": (
            ap_negative_fp / ap_negative
        ),

        "val_positive_images": ap_positive,
        "val_negative_images": ap_negative,

        "val_Tooth_positive_images": len(
            tooth_dices
        ),

        "val_RCF_positive_images": len(
            rcf_dices
        )
    }

# ============================================================
# 10. SAFE CHECKPOINT SAVING
# ============================================================

def save_checkpoint(state, path):

    temporary = path.with_suffix(".tmp")

    torch.save(state, temporary)

    os.replace(temporary, path)


def save_history(records):

    temporary = HISTORY_FILE.with_suffix(".tmp")

    pd.DataFrame(records).to_csv(
        temporary,
        index=False
    )

    os.replace(temporary, HISTORY_FILE)

# ============================================================
# 11. RESTORE PREVIOUS SEED2026 CHECKPOINT
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

    optimizer.load_state_dict(
        saved["optimizer_state"]
    )

    scheduler.load_state_dict(
        saved["scheduler_state"]
    )

    scaler.load_state_dict(
        saved["scaler_state"]
    )

    best_epoch = int(saved["best_epoch"])
    best_val_loss = float(saved["best_val_loss"])

    bad_epochs = int(saved["bad_epochs"])
    history = list(saved["history"])

    start_epoch = int(saved["epoch"]) + 1

    assert [
        int(row["epoch"])
        for row in history
    ] == list(range(1, start_epoch))

    best_saved = torch.load(
        BEST,
        map_location="cpu",
        weights_only=True
    )

    assert best_saved["config"] == CONFIG

    assert int(
        best_saved["epoch"]
    ) == best_epoch

    del saved, best_saved

    print(
        "Previous M6 Seed2026 checkpoint restored."
    )

else:

    print(
        "New independent M6 Seed2026 experiment."
    )

# ============================================================
# 12. PRE-TRAINING SAFETY CHECK
# ============================================================

print("\n" + "=" * 65)
print("M6 SEED2026 — PRE-TRAINING SAFETY CHECK")
print("=" * 65)

print("GPU:", torch.cuda.get_device_name(0))
print("Model: DeepLabV3+ ResNet18")
print("Tasks: AP + Tooth + RCF")
print("Seed:", SEED)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Starting epoch:", start_epoch)
print("Best previous epoch:", best_epoch)
print("Early stopping:", f"{bad_epochs}/{PATIENCE}")

# ------------------------------------------------------------
# Batch-4 forward/backward check
# No optimizer update
# ------------------------------------------------------------

model.eval()

probe = next(iter(build_train_loader(start_epoch)))

images = probe["image"].to(DEVICE)
ap = probe["AP_mask"].to(DEVICE)
tooth = probe["Tooth_mask"].to(DEVICE)
rcf = probe["RCF_mask"].to(DEVICE)

optimizer.zero_grad(set_to_none=True)

with torch.amp.autocast("cuda"):

    outputs = model(images)

    probe_loss, _, _, _ = training_loss(
        outputs,
        ap,
        tooth,
        rcf
    )

assert outputs.shape == (4, 3, 512, 512)
assert torch.isfinite(probe_loss).item()

probe_loss.backward()

optimizer.zero_grad(set_to_none=True)

del probe
del images, ap, tooth, rcf
del outputs, probe_loss

gc.collect()
torch.cuda.empty_cache()

print("Batch-4 forward/backward: PASSED")

# ------------------------------------------------------------
# Full validation preflight
# ------------------------------------------------------------

preflight = validate_m6_2026(
    model,
    val_loader
)

assert np.isfinite(preflight["val_loss"])

assert preflight["val_positive_images"] == 98
assert preflight["val_negative_images"] == 645
assert preflight["val_Tooth_positive_images"] == 742
assert preflight["val_RCF_positive_images"] == 287

print("Full validation preflight: PASSED")

print(
    "AP-positive:",
    preflight["val_positive_images"]
)

print(
    "Tooth-positive:",
    preflight["val_Tooth_positive_images"]
)

print(
    "RCF-positive:",
    preflight["val_RCF_positive_images"]
)

# ------------------------------------------------------------
# Validate restored checkpoint, if any
# ------------------------------------------------------------

if history:

    previous_loss = float(
        history[-1]["val_loss"]
    )

    recomputed_loss = float(
        preflight["val_loss"]
    )

    difference = abs(
        previous_loss - recomputed_loss
    )

    print(
        "Saved last AP loss:",
        round(previous_loss, 5)
    )

    print(
        "Recomputed AP loss:",
        round(recomputed_loss, 5)
    )

    if difference > 0.002:

        raise RuntimeError(
            "Restored checkpoint validation mismatch. "
            "Inspect preprocessing before resuming."
        )

    print("Restored validation: PASSED")

print("\nM6 SEED2026 PRE-TRAINING CHECK: PASSED")

# ============================================================
# 13. MAIN TRAINING LOOP
# ============================================================

print("\n" + "=" * 65)
print("M6 — DEEPLABV3+ AP + TOOTH + RCF")
print("SEED2026 TRAINING")
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

    loader = build_train_loader(epoch)

    model.train()

    running_total = 0.0
    running_ap = 0.0
    running_tooth = 0.0
    running_rcf = 0.0

    seen = 0

    total_batches = len(loader)

    print(
        f"\nStarting M6 Seed2026 Epoch "
        f"{epoch}/{MAX_EPOCHS}",
        flush=True
    )

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        ap = batch["AP_mask"].to(
            DEVICE,
            non_blocking=True
        )

        tooth = batch["Tooth_mask"].to(
            DEVICE,
            non_blocking=True
        )

        rcf = batch["RCF_mask"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            (
                loss,
                ap_loss,
                tooth_loss,
                rcf_loss
            ) = training_loss(
                outputs,
                ap,
                tooth,
                rcf
            )

        if not torch.isfinite(loss).item():

            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, "
                f"batch {batch_index}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        bs = images.shape[0]
        seen += bs

        running_total += loss.item() * bs
        running_ap += ap_loss.item() * bs
        running_tooth += tooth_loss.item() * bs
        running_rcf += rcf_loss.item() * bs

        if (
            batch_index % 250 == 0
            or batch_index == total_batches
        ):

            print(
                f"Epoch {epoch}: "
                f"Batch {batch_index}/{total_batches}",
                flush=True
            )

    assert seen == 3466

    train_loss = running_total / seen
    train_ap_loss = running_ap / seen
    train_tooth_loss = running_tooth / seen
    train_rcf_loss = running_rcf / seen

    # ========================================================
    # 14. VALIDATE FULL VALIDATION SET
    # ========================================================

    metrics = validate_m6_2026(
        model,
        val_loader
    )

    assert np.isfinite(metrics["val_loss"])

    assert metrics["val_positive_images"] == 98
    assert metrics["val_negative_images"] == 645
    assert metrics["val_RCF_positive_images"] == 287

    scheduler.step(metrics["val_loss"])

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    epoch_minutes = (
        time.time() - epoch_start
    ) / 60.0

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
    # 15. SELECT BEST CHECKPOINT
    # ========================================================

    improved = (
        metrics["val_loss"]
        < best_val_loss - 1e-4
    )

    if improved:

        best_val_loss = float(
            metrics["val_loss"]
        )

        best_epoch = epoch
        bad_epochs = 0

        save_checkpoint(
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

    # ========================================================
    # 16. SAVE RESUMABLE CHECKPOINT
    # ========================================================

    save_checkpoint(
        {
            "epoch": epoch,

            "model_state":
                model.state_dict(),

            "optimizer_state":
                optimizer.state_dict(),

            "scheduler_state":
                scheduler.state_dict(),

            "scaler_state":
                scaler.state_dict(),

            "best_val_loss":
                best_val_loss,

            "best_epoch":
                best_epoch,

            "bad_epochs":
                bad_epochs,

            "history":
                history,

            "config":
                CONFIG
        },
        LAST
    )

    save_history(history)

    # ========================================================
    # 17. EPOCH REPORT
    # ========================================================

    print("\n" + "-" * 65)
    print(f"M6 SEED2026 — EPOCH {epoch}/{MAX_EPOCHS}")
    print("-" * 65)

    print(f"Train Total Loss: {train_loss:.4f}")
    print(f"Train AP Loss: {train_ap_loss:.4f}")
    print(f"Train Tooth Loss: {train_tooth_loss:.4f}")
    print(f"Train RCF Loss: {train_rcf_loss:.4f}")

    print(f"Val AP Loss: {metrics['val_loss']:.4f}")
    print(f"Val AP Dice: {metrics['val_AP_Dice']:.4f}")
    print(f"Val AP IoU: {metrics['val_AP_IoU']:.4f}")

    print(
        "Val Tooth Dice:",
        f"{metrics['val_Tooth_Dice']:.4f}"
    )

    print(
        "Val RCF Dice:",
        f"{metrics['val_RCF_Dice']:.4f}"
    )

    print(
        "Val AP Negative FP Rate:",
        f"{metrics['val_negative_FP_rate']:.4f}"
    )

    print(f"Learning Rate: {current_lr:.7f}")
    print(f"Epoch Time: {epoch_minutes:.2f} minutes")
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
# 18. FINAL INTEGRITY CHECK
# ============================================================

assert BEST.is_file()
assert LAST.is_file()
assert HISTORY_FILE.is_file()

final_checkpoint = torch.load(
    LAST,
    map_location="cpu",
    weights_only=True
)

assert final_checkpoint["config"] == CONFIG

assert int(final_checkpoint["epoch"]) == int(
    history[-1]["epoch"]
)

assert len(final_checkpoint["history"]) == len(history)

best_checkpoint = torch.load(
    BEST,
    map_location="cpu",
    weights_only=True
)

assert best_checkpoint["config"] == CONFIG
assert int(best_checkpoint["epoch"]) == best_epoch

best_record = next(
    row for row in history
    if int(row["epoch"]) == best_epoch
)

# ============================================================
# 19. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("M6 SEED2026 TRAINING FINISHED")
print("=" * 65)

print("Seed:", SEED)
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
    "Val RCF Dice:",
    round(best_record["val_RCF_Dice"], 4)
)

print(
    "Val AP Negative FP Rate:",
    round(best_record["val_negative_FP_rate"], 4)
)

print("\nSaved files:")

print("Best:", BEST)
print("Last:", LAST)
print("History:", HISTORY_FILE)

print("\nSeed42 models: NOT MODIFIED")
print("Seed123 models: NOT MODIFIED")
print("M2/M4/M5 Seed2026: NOT MODIFIED")
print("Original PRAD dataset: NOT MODIFIED")
print("Held-out Test set: NOT USED")

gc.collect()
torch.cuda.empty_cache()