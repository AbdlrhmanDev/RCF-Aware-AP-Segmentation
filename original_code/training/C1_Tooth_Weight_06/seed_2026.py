# ORIGINAL SOURCE CELL 80 FROM USER TRAINING NOTEBOOK; no scientific logic modified.
# ============================================================
# CONTROL C1 — SEED 2026
# ============================================================
# Architecture: DeepLabV3+ / ResNet18
# Tasks: AP + Tooth
# Loss: AP + 0.60 * Tooth
#
# Frozen configuration from C1 Seed123
# Only seed and reference experiment change
#
# Independent model training
# Automatic checkpoint resume
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
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    SequentialSampler
)

# ============================================================
# 1. VERIFY ENVIRONMENT
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
        f"Missing environment variables: {missing}. "
        "Restore dataset setup before training."
    )

assert torch.cuda.is_available(), "Enable GPU in Colab."
assert smp.__version__ == "0.5.0"

assert len(train_dataset) == 3466
assert len(val_dataset) == 743

assert int(train_dataset.df["AP"].sum()) == 460
assert int(val_dataset.df["AP"].sum()) == 98

assert isinstance(
    val_loader.sampler,
    SequentialSampler
)

SEED = 2026
DEVICE = torch.device("cuda")

ROOT = (
    Path(REAL_MYDRIVE)
    / "RCF_Aware_AP_Segmentation_Research"
)

MODELS_DIR = ROOT / "04_Models"

C1_SOURCE = (
    MODELS_DIR
    / "C01_DeepLabV3Plus_AP_Tooth_AuxW06_Seed123"
)

M5_REFERENCE = (
    MODELS_DIR
    / "M05_DeepLabV3Plus_AP_Tooth_Seed2026"
)

RUN_DIR = (
    MODELS_DIR
    / "C01_DeepLabV3Plus_AP_Tooth_AuxW06_Seed2026"
)

print("=" * 70)
print("CONTROL C1 SEED2026 — ENVIRONMENT VERIFIED")
print("=" * 70)

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("SMP:", smp.__version__)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ============================================================
# 2. LOAD AND VERIFY FROZEN CONFIGURATION
# ============================================================

with open(
    C1_SOURCE / "training_config.json"
) as f:
    source_config = json.load(f)

with open(
    M5_REFERENCE / "training_config.json"
) as f:
    m5_config = json.load(f)

assert int(source_config["seed"]) == 123
assert int(m5_config["seed"]) == 2026

CONFIG = dict(source_config)

CONFIG["seed"] = SEED

CONFIG["control_reference"] = (
    "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed2026"
)

expected = dict(m5_config)

expected["tooth_weight"] = 0.6

expected["control_type"] = (
    "nominal_auxiliary_weight_matched"
)

expected["control_reference"] = (
    "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed2026"
)

assert CONFIG == expected, (
    "Configuration does not match the frozen "
    "C1 experiment specification."
)

assert CONFIG["model"] == "DeepLabV3Plus"
assert CONFIG["encoder"] == "resnet18"
assert CONFIG["encoder_weights"] == "imagenet"

assert CONFIG["tasks"] == ["AP", "Tooth"]
assert int(CONFIG["output_channels"]) == 2

assert int(CONFIG["image_size"]) == 512
assert int(CONFIG["encoder_output_stride"]) == 16

assert int(CONFIG["AP_channel"]) == 0
assert int(CONFIG["Tooth_channel"]) == 1

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

print("\nFrozen configuration: VERIFIED")
print("Seed:", SEED)
print("Tooth loss weight:", TOOTH_WEIGHT)

# ============================================================
# 3. CREATE INDEPENDENT EXPERIMENT DIRECTORY
# ============================================================

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST = RUN_DIR / "best_model.pt"
LAST = RUN_DIR / "last_checkpoint.pt"

HISTORY_FILE = (
    RUN_DIR / "training_history.csv"
)

CONFIG_FILE = (
    RUN_DIR / "training_config.json"
)

if CONFIG_FILE.exists():

    with open(CONFIG_FILE) as f:
        existing_config = json.load(f)

    assert existing_config == CONFIG, (
        "Existing C1 Seed2026 configuration mismatch."
    )

else:

    assert not any(
        path.exists()
        for path in [
            BEST,
            LAST,
            HISTORY_FILE
        ]
    ), "Unexpected experiment files."

    temporary = CONFIG_FILE.with_suffix(".tmp")

    with open(temporary, "w") as f:
        json.dump(CONFIG, f, indent=2)

    os.replace(temporary, CONFIG_FILE)

if BEST.exists() != LAST.exists():
    raise RuntimeError(
        "Incomplete checkpoint pair. "
        "Inspect existing files before resuming."
    )

if HISTORY_FILE.exists() and not LAST.exists():
    raise RuntimeError(
        "History exists without resumable checkpoint."
    )

# ============================================================
# 4. RELEASE PREVIOUS GPU OBJECTS
# ============================================================

for name in [
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "images",
    "outputs",
    "ap",
    "tooth",
    "rcf",
    "loss",
    "probe",
    "probe_loss"
]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 5. SET RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False

# ============================================================
# 6. FROZEN TRAINING AUGMENTATION
# ============================================================

class C1Seed2026Dataset(Dataset):

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):

        sample = self.base[idx]

        image = sample["image"].clone()
        ap = sample["AP_mask"].clone()
        tooth = sample["Tooth_mask"].clone()

        # Horizontal flip
        if random.random() < 0.5:

            image = torch.flip(
                image, dims=[2]
            )

            ap = torch.flip(
                ap, dims=[2]
            )

            tooth = torch.flip(
                tooth, dims=[2]
            )

        # Mild intensity augmentation
        if random.random() < 0.5:

            contrast = random.uniform(
                0.95, 1.05
            )

            brightness = random.uniform(
                -0.08, 0.08
            )

            image = torch.clamp(
                image * contrast + brightness,
                -3.0,
                3.0
            )

        return {
            "image": image,
            "AP_mask": ap,
            "Tooth_mask": tooth
        }


augmented_dataset = C1Seed2026Dataset(
    train_dataset
)

# ============================================================
# 7. AP-AWARE WEIGHTED SAMPLER
# ============================================================

labels = (
    train_dataset.df["AP"]
    .astype(int)
    .to_numpy()
)

n_positive = int(
    (labels == 1).sum()
)

n_negative = int(
    (labels == 0).sum()
)

assert n_positive == 460
assert n_negative == 3006

sample_weights = np.where(
    labels == 1,
    AP_FRACTION / n_positive,
    (1.0 - AP_FRACTION) / n_negative
)

sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.double
)


def build_c1_train_loader(epoch):

    generator = torch.Generator()

    generator.manual_seed(
        SEED + epoch
    )

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
# 8. BUILD INDEPENDENT MODEL
# ============================================================

model = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights=(
        None if LAST.exists()
        else "imagenet"
    ),
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
# 9. FROZEN LOSS FUNCTIONS
# ============================================================

def c1_2026_loss_parts(logits, targets):

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

    pred_area = probabilities.sum(
        dim=(1, 2, 3)
    )

    gt_area = targets.sum(
        dim=(1, 2, 3)
    )

    dice_loss = 1.0 - (
        (2.0 * intersection + 1.0)
        / (pred_area + gt_area + 1.0)
    )

    positive = gt_area > 0

    return bce, dice_loss, positive


def c1_2026_task_loss(logits, targets):

    bce, dice_loss, positive = (
        c1_2026_loss_parts(
            logits, targets
        )
    )

    result = bce.mean()

    if positive.any():

        result = (
            result
            + dice_loss[positive].mean()
        )

    return result


def c1_2026_training_loss(
    outputs, ap, tooth
):

    ap_loss = c1_2026_task_loss(
        outputs[:, 0:1],
        ap
    )

    tooth_loss = c1_2026_task_loss(
        outputs[:, 1:2],
        tooth
    )

    total_loss = (
        ap_loss
        + TOOTH_WEIGHT * tooth_loss
    )

    return total_loss, ap_loss, tooth_loss

# ============================================================
# 10. FULL VALIDATION
# ============================================================

@torch.inference_mode()
def validate_c1_2026(model, loader):

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

        tooth_logits = (
            outputs[:, 1:2].float()
        )

        bce, soft_dice, positive = (
            c1_2026_loss_parts(
                ap_logits,
                ap_gt
            )
        )

        bce_sum += bce.sum().item()

        soft_dice_sum += (
            soft_dice[positive].sum().item()
        )

        ap_pred = ap_logits >= 0
        tooth_pred = tooth_logits >= 0

        for i in range(images.shape[0]):

            gt = ap_gt[i, 0] >= 0.5
            pred = ap_pred[i, 0]

            gt_pixels = int(gt.sum().item())
            pred_pixels = int(pred.sum().item())

            if gt_pixels > 0:

                intersection = int(
                    (gt & pred).sum().item()
                )

                union = (
                    gt_pixels
                    + pred_pixels
                    - intersection
                )

                ap_dices.append(
                    2.0 * intersection
                    / (gt_pixels + pred_pixels)
                )

                ap_ious.append(
                    intersection / union
                )

                ap_positive += 1

            else:

                ap_negative += 1

                ap_negative_fp += int(
                    pred_pixels > 0
                )

            t_gt = tooth_gt[i, 0] >= 0.5
            t_pred = tooth_pred[i, 0]

            t_gt_pixels = int(
                t_gt.sum().item()
            )

            if t_gt_pixels > 0:

                t_pred_pixels = int(
                    t_pred.sum().item()
                )

                t_intersection = int(
                    (t_gt & t_pred).sum().item()
                )

                tooth_dices.append(
                    2.0 * t_intersection
                    / (t_gt_pixels + t_pred_pixels)
                )

        image_count += images.shape[0]

    assert image_count == 743

    assert ap_positive == 98
    assert ap_negative == 645

    assert len(ap_dices) == 98
    assert len(tooth_dices) == 742

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

        "val_negative_FP_rate": (
            ap_negative_fp / ap_negative
        ),

        "val_positive_images": ap_positive,
        "val_negative_images": ap_negative,

        "val_Tooth_positive_images": len(
            tooth_dices
        )
    }

# ============================================================
# 11. ATOMIC SAVING FUNCTIONS
# ============================================================

def save_c1_2026_checkpoint(state, path):

    temporary = path.with_suffix(".tmp")

    torch.save(
        state,
        temporary
    )

    os.replace(
        temporary,
        path
    )


def save_c1_2026_history(records):

    temporary = (
        HISTORY_FILE.with_suffix(".tmp")
    )

    pd.DataFrame(records).to_csv(
        temporary,
        index=False
    )

    os.replace(
        temporary,
        HISTORY_FILE
    )

# ============================================================
# 12. RESUME PREVIOUS CHECKPOINT
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

    best_epoch = int(
        saved["best_epoch"]
    )

    best_val_loss = float(
        saved["best_val_loss"]
    )

    bad_epochs = int(
        saved["bad_epochs"]
    )

    history = list(
        saved["history"]
    )

    start_epoch = (
        int(saved["epoch"]) + 1
    )

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
        "C1 Seed2026 previous checkpoint restored."
    )

else:

    print(
        "New independent Control C1 Seed2026 experiment."
    )

# ============================================================
# 13. PRE-TRAINING SAFETY CHECK
# ============================================================

print("\n" + "=" * 70)
print("CONTROL C1 SEED2026 — PRE-TRAINING CHECK")
print("=" * 70)

print("GPU:", torch.cuda.get_device_name(0))
print("Architecture: DeepLabV3+ ResNet18")
print("Tasks: AP + Tooth")
print("Seed:", SEED)
print("Tooth weight:", TOOTH_WEIGHT)
print("Starting epoch:", start_epoch)

model.eval()

probe = next(
    iter(
        build_c1_train_loader(start_epoch)
    )
)

x = probe["image"].to(DEVICE)
a = probe["AP_mask"].to(DEVICE)
t = probe["Tooth_mask"].to(DEVICE)

optimizer.zero_grad(set_to_none=True)

with torch.amp.autocast("cuda"):

    y = model(x)

    probe_loss, _, _ = (
        c1_2026_training_loss(
            y, a, t
        )
    )

assert y.shape == (4, 2, 512, 512)

assert torch.isfinite(
    probe_loss
).item()

probe_loss.backward()

optimizer.zero_grad(set_to_none=True)

del probe, x, a, t, y, probe_loss

gc.collect()
torch.cuda.empty_cache()

print("Batch-4 forward/backward: PASSED")

preflight = validate_c1_2026(
    model,
    val_loader
)

assert np.isfinite(
    preflight["val_loss"]
)

assert preflight["val_positive_images"] == 98
assert preflight["val_negative_images"] == 645

assert (
    preflight["val_Tooth_positive_images"]
    == 742
)

print("Full validation preflight: PASSED")

print(
    "AP-positive:",
    preflight["val_positive_images"]
)

print(
    "Tooth-positive:",
    preflight["val_Tooth_positive_images"]
)

if history:

    saved_last_loss = float(
        history[-1]["val_loss"]
    )

    recomputed_loss = float(
        preflight["val_loss"]
    )

    difference = abs(
        saved_last_loss - recomputed_loss
    )

    if difference > 0.002:

        raise RuntimeError(
            "Restored checkpoint validation mismatch."
        )

    print("Restored checkpoint validation: PASSED")

print(
    "\nCONTROL C1 SEED2026 "
    "PRE-TRAINING CHECK: PASSED"
)

# ============================================================
# 14. COMPLETE TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("CONTROL C1 | DEEPLABV3+ | SEED2026 TRAINING")
print("=" * 70)

print("Maximum epochs:", MAX_EPOCHS)
print("Starting epoch:", start_epoch)
print("Output directory:", RUN_DIR)

for epoch in range(
    start_epoch,
    MAX_EPOCHS + 1
):

    if bad_epochs >= PATIENCE:

        print(
            "Early stopping already reached."
        )
        break

    epoch_start = time.time()

    random.seed(
        SEED + epoch
    )

    np.random.seed(
        SEED + epoch
    )

    torch.manual_seed(
        SEED + epoch
    )

    loader = build_c1_train_loader(
        epoch
    )

    model.train()

    running_total = 0.0
    running_ap = 0.0
    running_tooth = 0.0

    seen = 0

    print(
        f"\nStarting C1 Seed2026 "
        f"Epoch {epoch}/{MAX_EPOCHS}",
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

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            (
                loss,
                ap_loss,
                tooth_loss
            ) = c1_2026_training_loss(
                outputs,
                ap,
                tooth
            )

        if not torch.isfinite(
            loss
        ).item():

            raise RuntimeError(
                f"Non-finite loss at "
                f"epoch {epoch}, "
                f"batch {batch_index}"
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = images.shape[0]

        seen += batch_size

        running_total += (
            loss.item() * batch_size
        )

        running_ap += (
            ap_loss.item() * batch_size
        )

        running_tooth += (
            tooth_loss.item() * batch_size
        )

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

    # ========================================================
    # 15. FULL VALIDATION
    # ========================================================

    metrics = validate_c1_2026(
        model,
        val_loader
    )

    assert np.isfinite(
        metrics["val_loss"]
    )

    scheduler.step(
        metrics["val_loss"]
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    record = {
        "epoch": epoch,

        "train_loss": (
            running_total / seen
        ),

        "train_AP_loss": (
            running_ap / seen
        ),

        "train_Tooth_loss": (
            running_tooth / seen
        ),

        **metrics,

        "learning_rate": current_lr,

        "epoch_minutes": (
            time.time() - epoch_start
        ) / 60.0
    }

    # ========================================================
    # 16. BEST CHECKPOINT SELECTION
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

        save_c1_2026_checkpoint(
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

    history.append(
        record
    )

    # ========================================================
    # 17. SAVE RESUMABLE CHECKPOINT
    # ========================================================

    save_c1_2026_checkpoint(
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

    save_c1_2026_history(
        history
    )

    # ========================================================
    # 18. PRINT EPOCH METRICS
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"CONTROL C1 SEED2026 — "
        f"EPOCH {epoch}/{MAX_EPOCHS}"
    )

    print("-" * 70)

    print(
        f"Train Total Loss: "
        f"{record['train_loss']:.4f}"
    )

    print(
        f"Train AP Loss: "
        f"{record['train_AP_loss']:.4f}"
    )

    print(
        f"Train Tooth Loss: "
        f"{record['train_Tooth_loss']:.4f}"
    )

    print(
        f"Val AP Loss: "
        f"{metrics['val_loss']:.4f}"
    )

    print(
        f"Val AP Dice: "
        f"{metrics['val_AP_Dice']:.4f}"
    )

    print(
        f"Val AP IoU: "
        f"{metrics['val_AP_IoU']:.4f}"
    )

    print(
        f"Val Tooth Dice: "
        f"{metrics['val_Tooth_Dice']:.4f}"
    )

    print(
        "Val AP Negative FP Rate:",
        f"{metrics['val_negative_FP_rate']:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.7f}"
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Early Stopping: "
        f"{bad_epochs}/{PATIENCE}"
    )

    if improved:

        print("BEST MODEL UPDATED")

    print("CHECKPOINT SAVED")
    print("-" * 70)

    if bad_epochs >= PATIENCE:

        print(
            "\nEARLY STOPPING ACTIVATED"
        )
        break

# ============================================================
# 19. FINAL CHECKPOINT INTEGRITY
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

assert int(
    last_saved["epoch"]
) == int(
    history[-1]["epoch"]
)

assert int(
    best_saved["epoch"]
) == best_epoch

assert len(
    last_saved["history"]
) == len(history)

best_record = next(
    row for row in history
    if int(row["epoch"]) == best_epoch
)

# ============================================================
# 20. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("CONTROL C1 SEED2026 TRAINING FINISHED")
print("=" * 70)

print("Seed:", SEED)
print("Tooth loss weight:", TOOTH_WEIGHT)

print(
    "Epochs completed:",
    len(history)
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best AP validation loss:",
    round(best_val_loss, 5)
)

print("\nBEST CHECKPOINT METRICS")

print(
    "Val AP Dice:",
    round(
        best_record["val_AP_Dice"], 4
    )
)

print(
    "Val AP IoU:",
    round(
        best_record["val_AP_IoU"], 4
    )
)

print(
    "Val Tooth Dice:",
    round(
        best_record["val_Tooth_Dice"], 4
    )
)

print(
    "Val AP Negative FP Rate:",
    round(
        best_record["val_negative_FP_rate"], 4
    )
)

print("\nSaved files:")

print("Best:", BEST)
print("Last:", LAST)
print("History:", HISTORY_FILE)

print("\nOriginal M5/M6 checkpoints: NOT MODIFIED")
print("C1 Seeds 42/123: NOT MODIFIED")
print("Original PRAD dataset: NOT MODIFIED")
print("Held-out Test set: NOT USED")

gc.collect()
torch.cuda.empty_cache()