# ORIGINAL SOURCE CELL 85 FROM USER TRAINING NOTEBOOK; no scientific logic modified.
# ============================================================
# C2 — SHUFFLED RCF CONTROL | SEED 42
#
# DeepLabV3+ ResNet18
# Tasks: AP + Tooth + Shuffled RCF
# Loss = AP + 0.3*Tooth + 0.3*ShuffledRCF
#
# Reference: M6 DeepLabV3+ Seed42
# Fixed shuffle manifest: shuffle_manifest_v1.csv
#
# Training: 3466 images
# Validation: 743 images, ORIGINAL masks
# Checkpoint selection: minimum AP validation loss
#
# NO HELD-OUT TEST ACCESS
# ============================================================

import os
import gc
import json
import time
import random
import hashlib

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
# 1. ENVIRONMENT
# ============================================================

required = [
    "REAL_MYDRIVE",
    "train_dataset",
    "val_dataset",
    "val_loader"
]

missing = [x for x in required if x not in globals()]

assert not missing, f"Missing variables: {missing}"
assert torch.cuda.is_available(), "Enable Colab GPU."
assert smp.__version__ == "0.5.0"

assert len(train_dataset) == 3466
assert len(val_dataset) == 743
assert isinstance(val_loader.sampler, SequentialSampler)

SEED = 42
DEVICE = torch.device("cuda")

ROOT = (
    Path(REAL_MYDRIVE)
    / "RCF_Aware_AP_Segmentation_Research"
)

MODELS_DIR = ROOT / "04_Models"

M6_DIR = (
    MODELS_DIR
    / "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed42"
)

C2_ROOT = (
    MODELS_DIR
    / "C02_Shuffled_RCF_Control"
)

MANIFEST_FILE = C2_ROOT / "shuffle_manifest_v1.csv"
PROTOCOL_FILE = C2_ROOT / "shuffle_protocol_v1.json"

RUN_DIR = (
    C2_ROOT
    / "C02_DeepLabV3Plus_AP_Tooth_ShuffledRCF_Seed42"
)

BEST = RUN_DIR / "best_model.pt"
LAST = RUN_DIR / "last_checkpoint.pt"
HISTORY_FILE = RUN_DIR / "training_history.csv"
CONFIG_FILE = RUN_DIR / "training_config.json"

M6_CONFIG_FILE = M6_DIR / "training_config.json"

for path in [
    MANIFEST_FILE,
    PROTOCOL_FILE,
    M6_CONFIG_FILE
]:
    assert path.is_file(), f"Missing: {path}"

print("=" * 75)
print("C2 SEED42 — ENVIRONMENT VERIFIED")
print("=" * 75)

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("SMP:", smp.__version__)
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ============================================================
# 2. VERIFY SOURCE CONFIG AND SHUFFLE PROTOCOL
# ============================================================

with open(M6_CONFIG_FILE) as f:
    m6_config = json.load(f)

with open(PROTOCOL_FILE) as f:
    shuffle_protocol = json.load(f)

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

M6_CONFIG_SHA = sha256_bytes(
    json.dumps(
        m6_config,
        sort_keys=True
    ).encode("utf-8")
)

MANIFEST_SHA = sha256_bytes(
    MANIFEST_FILE.read_bytes()
)

PROTOCOL_SHA = sha256_bytes(
    PROTOCOL_FILE.read_bytes()
)

assert (
    M6_CONFIG_SHA
    == shuffle_protocol["source_model_config_sha256"]
)

assert (
    MANIFEST_SHA
    == shuffle_protocol["manifest_sha256"]
)

assert shuffle_protocol["shuffle_seed"] == 20261019
assert shuffle_protocol["training_images"] == 3466

assert SEED in shuffle_protocol["training_seeds"]

assert shuffle_protocol["AP_labels_changed"] is False
assert shuffle_protocol["Tooth_labels_changed"] is False
assert shuffle_protocol["RCF_spatial_masks_swapped"] is True
assert shuffle_protocol["held_out_test_used"] is False

assert m6_config["model"] == "DeepLabV3Plus"
assert m6_config["encoder"] == "resnet18"
assert m6_config["encoder_weights"] == "imagenet"

assert m6_config["tasks"] == ["AP", "Tooth", "RCF"]
assert int(m6_config["output_channels"]) == 3
assert int(m6_config["encoder_output_stride"]) == 16
assert int(m6_config["image_size"]) == 512

assert int(m6_config["seed"]) == SEED

assert abs(
    float(m6_config["tooth_weight"]) - 0.3
) < 1e-12

assert abs(
    float(m6_config["rcf_weight"]) - 0.3
) < 1e-12

# Preserve the original M6 configuration.
# Add only control metadata.

CONFIG = dict(m6_config)

CONFIG.update({
    "experiment": "C2_Shuffled_RCF_Control",
    "control_type":
        "group_preserving_spatial_RCF_label_shuffle",
    "shuffle_seed": 20261019,
    "shuffle_manifest_sha256": MANIFEST_SHA,
    "shuffle_protocol_sha256": PROTOCOL_SHA,
    "source_model": (
        "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed42"
    ),
    "source_model_config_sha256": M6_CONFIG_SHA
})

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
assert AP_FRACTION == 0.4

assert int(CONFIG["AP_channel"]) == 0
assert int(CONFIG["Tooth_channel"]) == 1
assert int(CONFIG["RCF_channel"]) == 2

print("\nC2 PROTOCOL AND ORIGINAL M6 CONFIG: VERIFIED")
print("Manifest SHA256:", MANIFEST_SHA[:16])
print("Tooth weight:", TOOTH_WEIGHT)
print("Shuffled RCF weight:", RCF_WEIGHT)
print("Model selection: AP validation loss")

# ============================================================
# 3. LOAD AND VERIFY FIXED SHUFFLE MANIFEST
# ============================================================

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={
        "target_image_id": str,
        "source_image_id": str
    }
)

df = train_dataset.df.copy().reset_index(drop=True)
df["image_id"] = df["image_id"].astype(str)

df["AP"] = df["AP"].astype(int)
df["RCF"] = df["RCF"].astype(int)

assert len(manifest) == 3466
assert len(df) == 3466

target_indices = manifest["target_index"].to_numpy(
    dtype=np.int64
)

source_indices = manifest["source_index"].to_numpy(
    dtype=np.int64
)

assert np.array_equal(
    target_indices,
    np.arange(3466)
)

assert np.array_equal(
    np.sort(source_indices),
    np.arange(3466)
)

assert not np.any(
    target_indices == source_indices
)

assert np.array_equal(
    manifest["target_image_id"].to_numpy(),
    df["image_id"].to_numpy()
)

assert np.array_equal(
    manifest["source_image_id"].to_numpy(),
    df["image_id"].iloc[source_indices].to_numpy()
)

assert np.array_equal(
    manifest["target_AP"].to_numpy(),
    df["AP"].to_numpy()
)

assert np.array_equal(
    manifest["target_RCF"].to_numpy(),
    df["RCF"].to_numpy()
)

assert (
    manifest["target_group"]
    == manifest["source_group"]
).all()

assert (
    manifest["target_RCF"]
    == manifest["source_RCF"]
).all()

assert (
    manifest["target_AP"]
    == manifest["source_AP"]
).all()

assert int(manifest["target_RCF"].sum()) == 1343

print("\nC2 SHUFFLE MANIFEST: VERIFIED")
print("Target images:", 3466)
print("Unique source masks:", 3466)
print("Self mappings:", 0)
print("Cross-group mappings:", 0)

# ============================================================
# 4. CREATE INDEPENDENT RUN DIRECTORY
# ============================================================

RUN_DIR.mkdir(parents=True, exist_ok=True)

if CONFIG_FILE.exists():

    with open(CONFIG_FILE) as f:
        existing_config = json.load(f)

    assert existing_config == CONFIG, (
        "Existing experiment configuration mismatch."
    )

else:

    assert not any(
        path.exists()
        for path in [BEST, LAST, HISTORY_FILE]
    ), "Unexpected existing training files."

    temp = CONFIG_FILE.with_suffix(".tmp")

    with open(temp, "w") as f:
        json.dump(CONFIG, f, indent=2)

    os.replace(temp, CONFIG_FILE)

if BEST.exists() != LAST.exists():
    raise RuntimeError(
        "Incomplete checkpoint pair. Inspect before resuming."
    )

if HISTORY_FILE.exists() and not LAST.exists():
    raise RuntimeError(
        "History exists without resumable checkpoint."
    )

# ============================================================
# 5. RELEASE PREVIOUS GPU OBJECTS
# ============================================================

for name in [
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "images",
    "outputs",
    "loss",
    "probe",
    "probe_loss"
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
# 6. C2 TRAIN DATASET
# ============================================================

class C2ShuffledRCFDataset(Dataset):

    def __init__(self, base, sources):
        self.base = base
        self.sources = np.asarray(
            sources, dtype=np.int64
        )

        assert len(self.base) == len(self.sources)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):

        idx = int(idx)
        source_idx = int(self.sources[idx])

        # AP, Tooth and image from the TARGET image.
        target = self.base[idx]

        # RCF ONLY from a DIFFERENT source image.
        source = self.base[source_idx]

        image = target["image"].clone()
        ap = target["AP_mask"].clone()
        tooth = target["Tooth_mask"].clone()

        shuffled_rcf = source["RCF_mask"].clone()

        assert tuple(image.shape) == (3, 512, 512)
        assert tuple(ap.shape) == (1, 512, 512)
        assert tuple(tooth.shape) == (1, 512, 512)

        assert tuple(shuffled_rcf.shape) == (
            1, 512, 512
        )

        # Same geometric augmentation for target image,
        # AP, Tooth and assigned shuffled RCF.

        if random.random() < 0.5:

            image = torch.flip(image, dims=[2])
            ap = torch.flip(ap, dims=[2])
            tooth = torch.flip(tooth, dims=[2])

            shuffled_rcf = torch.flip(
                shuffled_rcf, dims=[2]
            )

        # Same mild intensity augmentation.
        # Apply to image only.

        if random.random() < 0.5:

            contrast = random.uniform(
                0.95, 1.05
            )

            brightness = random.uniform(
                -0.08, 0.08
            )

            image = torch.clamp(
                image * contrast + brightness,
                -3.0, 3.0
            )

        return {
            "image": image,
            "AP_mask": ap,
            "Tooth_mask": tooth,
            "RCF_mask": shuffled_rcf,
            "target_index": idx,
            "source_index": source_idx
        }


c2_train_dataset = C2ShuffledRCFDataset(
    train_dataset,
    source_indices
)

# ============================================================
# 7. AP-AWARE TRAINING SAMPLER
# ============================================================

labels = df["AP"].to_numpy(dtype=np.int64)

n_positive = int((labels == 1).sum())
n_negative = int((labels == 0).sum())

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


def build_train_loader(epoch):

    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(c2_train_dataset),
        replacement=True,
        generator=generator
    )

    return DataLoader(
        c2_train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

# ============================================================
# 8. MODEL, OPTIMIZER AND SCHEDULER
# ============================================================

model = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights=(
        None if LAST.exists() else "imagenet"
    ),
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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

scaler = torch.amp.GradScaler("cuda")

# ============================================================
# 9. FROZEN TASK LOSS
# ============================================================

def task_loss_parts(logits, targets):

    logits = logits.float()
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
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


def task_loss(logits, targets):

    bce, dice_loss, positive = task_loss_parts(
        logits, targets
    )

    result = bce.mean()

    if positive.any():
        result = result + dice_loss[positive].mean()

    return result


def multitask_loss(outputs, ap, tooth, rcf):

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
# 10. VALIDATION — ORIGINAL LABELS ONLY
# ============================================================

@torch.inference_mode()
def validate_c2(model, loader):

    model.eval()

    ap_bce_sum = 0.0
    ap_dice_loss_sum = 0.0

    n_images = 0
    ap_positive = 0
    ap_negative = 0
    ap_negative_fp = 0

    ap_dices = []
    ap_ious = []
    tooth_dices = []
    rcf_dices = []

    for batch in loader:

        images = batch["image"].to(DEVICE)

        # ORIGINAL validation labels.
        ap_gt = batch["AP_mask"].to(DEVICE)
        tooth_gt = batch["Tooth_mask"].to(DEVICE)
        rcf_gt = batch["RCF_mask"].to(DEVICE)

        with torch.amp.autocast("cuda"):
            outputs = model(images)

        ap_logits = outputs[:, 0:1].float()
        tooth_logits = outputs[:, 1:2].float()
        rcf_logits = outputs[:, 2:3].float()

        bce, soft_dice, positive = task_loss_parts(
            ap_logits, ap_gt
        )

        ap_bce_sum += bce.sum().item()

        ap_dice_loss_sum += (
            soft_dice[positive].sum().item()
        )

        ap_pred = ap_logits >= 0
        tooth_pred = tooth_logits >= 0
        rcf_pred = rcf_logits >= 0

        for i in range(images.shape[0]):

            gt = ap_gt[i, 0] >= 0.5
            pred = ap_pred[i, 0]

            gt_count = int(gt.sum().item())
            pred_count = int(pred.sum().item())

            if gt_count > 0:

                overlap = int(
                    (gt & pred).sum().item()
                )

                ap_dices.append(
                    2.0 * overlap / (gt_count + pred_count)
                )

                ap_ious.append(
                    overlap
                    / (gt_count + pred_count - overlap)
                )

                ap_positive += 1

            else:

                ap_negative += 1
                ap_negative_fp += int(pred_count > 0)

            # Tooth Dice on positive Tooth images.

            tooth_true = tooth_gt[i, 0] >= 0.5
            tooth_hat = tooth_pred[i, 0]

            tooth_count = int(
                tooth_true.sum().item()
            )

            if tooth_count > 0:

                tooth_pred_count = int(
                    tooth_hat.sum().item()
                )

                tooth_intersection = int(
                    (tooth_true & tooth_hat).sum().item()
                )

                tooth_dices.append(
                    2.0 * tooth_intersection
                    / (tooth_count + tooth_pred_count)
                )

            # RCF Dice uses TRUE validation RCF labels.
            # This is diagnostic only, NOT model selection.

            rcf_true = rcf_gt[i, 0] >= 0.5
            rcf_hat = rcf_pred[i, 0]

            rcf_count = int(rcf_true.sum().item())

            if rcf_count > 0:

                rcf_pred_count = int(
                    rcf_hat.sum().item()
                )

                rcf_intersection = int(
                    (rcf_true & rcf_hat).sum().item()
                )

                rcf_dices.append(
                    2.0 * rcf_intersection
                    / (rcf_count + rcf_pred_count)
                )

        n_images += images.shape[0]

    assert n_images == 743
    assert ap_positive == 98
    assert ap_negative == 645

    assert len(ap_dices) == 98
    assert len(tooth_dices) == 742
    assert len(rcf_dices) == 287

    val_ap_loss = (
        ap_bce_sum / n_images
        + ap_dice_loss_sum / ap_positive
    )

    return {
        "val_loss": float(val_ap_loss),

        "val_AP_Dice": float(np.mean(ap_dices)),
        "val_AP_IoU": float(np.mean(ap_ious)),

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

        "val_Tooth_positive_images":
            len(tooth_dices),

        "val_RCF_positive_masks":
            len(rcf_dices)
    }

# ============================================================
# 11. ATOMIC CHECKPOINT SAVING
# ============================================================

def save_checkpoint(state, path):

    temp = path.with_suffix(".tmp")

    torch.save(state, temp)

    os.replace(temp, path)


def save_history(records):

    temp = HISTORY_FILE.with_suffix(".tmp")

    pd.DataFrame(records).to_csv(
        temp,
        index=False
    )

    os.replace(temp, HISTORY_FILE)

# ============================================================
# 12. RESTORE EXISTING C2 TRAINING
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
        int(row["epoch"]) for row in history
    ] == list(range(1, start_epoch))

    assert HISTORY_FILE.exists()

    best_saved = torch.load(
        BEST,
        map_location="cpu",
        weights_only=True
    )

    assert best_saved["config"] == CONFIG
    assert int(best_saved["epoch"]) == best_epoch

    del saved, best_saved

    print("\nC2 Seed42 previous checkpoint restored.")

else:

    print("\nNew independent C2 Seed42 experiment.")

# ============================================================
# 13. PRE-TRAINING SAFETY CHECK
# ============================================================

print("\n" + "=" * 75)
print("C2 SEED42 — PRE-TRAINING CHECK")
print("=" * 75)

print("Model: DeepLabV3+ ResNet18")
print("Tasks: AP + Tooth + Shuffled RCF")
print("Seed:", SEED)
print("Starting epoch:", start_epoch)
print("Shuffled RCF weight:", RCF_WEIGHT)

model.eval()

probe = next(
    iter(build_train_loader(start_epoch))
)

x = probe["image"].to(DEVICE)
a = probe["AP_mask"].to(DEVICE)
t = probe["Tooth_mask"].to(DEVICE)
r = probe["RCF_mask"].to(DEVICE)

# Confirm mapping was actually applied.

probe_targets = probe["target_index"].numpy()
probe_sources = probe["source_index"].numpy()

assert np.array_equal(
    source_indices[probe_targets],
    probe_sources
)

assert np.all(
    probe_targets != probe_sources
)

optimizer.zero_grad(set_to_none=True)

with torch.amp.autocast("cuda"):

    y = model(x)

    probe_loss, _, _, _ = multitask_loss(
        y, a, t, r
    )

assert y.shape == (4, 3, 512, 512)
assert torch.isfinite(probe_loss).item()

probe_loss.backward()

optimizer.zero_grad(set_to_none=True)

del probe, x, a, t, r, y, probe_loss

gc.collect()
torch.cuda.empty_cache()

print("Shuffled RCF source mapping: PASSED")
print("Batch-4 forward/backward: PASSED")

preflight = validate_c2(
    model,
    val_loader
)

assert np.isfinite(preflight["val_loss"])

assert preflight["val_positive_images"] == 98
assert preflight["val_negative_images"] == 645

assert preflight["val_Tooth_positive_images"] == 742
assert preflight["val_RCF_positive_masks"] == 287

print("Full validation preflight: PASSED")
print("AP-positive images: 98")
print("Tooth-positive images: 742")
print("RCF-positive resized masks: 287")

if history:

    previous_loss = float(
        history[-1]["val_loss"]
    )

    assert abs(
        preflight["val_loss"] - previous_loss
    ) < 0.002

    print("Restored checkpoint validation: PASSED")

print("\nC2 SEED42 PRE-TRAINING CHECK: PASSED")

# ============================================================
# 14. COMPLETE TRAINING LOOP
# ============================================================

print("\n" + "=" * 75)
print("C2 | DEEPLABV3+ | SHUFFLED RCF | SEED42 TRAINING")
print("=" * 75)

print("Maximum epochs:", MAX_EPOCHS)
print("Starting epoch:", start_epoch)
print("Output:", RUN_DIR)

for epoch in range(
    start_epoch,
    MAX_EPOCHS + 1
):

    if bad_epochs >= PATIENCE:
        print("Early stopping already reached.")
        break

    epoch_start = time.time()

    random.seed(SEED + epoch)
    np.random.seed(SEED + epoch)

    torch.manual_seed(SEED + epoch)

    train_loader = build_train_loader(epoch)

    model.train()

    running_total = 0.0
    running_ap = 0.0
    running_tooth = 0.0
    running_rcf = 0.0

    seen = 0

    print(
        f"\nStarting C2 Seed42 Epoch "
        f"{epoch}/{MAX_EPOCHS}",
        flush=True
    )

    for batch_index, batch in enumerate(
        train_loader,
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
            ) = multitask_loss(
                outputs,
                ap,
                tooth,
                rcf
            )

        if not torch.isfinite(loss).item():

            raise RuntimeError(
                f"Non-finite loss at epoch "
                f"{epoch}, batch {batch_index}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
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

        running_rcf += (
            rcf_loss.item() * batch_size
        )

        if (
            batch_index % 250 == 0
            or batch_index == len(train_loader)
        ):

            print(
                f"Epoch {epoch}: "
                f"Batch {batch_index}/"
                f"{len(train_loader)}",
                flush=True
            )

    assert seen == 3466

    # ========================================================
    # 15. VALIDATION ON TRUE LABELS
    # ========================================================

    metrics = validate_c2(
        model,
        val_loader
    )

    assert np.isfinite(metrics["val_loss"])

    scheduler.step(metrics["val_loss"])

    current_lr = optimizer.param_groups[0]["lr"]

    record = {
        "epoch": epoch,

        "train_loss": running_total / seen,

        "train_AP_loss": running_ap / seen,

        "train_Tooth_loss": (
            running_tooth / seen
        ),

        "train_Shuffled_RCF_loss": (
            running_rcf / seen
        ),

        **metrics,

        "learning_rate": current_lr,

        "epoch_minutes": (
            time.time() - epoch_start
        ) / 60.0
    }

    # ========================================================
    # 16. BEST CHECKPOINT: AP VALIDATION LOSS
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
    # 17. RESUMABLE CHECKPOINT
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
    # 18. EPOCH REPORT
    # ========================================================

    print("\n" + "-" * 75)

    print(
        f"C2 SEED42 — EPOCH {epoch}/{MAX_EPOCHS}"
    )

    print("-" * 75)

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
        f"Train Shuffled RCF Loss: "
        f"{record['train_Shuffled_RCF_loss']:.4f}"
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
        f"Val RCF Dice (true labels): "
        f"{metrics['val_RCF_Dice']:.4f}"
    )

    print(
        "Val AP Negative FP Rate:",
        f"{metrics['val_negative_FP_rate']:.4f}"
    )

    print(f"Learning Rate: {current_lr:.7f}")

    print(f"Best Epoch: {best_epoch}")

    print(
        f"Early Stopping: "
        f"{bad_epochs}/{PATIENCE}"
    )

    if improved:
        print("BEST MODEL UPDATED")

    print("CHECKPOINT SAVED")
    print("-" * 75)

    if bad_epochs >= PATIENCE:

        print("\nEARLY STOPPING ACTIVATED")
        break

# ============================================================
# 19. FINAL CHECKPOINT INTEGRITY
# ============================================================

assert BEST.is_file()
assert LAST.is_file()
assert HISTORY_FILE.is_file()

best_saved = torch.load(
    BEST,
    map_location="cpu",
    weights_only=True
)

last_saved = torch.load(
    LAST,
    map_location="cpu",
    weights_only=True
)

assert best_saved["config"] == CONFIG
assert last_saved["config"] == CONFIG

assert int(best_saved["epoch"]) == best_epoch

assert int(last_saved["epoch"]) == int(
    history[-1]["epoch"]
)

assert len(last_saved["history"]) == len(history)

best_record = next(
    item for item in history
    if int(item["epoch"]) == best_epoch
)

assert abs(
    float(best_saved["val_loss"])
    - float(best_record["val_loss"])
) < 1e-7

# ============================================================
# 20. FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("C2 SEED42 SHUFFLED RCF TRAINING FINISHED")
print("=" * 75)

print("Seed:", SEED)
print("Experiment: C2 Shuffled RCF")
print("Architecture: DeepLabV3+ ResNet18")

print("Tooth loss weight:", TOOTH_WEIGHT)
print("Shuffled RCF loss weight:", RCF_WEIGHT)

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
    "Val RCF Dice (true labels):",
    round(best_record["val_RCF_Dice"], 4)
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
print("Configuration:", CONFIG_FILE)

print("\nOriginal M6 checkpoints: NOT MODIFIED")
print("Original PRAD dataset: NOT MODIFIED")
print("Shuffle manifest: NOT MODIFIED")
print("Validation labels: ORIGINAL")
print("Held-out Test set: NOT USED")

print("\nFINAL STATUS: PASSED")

gc.collect()
torch.cuda.empty_cache()