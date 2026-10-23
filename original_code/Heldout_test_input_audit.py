# ============================================================
# HELD-OUT TEST — INPUT INTEGRITY AUDIT V1
# ============================================================
#
# Frozen plan must exist and match its original SHA256.
#
# Checks:
#   1. Frozen plan integrity
#   2. All 12 checkpoint paths exist
#   3. Train / Validation / Test image-ID disjointness
#   4. Test split size and frozen strata
#   5. All Test images and masks exist
#   6. Duplicate-group isolation, where applicable
#
# NO INFERENCE
# NO MODEL SELECTION
# NO TEST IMAGE OR MASK CONTENT READ
# NO METRIC CALCULATION
# NO OVERWRITING
# ============================================================

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# ============================================================
# 1. GOOGLE DRIVE
# ============================================================

MYDRIVE = Path("/content/drive/MyDrive")

if not MYDRIVE.is_dir():

    from google.colab import drive
    drive.mount("/content/drive")

assert MYDRIVE.is_dir()

ROOT = (
    MYDRIVE
    / "RCF_Aware_AP_Segmentation_Research"
)

RAW = (
    MYDRIVE
    / "PRAD_MultiTask_Segmentation"
)

ANALYSIS = ROOT / "06_Lesion_Level_Analysis"

SPLITS = ROOT / "02_Dataset_Splits"

PLAN_FILE = (
    ANALYSIS
    / "heldout_test_analysis_plan_frozen_v1.json"
)

SHA_FILE = (
    ANALYSIS
    / "heldout_test_analysis_plan_frozen_v1.sha256"
)

AUDIT_FILE = (
    ANALYSIS
    / "heldout_test_input_integrity_audit_v1.json"
)

EXPECTED_PLAN_SHA = (
    "35dbf447bbabb1cec4bacebe5ade8c90"
    "d9d9fecf4a6e16f40db4ac1add635800"
)

assert PLAN_FILE.is_file()
assert SHA_FILE.is_file()

assert not AUDIT_FILE.exists(), (
    "Audit already exists. Do not overwrite."
)

def file_sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()

# ============================================================
# 2. FROZEN PLAN INTEGRITY
# ============================================================

actual_plan_sha = file_sha256(PLAN_FILE)

assert actual_plan_sha == EXPECTED_PLAN_SHA, (
    "Frozen plan SHA256 mismatch. "
    "STOP: do not modify the frozen plan."
)

with open(SHA_FILE, encoding="utf-8") as f:
    sha_record = f.read().strip()

assert sha_record.split()[0] == EXPECTED_PLAN_SHA

with open(PLAN_FILE, encoding="utf-8") as f:
    plan = json.load(f)

assert plan["plan_status"] == (
    "FROZEN_BEFORE_HELD_OUT_TEST"
)

assert plan["held_out_test_accessed"] is False

assert plan["training_seeds"] == [42, 123, 2026]

assert set(plan["fixed_models"]) == {
    "M5", "M6", "C1", "C2"
}

assert len(plan["checkpoints"]) == 12

assert plan["primary_analysis"]["comparison"] == (
    "M6_minus_C2"
)

assert plan["primary_analysis"]["endpoint"] == (
    "Lesion_F1"
)

assert plan["lesion_evaluation"][
    "probability_threshold"
] == 0.50

assert plan["lesion_evaluation"][
    "primary_IoU_threshold"
] == 0.10

assert plan["lesion_evaluation"][
    "primary_min_predicted_area_pixels"
] == 1

print("=" * 80)
print("FROZEN PRE-TEST PLAN: VERIFIED")
print("=" * 80)

print("Plan SHA256:", actual_plan_sha)
print("Primary: M6 minus C2 / Lesion F1")
print("Training seeds: 42, 123, 2026")
print("Checkpoints:", len(plan["checkpoints"]))

# ============================================================
# 3. CHECKPOINT AVAILABILITY
# ============================================================

registered = set()

for item in plan["checkpoints"]:

    key = (
        item["model_id"],
        int(item["seed"])
    )

    assert key not in registered

    registered.add(key)

    checkpoint_path = (
        ROOT / item["checkpoint_relative_path"]
    )

    config_path = (
        ROOT / item["config_relative_path"]
    )

    assert checkpoint_path.is_file(), (
        f"Missing checkpoint: {checkpoint_path}"
    )

    assert config_path.is_file(), (
        f"Missing configuration: {config_path}"
    )

assert len(registered) == 12

print("\nCHECKPOINT FILE AVAILABILITY: PASSED")
print("Available registered checkpoints: 12/12")
print("Checkpoint contents: NOT MODIFIED")

# ============================================================
# 4. OPEN THE FROZEN SPLIT FILES
# ============================================================

split_paths = {
    "train": SPLITS / "train_v1.csv",
    "validation": SPLITS / "validation_v1.csv",
    "test": (
        ROOT
        / plan["test_split"]["relative_csv_path"]
    )
}

expected_sizes = {
    "train": 3466,
    "validation": 743,
    "test": 743
}

split_frames = {}

for name, path in split_paths.items():

    assert path.is_file(), (
        f"Missing split file: {path}"
    )

    frame = pd.read_csv(
        path,
        dtype={"image_id": str}
    )

    assert len(frame) == expected_sizes[name]

    assert "image_id" in frame.columns
    assert "stratum" in frame.columns

    assert frame["image_id"].notna().all()

    frame["image_id"] = (
        frame["image_id"].astype(str).str.strip()
    )

    assert frame["image_id"].ne("").all()

    assert frame["image_id"].is_unique, (
        f"Duplicate image_id in {name}"
    )

    split_frames[name] = frame

train_df = split_frames["train"]
val_df = split_frames["validation"]
test_df = split_frames["test"]

# ============================================================
# 5. IMAGE-ID INDEPENDENCE ACROSS SPLITS
# ============================================================

train_ids = set(train_df["image_id"])
val_ids = set(val_df["image_id"])
test_ids = set(test_df["image_id"])

train_val_overlap = train_ids & val_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = val_ids & test_ids

assert len(train_val_overlap) == 0
assert len(train_test_overlap) == 0
assert len(val_test_overlap) == 0

assert len(
    train_ids | val_ids | test_ids
) == 4952

print("\n" + "=" * 80)
print("SPLIT INTEGRITY")
print("=" * 80)

print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Held-out Test:", len(test_ids))

print("Train/Validation image overlap:", 0)
print("Train/Test image overlap:", 0)
print("Validation/Test image overlap:", 0)

print("\nIMAGE-ID DISJOINTNESS: PASSED")

# ============================================================
# 6. VERIFY FROZEN TEST STRATA
# ============================================================

expected_strata = {
    str(k): int(v)
    for k, v in plan[
        "test_split"
    ]["expected_strata"].items()
}

actual_strata = {
    str(k): int(v)
    for k, v in test_df[
        "stratum"
    ].value_counts().items()
}

assert actual_strata == expected_strata, (
    "Test strata differ from frozen plan. "
    "Do not update the plan based on Test."
)

assert sum(actual_strata.values()) == 743

print("\n" + "=" * 80)
print("TEST STRATIFICATION: VERIFIED")
print("=" * 80)

for group in [
    "AP_RCF",
    "AP_ONLY",
    "RCF_ONLY",
    "NEITHER"
]:

    print(
        f"{group}:",
        actual_strata[group]
    )

# ============================================================
# 7. CHECK KNOWN DUPLICATE-GROUP ISOLATION
# ============================================================

def active_duplicate_groups(frame):

    columns = {
        "duplicate_group_id",
        "duplicate_group_size"
    }

    if not columns.issubset(frame.columns):
        return set()

    sizes = pd.to_numeric(
        frame["duplicate_group_size"],
        errors="coerce"
    ).fillna(1)

    identifiers = frame.loc[
        sizes > 1,
        "duplicate_group_id"
    ].dropna()

    return set(
        identifiers.astype(str)
    )

duplicate_groups = {
    name: active_duplicate_groups(frame)
    for name, frame in split_frames.items()
}

assert not (
    duplicate_groups["train"]
    & duplicate_groups["validation"]
)

assert not (
    duplicate_groups["train"]
    & duplicate_groups["test"]
)

assert not (
    duplicate_groups["validation"]
    & duplicate_groups["test"]
)

print("\nKNOWN DUPLICATE-GROUP ISOLATION: PASSED")

# ============================================================
# 8. VERIFY ALL TEST FILE PATHS
# ============================================================

IMAGE_DIR = RAW / "image"
MASK_DIR = RAW / "label"

assert IMAGE_DIR.is_dir()
assert MASK_DIR.is_dir()

missing_images = []
missing_masks = []

for image_id in test_df["image_id"]:

    image_path = IMAGE_DIR / f"{image_id}.jpg"
    mask_path = MASK_DIR / f"{image_id}.png"

    if not image_path.is_file():
        missing_images.append(image_id)

    if not mask_path.is_file():
        missing_masks.append(image_id)

print("\n" + "=" * 80)
print("TEST SOURCE FILE AUDIT")
print("=" * 80)

print("Expected Test images:", 743)
print("Missing image files:", len(missing_images))
print("Missing mask files:", len(missing_masks))

if missing_images:
    print("First missing images:", missing_images[:10])

if missing_masks:
    print("First missing masks:", missing_masks[:10])

assert not missing_images
assert not missing_masks

print("TEST IMAGE AND MASK PATHS: VERIFIED")

# Important:
# This stage checks file existence only.
# It does NOT open, resize, or inspect Test images/masks.

# ============================================================
# 9. SAVE INPUT AUDIT
# ============================================================

audit = {
    "audit_id": "HeldOut_Test_Input_Integrity_v1",

    "status": "PASSED",

    "created_at_utc": (
        datetime.now(timezone.utc).isoformat()
    ),

    "frozen_plan_sha256": actual_plan_sha,

    "test_split_relative_path": (
        plan["test_split"]["relative_csv_path"]
    ),

    "test_csv_sha256": file_sha256(
        split_paths["test"]
    ),

    "split_counts": {
        "train": 3466,
        "validation": 743,
        "test": 743
    },

    "test_strata": actual_strata,

    "train_validation_overlap": 0,
    "train_test_overlap": 0,
    "validation_test_overlap": 0,

    "missing_test_images": 0,
    "missing_test_masks": 0,

    "checkpoints_registered": 12,

    "test_csv_opened_after_pretest_lock": True,

    "test_image_contents_read": False,
    "test_mask_contents_read": False,

    "test_predictions_generated": False,
    "test_metrics_computed": False,

    "test_based_tuning_performed": False,

    "patient_disjointness_verified": False,

    "frozen_plan_modified": False
}

temporary = AUDIT_FILE.with_suffix(".tmp")

with open(
    temporary,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit,
        f,
        indent=2,
        ensure_ascii=False
    )

with open(
    temporary,
    encoding="utf-8"
) as f:

    verify_audit = json.load(f)

assert verify_audit == audit

assert not AUDIT_FILE.exists()

os.replace(
    temporary,
    AUDIT_FILE
)

assert AUDIT_FILE.is_file()

# Check the original frozen plan remains unchanged.
assert file_sha256(
    PLAN_FILE
) == EXPECTED_PLAN_SHA

print("\n" + "=" * 85)
print("HELD-OUT TEST INPUT INTEGRITY AUDIT COMPLETED")
print("=" * 85)

print("Frozen plan: VERIFIED")
print("Frozen plan SHA256:", actual_plan_sha)

print("Train/Validation/Test disjointness: PASSED")
print("Test images available: 743/743")
print("Test masks available: 743/743")

print("Registered checkpoints: 12")

print("\nAudit file:")
print(AUDIT_FILE)

print("\nTest CSV: OPENED AFTER PLAN FREEZE")
print("Test image contents read: NO")
print("Test mask contents read: NO")

print("Model inference: NONE")
print("Test metrics computed: NONE")
print("Model selection or retuning: NONE")

print("Frozen plan modified: NO")
print("Previous results modified: NO")

print("\nFINAL STATUS: PASSED")