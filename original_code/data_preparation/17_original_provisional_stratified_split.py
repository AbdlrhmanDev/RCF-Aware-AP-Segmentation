# SOURCE CELL 17 — original, method logic unchanged.
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import json

# ============================================================
# STEP: PROVISIONAL STRATIFIED DATASET SPLITTING
# Original PRAD images and masks remain unchanged.
# ============================================================

DRIVE_ROOT = Path(REAL_MYDRIVE)

PROJECT_ROOT = (
    DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"
)

ANALYSIS_DIR = PROJECT_ROOT / "01_Dataset_Analysis"

MANIFEST_PATH = (
    ANALYSIS_DIR
    / "Duplicate_Check"
    / "PRAD_duplicate_group_manifest.csv"
)

SPLIT_DIR = PROJECT_ROOT / "02_Dataset_Splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD AND VERIFY DATASET MANIFEST
# ============================================================

df = pd.read_csv(
    MANIFEST_PATH,
    dtype={
        "image_id": str,
        "duplicate_group_id": str
    }
)

required_columns = {
    "image_id",
    "AP",
    "RCF",
    "duplicate_group_id",
    "duplicate_group_size",
    "annotation_review_required"
}

assert required_columns.issubset(df.columns)
assert len(df) == 5000
assert df["image_id"].is_unique

# ============================================================
# 2. QUARANTINE ALL DETECTED DUPLICATE GROUPS
# ============================================================

quarantine = df[
    df["duplicate_group_size"] > 1
].copy()

clean = df[
    df["duplicate_group_size"] == 1
].copy()

assert len(quarantine) == 48
assert len(clean) == 4952

# Check flagged cases are not retained
flagged = (
    clean["annotation_review_required"]
    .astype(str)
    .str.lower()
    .eq("true")
)

assert not flagged.any(), (
    "Annotation-review cases remain in clean dataset."
)

print("=" * 60)
print("DATASET CLEANING SUMMARY")
print("=" * 60)

print("Original images:", len(df))
print("Quarantined images:", len(quarantine))
print("Eligible images:", len(clean))

# ============================================================
# 3. CREATE FOUR STRATIFICATION GROUPS
# ============================================================

group_names = {
    (1, 1): "AP_RCF",
    (1, 0): "AP_ONLY",
    (0, 1): "RCF_ONLY",
    (0, 0): "NEITHER"
}

clean["stratum"] = [
    group_names[(int(ap), int(rcf))]
    for ap, rcf in zip(clean["AP"], clean["RCF"])
]

print("\nEligible cases by group:")
print(clean["stratum"].value_counts().sort_index())

# ============================================================
# 4. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

SEED = 42

train_df, remaining_df = train_test_split(
    clean,
    test_size=0.30,
    stratify=clean["stratum"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    remaining_df,
    test_size=0.50,
    stratify=remaining_df["stratum"],
    random_state=SEED
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

# ============================================================
# 5. VERIFY NO IMAGE OVERLAP
# ============================================================

train_ids = set(train_df["image_id"])
val_ids = set(val_df["image_id"])
test_ids = set(test_df["image_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

assert len(train_ids | val_ids | test_ids) == len(clean)

print("\nImage overlap check: PASSED")

# ============================================================
# 6. GENERATE DISTRIBUTION REPORT
# ============================================================

combined = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

summary = pd.crosstab(
    combined["split"],
    combined["stratum"]
).reindex(
    index=["train", "validation", "test"],
    columns=[
        "AP_RCF",
        "AP_ONLY",
        "RCF_ONLY",
        "NEITHER"
    ],
    fill_value=0
)

summary["Total"] = summary.sum(axis=1)

print("\n" + "=" * 60)
print("FINAL DATASET SPLIT SUMMARY")
print("=" * 60)

print(summary.to_string())

# ============================================================
# 7. SAVE SPLIT MANIFESTS (VERSION 1)
# ============================================================

paths = {
    "train": SPLIT_DIR / "train_v1.csv",
    "validation": SPLIT_DIR / "validation_v1.csv",
    "test": SPLIT_DIR / "test_v1.csv",
    "quarantine": SPLIT_DIR / "quarantine_v1.csv",
    "summary": SPLIT_DIR / "split_summary_v1.csv",
    "config": SPLIT_DIR / "split_config_v1.json"
}

# Prevent silent overwriting of a frozen split
existing = [
    str(path)
    for path in paths.values()
    if path.exists()
]

if existing:
    raise FileExistsError(
        "Split v1 already exists. Files will not be overwritten:\n"
        + "\n".join(existing)
    )

train_df.to_csv(paths["train"], index=False)
val_df.to_csv(paths["validation"], index=False)
test_df.to_csv(paths["test"], index=False)
quarantine.to_csv(paths["quarantine"], index=False)
summary.to_csv(paths["summary"])

config = {
    "dataset": "PRAD",
    "original_images": 5000,
    "quarantined_images": len(quarantine),
    "eligible_images": len(clean),
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "random_seed": SEED,
    "stratification": "Image-level AP/RCF co-occurrence",
    "duplicate_policy": "Quarantine all detected multi-image groups",
    "patient_ids_available": False,
    "status": "PROVISIONAL - patient independence unverified"
}

with open(paths["config"], "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 60)
print("SPLIT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Saved directory:", SPLIT_DIR)
print("Original PRAD dataset: NOT MODIFIED")
print("All six models will use the same split.")