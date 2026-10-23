# SOURCE CELL 13 — original, method logic unchanged.
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

# ============================================================
# STEP: EXACT DUPLICATE ANNOTATION CONSISTENCY AUDIT
# Original dataset is read-only.
# ============================================================

DRIVE_ROOT = Path(REAL_MYDRIVE)

DATASET_ROOT = DRIVE_ROOT / "PRAD_MultiTask_Segmentation"
LABEL_DIR = DATASET_ROOT / "label"

RESEARCH_ROOT = (
    DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"
)

REVIEW_DIR = (
    RESEARCH_ROOT
    / "01_Dataset_Analysis"
    / "Duplicate_Check"
    / "Visual_Review"
)

audit = pd.read_csv(
    REVIEW_DIR / "duplicate_mask_audit.csv",
    dtype={"image_1": str, "image_2": str}
)

exact_pairs = audit[
    audit["pair_type"] == "EXACT"
].copy()

CLASSES = {
    "Tooth": 1,
    "RCF": 3,
    "AP": 9
}

results = []

for _, row in exact_pairs.iterrows():

    id_a = str(row["image_1"])
    id_b = str(row["image_2"])

    with Image.open(LABEL_DIR / f"{id_a}.png") as m:
        mask_a = np.asarray(m.copy())

    with Image.open(LABEL_DIR / f"{id_b}.png") as m:
        mask_b = np.asarray(m.copy())

    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask dimensions differ for pair {row['pair_id']}"
        )

    result = {
        "pair_id": int(row["pair_id"]),
        "image_1": id_a,
        "image_2": id_b,
        "total_different_pixels": int(
            np.count_nonzero(mask_a != mask_b)
        )
    }

    for name, class_id in CLASSES.items():

        binary_a = mask_a == class_id
        binary_b = mask_b == class_id

        intersection = np.count_nonzero(
            binary_a & binary_b
        )

        pixels_a = np.count_nonzero(binary_a)
        pixels_b = np.count_nonzero(binary_b)

        difference = np.count_nonzero(
            binary_a ^ binary_b
        )

        total = pixels_a + pixels_b

        # Undefined when both class masks are empty
        dice = (
            2 * intersection / total
            if total > 0 else np.nan
        )

        result[f"{name}_pixels_A"] = pixels_a
        result[f"{name}_pixels_B"] = pixels_b
        result[f"{name}_different_pixels"] = difference
        result[f"{name}_Dice"] = (
            round(dice, 4)
            if not np.isnan(dice) else np.nan
        )

    results.append(result)

results_df = pd.DataFrame(results)

OUTPUT = (
    REVIEW_DIR / "exact_duplicate_annotation_consistency.csv"
)

results_df.to_csv(OUTPUT, index=False)

print("=" * 65)
print("EXACT DUPLICATE ANNOTATION CONSISTENCY")
print("=" * 65)

columns = [
    "pair_id",
    "total_different_pixels",
    "AP_different_pixels",
    "AP_Dice",
    "RCF_different_pixels",
    "RCF_Dice",
    "Tooth_different_pixels",
    "Tooth_Dice"
]

print(
    results_df[columns].to_string(
        index=False,
        na_rep="N/A"
    )
)

print("\nResults saved to:", OUTPUT)
print("Original PRAD dataset: NOT MODIFIED")