# SOURCE CELL 18 — original, method logic unchanged.
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

# ============================================================
# STEP: TRAINING RESOLUTION AUDIT
# Original PRAD dataset remains read-only.
# ============================================================

DRIVE_ROOT = Path(REAL_MYDRIVE)

DATASET_ROOT = (
    DRIVE_ROOT / "PRAD_MultiTask_Segmentation"
)

LABEL_DIR = DATASET_ROOT / "label"

PROJECT_ROOT = (
    DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"
)

SPLIT_DIR = PROJECT_ROOT / "02_Dataset_Splits"

QC_DIR = PROJECT_ROOT / "03_Preprocessing_QC"
QC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD TRAINING MANIFEST ONLY
# ============================================================

train_df = pd.read_csv(
    SPLIT_DIR / "train_v1.csv",
    dtype={"image_id": str}
)

ap_train = train_df[
    train_df["AP"] == 1
].copy()

print("Training images:", len(train_df))
print("AP-positive training images:", len(ap_train))

assert len(train_df) == 3466
assert len(ap_train) == 460

# ============================================================
# 2. RESIZE MASK PRESERVING ASPECT RATIO
# ============================================================

AP_ID = 9

def resize_mask(mask_array, target_size):

    height, width = mask_array.shape

    scale = min(
        target_size / width,
        target_size / height
    )

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    # Convert indexed IDs into a grayscale image
    # without applying the original RGB palette.
    mask_image = Image.fromarray(
        mask_array.astype(np.uint8)
    )

    resized = mask_image.resize(
        (new_width, new_height),
        Image.Resampling.NEAREST
    )

    canvas = Image.new(
        "L",
        (target_size, target_size),
        color=0
    )

    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2

    canvas.paste(resized, (left, top))

    return np.asarray(canvas)

# ============================================================
# 3. COMPARE 512 AND 768 RESOLUTIONS
# ============================================================

records = []

for _, row in tqdm(
    ap_train.iterrows(),
    total=len(ap_train),
    desc="Checking AP mask preservation"
):

    image_id = row["image_id"]
    mask_path = LABEL_DIR / f"{image_id}.png"

    with Image.open(mask_path) as mask:

        if mask.mode != "P":
            raise ValueError(
                f"Unexpected mask mode: {image_id}"
            )

        mask_array = np.asarray(mask.copy())

    original_ap_pixels = int(
        np.count_nonzero(mask_array == AP_ID)
    )

    result = {
        "image_id": image_id,
        "original_AP_pixels": original_ap_pixels
    }

    for size in [512, 768]:

        resized_mask = resize_mask(
            mask_array, size
        )

        resized_ap_pixels = int(
            np.count_nonzero(
                resized_mask == AP_ID
            )
        )

        result[f"AP_pixels_{size}"] = (
            resized_ap_pixels
        )

        result[f"AP_disappeared_{size}"] = (
            resized_ap_pixels == 0
        )

    records.append(result)

results_df = pd.DataFrame(records)

# ============================================================
# 4. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("AP MASK RESOLUTION PRESERVATION RESULTS")
print("=" * 60)

for size in [512, 768]:

    counts = results_df[f"AP_pixels_{size}"]

    print(f"\nResolution: {size}x{size}")

    print(
        "Images where AP disappeared:",
        int((counts == 0).sum())
    )

    print(
        "Images with 1-3 AP pixels:",
        int(((counts > 0) & (counts <= 3)).sum())
    )

    print(
        "Images with 1-10 AP pixels:",
        int(((counts > 0) & (counts <= 10)).sum())
    )

    print(
        "Median AP pixels:",
        round(float(counts.median()), 2)
    )

    print(
        "Minimum AP pixels:",
        int(counts.min())
    )

# ============================================================
# 5. SAVE RESULTS IN RESEARCH FOLDER
# ============================================================

output_path = (
    QC_DIR / "AP_resolution_preservation_audit.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("RESOLUTION AUDIT COMPLETED")
print("=" * 60)

print("Saved results:", output_path)
print("Original dataset: NOT MODIFIED")
print("Test split: NOT USED")