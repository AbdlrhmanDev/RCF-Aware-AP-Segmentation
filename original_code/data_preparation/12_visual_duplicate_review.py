# SOURCE CELL 12 — original, method logic unchanged.
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. PATHS — ORIGINAL DATA IS READ ONLY
# ============================================================

DRIVE_ROOT = Path(REAL_MYDRIVE)

DATASET_ROOT = DRIVE_ROOT / "PRAD_MultiTask_Segmentation"
IMAGE_DIR = DATASET_ROOT / "image"
LABEL_DIR = DATASET_ROOT / "label"

PROJECT_ROOT = DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"

DUP_DIR = (
    PROJECT_ROOT / "01_Dataset_Analysis" / "Duplicate_Check"
)

REVIEW_DIR = DUP_DIR / "Visual_Review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD PREVIOUS DUPLICATE RESULTS
# ============================================================

exact_df = pd.read_csv(
    DUP_DIR / "exact_duplicates.csv",
    dtype={"image_1": str, "image_2": str}
)

near_df = pd.read_csv(
    DUP_DIR / "near_duplicate_candidates.csv",
    dtype={"image_1": str, "image_2": str}
)

pairs = []
seen = set()

for pair_type, source in [
    ("EXACT", exact_df),
    ("NEAR", near_df)
]:
    for _, row in source.iterrows():

        id1 = str(row["image_1"])
        id2 = str(row["image_2"])

        key = tuple(sorted((id1, id2)))

        if key in seen:
            continue

        seen.add(key)

        distance = (
            0 if pair_type == "EXACT"
            else int(row["hamming_distance"])
        )

        pairs.append({
            "image_1": id1,
            "image_2": id2,
            "pair_type": pair_type,
            "hamming_distance": distance
        })

print("Unique candidate pairs:", len(pairs))

# ============================================================
# 3. LOCATE ORIGINAL IMAGES
# ============================================================

image_lookup = {
    p.stem: p
    for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg"]
}

def load_pair(image_id):

    image_path = image_lookup.get(image_id)

    if image_path is None:
        raise FileNotFoundError(image_id)

    mask_path = LABEL_DIR / f"{image_id}.png"

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    with Image.open(mask_path) as m:
        if m.mode != "P":
            raise ValueError(
                f"Unexpected mask mode: {image_id}"
            )
        mask = m.copy()

    if image.size != mask.size:
        raise ValueError(
            f"Image-mask mismatch: {image_id}"
        )

    return image, mask

# ============================================================
# 4. GENERATE SEGMENTATION OVERLAY
# ============================================================

COLORS = {
    1: [0, 160, 255],     # Tooth
    3: [255, 210, 0],     # RCF
    9: [255, 50, 70]      # AP
}

def make_preview(image, mask):

    # Preserve original aspect ratio
    img = image.copy()
    img.thumbnail((650, 650), Image.Resampling.LANCZOS)

    resized_mask = mask.resize(
        img.size,
        Image.Resampling.NEAREST
    )

    image_array = np.asarray(img)
    mask_array = np.asarray(resized_mask)

    overlay = image_array.copy()

    for class_id, color in COLORS.items():

        region = mask_array == class_id

        if region.any():
            overlay[region] = (
                0.55 * image_array[region]
                + 0.45 * np.asarray(color)
            ).astype(np.uint8)

    return image_array, overlay

# ============================================================
# 5. COMPARE ANNOTATIONS FOR EXACT DUPLICATES
# ============================================================

review_records = []

for index, pair in enumerate(pairs, start=1):

    image1, mask1 = load_pair(pair["image_1"])
    image2, mask2 = load_pair(pair["image_2"])

    record = {
        "pair_id": index,
        **pair,
        "same_mask_dimensions": mask1.size == mask2.size,
        "exact_mask_equal": None,
        "AP_mask_equal": None,
        "RCF_mask_equal": None,
        "Tooth_mask_equal": None,
        "manual_decision": "PENDING"
    }

    if pair["pair_type"] == "EXACT":

        if mask1.size == mask2.size:

            arr1 = np.asarray(mask1)
            arr2 = np.asarray(mask2)

            record["exact_mask_equal"] = bool(
                np.array_equal(arr1, arr2)
            )

            for name, class_id in [
                ("AP", 9),
                ("RCF", 3),
                ("Tooth", 1)
            ]:
                record[f"{name}_mask_equal"] = bool(
                    np.array_equal(
                        arr1 == class_id,
                        arr2 == class_id
                    )
                )

    review_records.append(record)

# ============================================================
# 6. SAVE ANNOTATION AUDIT
# ============================================================

audit_df = pd.DataFrame(review_records)

audit_path = REVIEW_DIR / "duplicate_mask_audit.csv"
audit_df.to_csv(audit_path, index=False)

# ============================================================
# 7. CREATE REVIEW FIGURES — 5 PAIRS PER PAGE
# ============================================================

PAIRS_PER_PAGE = 5

for start in range(0, len(review_records), PAIRS_PER_PAGE):

    batch = review_records[start:start + PAIRS_PER_PAGE]

    fig, axes = plt.subplots(
        len(batch),
        4,
        figsize=(16, 3.8 * len(batch)),
        squeeze=False
    )

    for row_index, pair in enumerate(batch):

        img1, mask1 = load_pair(pair["image_1"])
        img2, mask2 = load_pair(pair["image_2"])

        original1, overlay1 = make_preview(img1, mask1)
        original2, overlay2 = make_preview(img2, mask2)

        panels = [
            (original1, "Image A"),
            (original2, "Image B"),
            (overlay1, "Mask Overlay A"),
            (overlay2, "Mask Overlay B")
        ]

        for col_index, (panel, title) in enumerate(panels):

            ax = axes[row_index, col_index]

            ax.imshow(panel)
            ax.axis("off")

            ax.set_title(
                f"Pair {pair['pair_id']} | {title}",
                fontsize=10
            )

        axes[row_index, 0].set_ylabel(
            f"{pair['pair_type']}\n"
            f"Distance: {pair['hamming_distance']}",
            fontsize=10
        )

    plt.tight_layout()

    page_number = start // PAIRS_PER_PAGE + 1

    output_path = REVIEW_DIR / (
        f"duplicate_review_page_{page_number:02d}.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("Saved review page:", output_path)

# ============================================================
# 8. FINAL SUMMARY
# ============================================================

exact_audit = audit_df[
    audit_df["pair_type"] == "EXACT"
]

mask_disagreements = exact_audit[
    exact_audit["exact_mask_equal"] == False
]

print("\n" + "=" * 60)
print("DUPLICATE REVIEW SUMMARY")
print("=" * 60)

print("Candidate pairs reviewed:", len(audit_df))
print("Exact duplicate pairs:", len(exact_audit))
print("Near-duplicate candidates:", len(near_df))

print(
    "Exact image pairs with different masks:",
    len(mask_disagreements)
)

if len(mask_disagreements) > 0:
    print("\nPairs requiring annotation review:")
    print(
        mask_disagreements[
            [
                "pair_id",
                "AP_mask_equal",
                "RCF_mask_equal",
                "Tooth_mask_equal"
            ]
        ].to_string(index=False)
    )

print("\nAudit saved:", audit_path)
print("Review images saved:", REVIEW_DIR)
print("Original dataset: NOT MODIFIED")