# SOURCE CELL 10 — original, method logic unchanged.
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PROJECT PATHS
# ============================================================

DRIVE_ROOT = Path(REAL_MYDRIVE)

DATASET_ROOT = DRIVE_ROOT / "PRAD_MultiTask_Segmentation"
PROJECT_ROOT = DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"

IMAGE_DIR = DATASET_ROOT / "image"
LABEL_DIR = DATASET_ROOT / "label"
ANALYSIS_DIR = PROJECT_ROOT / "01_Dataset_Analysis"

QC_DIR = ANALYSIS_DIR / "Visual_Quality_Check"
QC_DIR.mkdir(parents=True, exist_ok=True)

inventory_path = ANALYSIS_DIR / "PRAD_AP_RCF_inventory.csv"

df = pd.read_csv(inventory_path)

# ============================================================
# 2. SELECT SAMPLES FROM THE FOUR GROUPS
# ============================================================

groups = {
    "AP + RCF": (df["AP"] == 1) & (df["RCF"] == 1),
    "AP only": (df["AP"] == 1) & (df["RCF"] == 0),
    "RCF only": (df["AP"] == 0) & (df["RCF"] == 1),
    "Neither": (df["AP"] == 0) & (df["RCF"] == 0),
}

selected = []

for group_name, condition in groups.items():

    subset = df[condition]

    samples = subset.sample(
        n=min(2, len(subset)),
        random_state=42
    )

    for _, row in samples.iterrows():
        selected.append((row["image_id"], group_name))

# Add difficult AP cases: smallest and largest masks
ap_cases = df[df["AP"] == 1]

for _, row in ap_cases.nsmallest(2, "AP_pixels").iterrows():
    selected.append((row["image_id"], "Very small AP"))

for _, row in ap_cases.nlargest(2, "AP_pixels").iterrows():
    selected.append((row["image_id"], "Large AP"))

# Remove duplicate selections
selected = list(dict.fromkeys(selected))

print("Total samples selected:", len(selected))

# ============================================================
# 3. LOCATE IMAGES
# ============================================================

image_lookup = {
    p.stem: p
    for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg"]
}

# ============================================================
# 4. VISUALIZE ORIGINAL IMAGE, MASKS AND OVERLAY
# ============================================================

colors = {
    1: [0, 160, 255],     # Tooth - Blue
    3: [255, 210, 0],     # Root Canal Filling - Yellow
    9: [255, 50, 70],     # Apical Periodontitis - Red
}

fig, axes = plt.subplots(
    len(selected), 3,
    figsize=(14, 3.2 * len(selected))
)

for i, (image_id, group_name) in enumerate(selected):

    image_path = image_lookup[image_id]
    mask_path = LABEL_DIR / f"{image_id}.png"

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    with Image.open(mask_path) as source_mask:
        if source_mask.mode != "P":
            raise ValueError(f"Unexpected mask mode: {image_id}")
        mask = source_mask.copy()

    if image.size != mask.size:
        raise ValueError(f"Image-mask size mismatch: {image_id}")

    # Resize only for visualization, not training
    preview_size = (400, 300)

    image = image.resize(
        preview_size,
        Image.Resampling.BILINEAR
    )

    mask = mask.resize(
        preview_size,
        Image.Resampling.NEAREST
    )

    img_array = np.array(image)
    mask_array = np.array(mask)

    colored_mask = np.zeros_like(img_array)

    for class_id, color in colors.items():
        colored_mask[mask_array == class_id] = color

    overlay = img_array.astype(np.float32).copy()

    for class_id, color in colors.items():
        region = mask_array == class_id

        overlay[region] = (
            0.55 * overlay[region]
            + 0.45 * np.array(color)
        )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    axes[i, 0].imshow(img_array)
    axes[i, 0].set_title(f"Original - {group_name}")

    axes[i, 1].imshow(colored_mask)
    axes[i, 1].set_title("Selected Masks")

    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title("Image + Masks")

    for ax in axes[i]:
        ax.axis("off")

fig.suptitle(
    "PRAD Mask Quality Inspection\n"
    "Blue: Tooth | Yellow: RCF | Red: AP",
    fontsize=15,
    y=1.001
)

plt.tight_layout()

output_file = QC_DIR / "PRAD_visual_mask_alignment_audit.png"

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ============================================================
# 5. SAVE SELECTED SAMPLE LIST
# ============================================================

sample_df = pd.DataFrame(
    selected,
    columns=["image_id", "sample_group"]
)

sample_df.to_csv(
    QC_DIR / "PRAD_visual_QC_samples.csv",
    index=False
)

# ============================================================
# 6. CHECK VERY SMALL AP ANNOTATIONS
# ============================================================

print("\n" + "=" * 55)
print("SMALL AP ANNOTATION CHECK")
print("=" * 55)

for threshold in [0.01, 0.05, 0.1]:
    count = (
        (df["AP"] == 1) &
        (df["AP_area_percent"] < threshold)
    ).sum()

    print(f"AP area < {threshold}% : {count} images")

print("\nVisual inspection completed.")
print("Saved figure:", output_file)
print("Original dataset was NOT modified.")