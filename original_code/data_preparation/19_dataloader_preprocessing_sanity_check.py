# SOURCE CELL 19 — original, method logic unchanged.
# ============================================================
# STEP 1: PRAD PYTORCH DATASET AND DATALOADER
# Research: RCF-Aware AP Segmentation
# Training resolution: 512 x 512
# Original dataset: READ ONLY
# ============================================================

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

DRIVE_ROOT = Path(REAL_MYDRIVE)

DATASET_ROOT = DRIVE_ROOT / "PRAD_MultiTask_Segmentation"

IMAGE_DIR = DATASET_ROOT / "image"
LABEL_DIR = DATASET_ROOT / "label"

PROJECT_ROOT = (
    DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"
)

SPLIT_DIR = PROJECT_ROOT / "02_Dataset_Splits"
QC_DIR = PROJECT_ROOT / "03_Preprocessing_QC"

QC_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
BATCH_SIZE = 4
SEED = 42

assert IMAGE_DIR.is_dir()
assert LABEL_DIR.is_dir()

# ============================================================
# 2. PRAD CLASS IDS
# ============================================================

TOOTH_ID = 1
RCF_ID = 3
AP_ID = 9

# ImageNet normalization for pretrained encoders
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ============================================================
# 3. RESIZE WITH ASPECT RATIO PRESERVATION
# ============================================================

def resize_with_padding(image, mask, target_size=512):

    original_width, original_height = image.size

    if image.size != mask.size:
        raise ValueError("Image and mask dimensions do not match.")

    scale = min(
        target_size / original_width,
        target_size / original_height
    )

    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))

    resized_image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    resized_mask = mask.resize(
        (new_width, new_height),
        Image.Resampling.NEAREST
    )

    image_canvas = Image.new(
        "RGB",
        (target_size, target_size),
        color=(0, 0, 0)
    )

    mask_canvas = Image.new(
        "L",
        (target_size, target_size),
        color=0
    )

    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2

    image_canvas.paste(resized_image, (left, top))
    mask_canvas.paste(resized_mask, (left, top))

    return image_canvas, mask_canvas


# ============================================================
# 4. PYTORCH DATASET
# ============================================================

class PRADDataset(Dataset):

    def __init__(self, manifest_path):

        self.df = pd.read_csv(
            manifest_path,
            dtype={"image_id": str}
        )

        self.image_lookup = {
            p.stem: p
            for p in IMAGE_DIR.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg"]
        }

        if not self.df["image_id"].is_unique:
            raise ValueError("Duplicate image IDs in manifest.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image_id = self.df.iloc[index]["image_id"]

        image_path = self.image_lookup.get(image_id)
        mask_path = LABEL_DIR / f"{image_id}.png"

        if image_path is None or not mask_path.exists():
            raise FileNotFoundError(image_id)

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as m:

            if m.mode != "P":
                raise ValueError(
                    f"Unexpected mask mode: {image_id}"
                )

            # Preserve original class IDs, not RGB colors
            mask_array = np.array(m, dtype=np.uint8)

        if not np.isin(mask_array, np.arange(10)).all():
            raise ValueError(
                f"Unexpected class IDs: {image_id}"
            )

        mask_image = Image.fromarray(mask_array)

        image, mask_image = resize_with_padding(
            image,
            mask_image,
            IMAGE_SIZE
        )

        # Convert image to float32, range [0,1]
        image_array = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        # Normalize for pretrained encoder
        image_array = (
            image_array - MEAN
        ) / STD

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        ).float()

        # Convert integer class map into binary masks
        class_mask = np.asarray(mask_image)

        ap_mask = torch.from_numpy(
            (class_mask == AP_ID).astype(np.float32)
        ).unsqueeze(0)

        tooth_mask = torch.from_numpy(
            (class_mask == TOOTH_ID).astype(np.float32)
        ).unsqueeze(0)

        rcf_mask = torch.from_numpy(
            (class_mask == RCF_ID).astype(np.float32)
        ).unsqueeze(0)

        return {
            "image": image_tensor,
            "AP_mask": ap_mask,
            "Tooth_mask": tooth_mask,
            "RCF_mask": rcf_mask,
            "image_id": image_id
        }


# ============================================================
# 5. LOAD FIXED TRAIN AND VALIDATION SPLITS
# ============================================================

train_dataset = PRADDataset(
    SPLIT_DIR / "train_v1.csv"
)

val_dataset = PRADDataset(
    SPLIT_DIR / "validation_v1.csv"
)

assert len(train_dataset) == 3466
assert len(val_dataset) == 743

# Do not load the held-out test split during development

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# 6. CHECK BATCH DIMENSIONS
# ============================================================

batch = next(iter(train_loader))

print("=" * 60)
print("PRAD DATALOADER VERIFICATION")
print("=" * 60)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

print("\nBatch size:", batch["image"].shape[0])

print("Image shape:", tuple(batch["image"].shape))
print("AP mask shape:", tuple(batch["AP_mask"].shape))
print("Tooth mask shape:", tuple(batch["Tooth_mask"].shape))
print("RCF mask shape:", tuple(batch["RCF_mask"].shape))

for name in ["AP_mask", "Tooth_mask", "RCF_mask"]:
    values = torch.unique(batch[name]).tolist()
    print(f"{name} unique values: {values}")

assert batch["image"].shape == (
    BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE
)

for name in ["AP_mask", "Tooth_mask", "RCF_mask"]:
    assert batch[name].shape == (
        BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE
    )

print("\nDataloader verification: PASSED")

# ============================================================
# 7. VISUAL CHECK USING TRAINING IMAGES ONLY
# ============================================================

groups = ["AP_RCF", "AP_ONLY", "RCF_ONLY", "NEITHER"]

fig, axes = plt.subplots(
    4, 2,
    figsize=(10, 19)
)

for row_idx, group in enumerate(groups):

    subset = train_dataset.df[
        (train_dataset.df["AP"] == int(group in ["AP_RCF", "AP_ONLY"])) &
        (train_dataset.df["RCF"] == int(group in ["AP_RCF", "RCF_ONLY"]))
    ]

    sample_index = int(subset.index[0])

    sample = train_dataset[sample_index]

    image_array = sample["image"].numpy().transpose(1, 2, 0)

    # Reverse normalization for visualization
    image_array = np.clip(
        image_array * STD + MEAN, 0, 1
    )

    ap = sample["AP_mask"][0].numpy().astype(bool)
    tooth = sample["Tooth_mask"][0].numpy().astype(bool)
    rcf = sample["RCF_mask"][0].numpy().astype(bool)

    overlay = image_array.copy()

    for region, color in [
        (tooth, np.array([0.0, 0.6, 1.0])),
        (rcf, np.array([1.0, 0.85, 0.0])),
        (ap, np.array([1.0, 0.1, 0.2]))
    ]:
        overlay[region] = (
            0.55 * overlay[region]
            + 0.45 * color
        )

    axes[row_idx, 0].imshow(image_array)
    axes[row_idx, 0].set_title(f"{group} - Original")

    axes[row_idx, 1].imshow(overlay)
    axes[row_idx, 1].set_title(f"{group} - Segmentation Overlay")

    for ax in axes[row_idx]:
        ax.axis("off")

plt.tight_layout()

preview_path = QC_DIR / "PRAD_dataloader_sanity_v1.png"

plt.savefig(
    preview_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ============================================================
# 8. SAVE PREPROCESSING CONFIGURATION
# ============================================================

config_path = QC_DIR / "dataloader_config_v1.json"

config = {
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "resize_method": "aspect_ratio_preserving_padding",
    "image_interpolation": "bilinear",
    "mask_interpolation": "nearest",
    "image_normalization": "imagenet",
    "classes": {
        "Tooth": 1,
        "RCF": 3,
        "AP": 9
    },
    "train_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "split_version": "v1",
    "test_split_used": False,
    "augmentation": "not applied during sanity check"
}

if not config_path.exists():
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
else:
    with open(config_path) as f:
        if json.load(f) != config:
            raise ValueError(
                "An existing DataLoader configuration differs. "
                "Review before overwriting."
            )

print("\n" + "=" * 60)
print("DATALOADER PREPARATION COMPLETED")
print("=" * 60)

print("Preview saved:", preview_path)
print("Configuration:", config_path)
print("Original dataset: NOT MODIFIED")
print("Test set: NOT USED")
print("No model training performed.")