# SOURCE CELL 9 — original, method logic unchanged.
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd

# ============================================================
# STEP 1: USE THE CORRECT GOOGLE DRIVE
# ============================================================

# REAL_MYDRIVE was defined in the previous successful cell.
DRIVE_ROOT = Path(REAL_MYDRIVE)

DATASET_ROOT = DRIVE_ROOT / "PRAD_MultiTask_Segmentation"

IMAGE_DIR = DATASET_ROOT / "image"
LABEL_DIR = DATASET_ROOT / "label"

# ============================================================
# STEP 2: VERIFY ORIGINAL DATASET
# ============================================================

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

if not LABEL_DIR.is_dir():
    raise FileNotFoundError(f"Label folder not found: {LABEL_DIR}")

image_files = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg"]
)

mask_files = sorted(
    p for p in LABEL_DIR.iterdir()
    if p.suffix.lower() == ".png"
)

image_ids = {p.stem for p in image_files}
mask_ids = {p.stem for p in mask_files}

print("=" * 60)
print("PRAD DATASET VERIFICATION")
print("=" * 60)

print("Images:", len(image_files))
print("Masks:", len(mask_files))
print("Matching pairs:", len(image_ids & mask_ids))

if len(image_files) != 5000 or len(mask_files) != 5000:
    raise ValueError("Expected 5,000 images and 5,000 masks.")

if image_ids != mask_ids:
    raise ValueError("Image and mask filenames do not match.")

print("Dataset verification: PASSED")

# ============================================================
# STEP 3: CREATE A SEPARATE RESEARCH FOLDER
# ============================================================

PROJECT_ROOT = (
    DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"
)

ANALYSIS_DIR = PROJECT_ROOT / "01_Dataset_Analysis"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

print("\nResearch folder:", PROJECT_ROOT)

# ============================================================
# STEP 4: PRAD CLASS IDS
# ============================================================

TOOTH_ID = 1
RCF_ID = 3
AP_ID = 9

# ============================================================
# STEP 5: ANALYZE ALL SEGMENTATION MASKS
# ============================================================

rows = []

for mask_path in tqdm(mask_files, desc="Analyzing PRAD masks"):

    with Image.open(mask_path) as mask:

        if mask.mode != "P":
            raise ValueError(
                f"Unexpected mask mode: {mask_path.name}"
            )

        histogram = mask.histogram()

        if any(histogram[10:]):
            raise ValueError(
                f"Unexpected class ID in {mask_path.name}"
            )

        total_pixels = mask.width * mask.height

        ap_pixels = histogram[AP_ID]
        rcf_pixels = histogram[RCF_ID]
        tooth_pixels = histogram[TOOTH_ID]

        rows.append({
            "image_id": mask_path.stem,
            "AP": int(ap_pixels > 0),
            "RCF": int(rcf_pixels > 0),
            "Tooth": int(tooth_pixels > 0),
            "AP_pixels": ap_pixels,
            "RCF_pixels": rcf_pixels,
            "AP_area_percent": (
                100 * ap_pixels / total_pixels
            )
        })

df = pd.DataFrame(rows)

# ============================================================
# STEP 6: AP-RCF CO-OCCURRENCE ANALYSIS
# ============================================================

table = pd.crosstab(
    df["AP"], df["RCF"]
).reindex(
    index=[0, 1],
    columns=[0, 1],
    fill_value=0
)

both = int(table.loc[1, 1])
ap_only = int(table.loc[1, 0])
rcf_only = int(table.loc[0, 1])
neither = int(table.loc[0, 0])

print("\n" + "=" * 60)
print("AP-RCF CO-OCCURRENCE RESULTS")
print("=" * 60)

print(f"A. AP + RCF        : {both}")
print(f"B. AP without RCF  : {ap_only}")
print(f"C. RCF without AP  : {rcf_only}")
print(f"D. Neither         : {neither}")

print("\nTotal images:", len(df))
print("AP-positive images:", int(df["AP"].sum()))
print("RCF-positive images:", int(df["RCF"].sum()))

if len(df) != 5000 or df["AP"].sum() != 670 or df["RCF"].sum() != 1943:
    raise ValueError(
        "Counts differ from the previous dataset audit. "
        "Please investigate before saving results."
    )

print("\nAP-positive images containing RCF (%):")
print(round(100 * both / 670, 2))

# ============================================================
# STEP 7: AP AREA DISTRIBUTION
# ============================================================

ap_positive = df[df["AP"] == 1]

print("\n" + "=" * 60)
print("AP AREA DISTRIBUTION")
print("=" * 60)

print(
    ap_positive["AP_area_percent"]
    .describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    .round(4)
)

# ============================================================
# STEP 8: SAVE RESULTS ONLY IN RESEARCH FOLDER
# ============================================================

inventory_path = (
    ANALYSIS_DIR / "PRAD_AP_RCF_inventory.csv"
)

summary_path = (
    ANALYSIS_DIR / "PRAD_AP_RCF_cooccurrence.csv"
)

df.to_csv(inventory_path, index=False)

summary_df = pd.DataFrame({
    "Group": [
        "AP + RCF",
        "AP without RCF",
        "RCF without AP",
        "Neither"
    ],
    "Image_Count": [
        both,
        ap_only,
        rcf_only,
        neither
    ]
})

summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Inventory:", inventory_path)
print("Summary:", summary_path)

print("\nOriginal PRAD dataset: NOT MODIFIED")
print("Research outputs: SAVED IN GOOGLE DRIVE")