# SOURCE CELL 11 — original, method logic unchanged.
from pathlib import Path
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
import hashlib
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

DRIVE_ROOT = Path(REAL_MYDRIVE)

DATASET_ROOT = DRIVE_ROOT / "PRAD_MultiTask_Segmentation"
IMAGE_DIR = DATASET_ROOT / "image"

PROJECT_ROOT = DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"

DUPLICATE_DIR = (
    PROJECT_ROOT / "01_Dataset_Analysis" / "Duplicate_Check"
)

DUPLICATE_DIR.mkdir(parents=True, exist_ok=True)

image_files = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg"]
)

print("Images found:", len(image_files))

assert len(image_files) == 5000, "Unexpected image count"

# ============================================================
# 2. HASH FUNCTIONS
# ============================================================

def calculate_sha256(path):

    sha = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def calculate_dhash(path):

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = img.resize((9, 8), Image.Resampling.LANCZOS)

        pixels = np.asarray(img, dtype=np.uint8)

    differences = pixels[:, 1:] > pixels[:, :-1]

    hash_value = 0

    for value in differences.flatten():
        hash_value = (hash_value << 1) | int(value)

    return hash_value

# ============================================================
# 3. ANALYZE ALL IMAGES
# ============================================================

records = []

for image_path in tqdm(
    image_files,
    desc="Computing image hashes"
):

    records.append({
        "image_id": image_path.stem,
        "sha256": calculate_sha256(image_path),
        "dhash": calculate_dhash(image_path)
    })

# ============================================================
# 4. EXACT DUPLICATES
# ============================================================

hash_groups = defaultdict(list)

for record in records:
    hash_groups[record["sha256"]].append(
        record["image_id"]
    )

exact_pairs = []

for group in hash_groups.values():

    if len(group) > 1:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):

                exact_pairs.append({
                    "image_1": group[i],
                    "image_2": group[j]
                })

print("\n" + "=" * 60)
print("EXACT DUPLICATE RESULTS")
print("=" * 60)

print("Exact duplicate pairs:", len(exact_pairs))

# ============================================================
# 5. NEAR-DUPLICATE CANDIDATES
# ============================================================

near_pairs = []

# Conservative threshold for candidate screening
HAMMING_THRESHOLD = 4

for i in tqdm(
    range(len(records)),
    desc="Checking similar image pairs"
):

    hash_i = records[i]["dhash"]

    for j in range(i + 1, len(records)):

        hash_j = records[j]["dhash"]

        distance = (hash_i ^ hash_j).bit_count()

        if distance <= HAMMING_THRESHOLD:

            # Exact file duplicates are already recorded
            if records[i]["sha256"] == records[j]["sha256"]:
                continue

            near_pairs.append({
                "image_1": records[i]["image_id"],
                "image_2": records[j]["image_id"],
                "hamming_distance": distance
            })

# ============================================================
# 6. SAVE RESULTS IN THE RESEARCH FOLDER
# ============================================================

exact_df = pd.DataFrame(
    exact_pairs,
    columns=["image_1", "image_2"]
)

near_df = pd.DataFrame(
    near_pairs,
    columns=[
        "image_1",
        "image_2",
        "hamming_distance"
    ]
)

near_df = near_df.sort_values(
    "hamming_distance"
).reset_index(drop=True)

exact_path = DUPLICATE_DIR / "exact_duplicates.csv"
near_path = DUPLICATE_DIR / "near_duplicate_candidates.csv"

exact_df.to_csv(exact_path, index=False)
near_df.to_csv(near_path, index=False)

# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PRAD DUPLICATE ANALYSIS SUMMARY")
print("=" * 60)

print("Total images:", len(records))
print("Exact duplicate pairs:", len(exact_df))
print("Near-duplicate candidate pairs:", len(near_df))

if len(near_df) > 0:
    print("\nCandidate distance distribution:")
    print(
        near_df["hamming_distance"]
        .value_counts()
        .sort_index()
    )

print("\nSaved results:")
print(exact_path)
print(near_path)

print("\nOriginal PRAD images were NOT modified.")
print("No images were deleted.")