# SOURCE CELL 15 — original, method logic unchanged.
from pathlib import Path
import pandas as pd
from collections import defaultdict

# ============================================================
# 1. PROJECT PATHS
# ============================================================

DRIVE_ROOT = Path(REAL_MYDRIVE)

PROJECT_ROOT = (
    DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"
)

ANALYSIS_DIR = PROJECT_ROOT / "01_Dataset_Analysis"
DUP_DIR = ANALYSIS_DIR / "Duplicate_Check"
REVIEW_DIR = DUP_DIR / "Visual_Review"

# ============================================================
# 2. LOAD PREVIOUS AUDIT RESULTS
# ============================================================

pairs = pd.read_csv(
    REVIEW_DIR / "duplicate_mask_audit.csv",
    dtype={"image_1": str, "image_2": str}
)

inventory = pd.read_csv(
    ANALYSIS_DIR / "PRAD_AP_RCF_inventory.csv",
    dtype={"image_id": str}
)

assert len(inventory) == 5000
assert inventory["image_id"].is_unique
assert len(pairs) == 25
assert set(pairs["pair_id"]) == set(range(1, 26))

image_ids = set(inventory["image_id"])

# ============================================================
# 3. RECORD VISUAL REVIEW DECISIONS
# ============================================================

# Pair 24 is visually different.
# Other near pairs are provisionally grouped to prevent leakage.

AP_REVIEW_PAIRS = {3, 5, 11, 13, 15, 16}

decisions = []

for _, row in pairs.iterrows():

    pair_id = int(row["pair_id"])
    pair_type = str(row["pair_type"])

    if pair_id == 24:
        decision = "DIFFERENT"
    elif pair_type == "EXACT":
        decision = "EXACT_DUPLICATE"
    else:
        decision = "PROBABLE_DUPLICATE"

    # All exact duplicates have annotation differences.
    needs_annotation_review = (
        pair_type == "EXACT"
        or pair_id in AP_REVIEW_PAIRS
    )

    decisions.append({
        "pair_id": pair_id,
        "image_1": row["image_1"],
        "image_2": row["image_2"],
        "visual_decision": decision,
        "annotation_review_required": needs_annotation_review,
        "split_together": decision != "DIFFERENT"
    })

decision_df = pd.DataFrame(decisions)

# Verify referenced images exist
referenced_ids = (
    set(decision_df["image_1"])
    | set(decision_df["image_2"])
)

assert referenced_ids.issubset(image_ids), (
    "Some duplicate IDs are missing from the inventory."
)

# ============================================================
# 4. BUILD CONNECTED DUPLICATE GROUPS
# ============================================================

parent = {image_id: image_id for image_id in image_ids}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a

for _, row in decision_df.iterrows():

    if row["split_together"]:
        union(row["image_1"], row["image_2"])

groups = defaultdict(list)

for image_id in sorted(image_ids):
    groups[find(image_id)].append(image_id)

# Give each group a stable ID
group_lookup = {}
group_sizes = {}

for members in groups.values():

    group_id = "GROUP_" + min(members)

    for image_id in members:
        group_lookup[image_id] = group_id
        group_sizes[image_id] = len(members)

# ============================================================
# 5. FLAG ANNOTATION REVIEW GROUPS
# ============================================================

review_groups = set()

for _, row in decision_df.iterrows():

    if row["annotation_review_required"]:
        review_groups.add(
            group_lookup[row["image_1"]]
        )

inventory["duplicate_group_id"] = (
    inventory["image_id"].map(group_lookup)
)

inventory["duplicate_group_size"] = (
    inventory["image_id"].map(group_sizes)
)

inventory["annotation_review_required"] = (
    inventory["duplicate_group_id"].isin(review_groups)
)

# ============================================================
# 6. SAVE ALL RESULTS OUTSIDE ORIGINAL DATASET
# ============================================================

decision_path = (
    DUP_DIR / "duplicate_review_decisions.csv"
)

manifest_path = (
    DUP_DIR / "PRAD_duplicate_group_manifest.csv"
)

decision_df.to_csv(decision_path, index=False)
inventory.to_csv(manifest_path, index=False)

# ============================================================
# 7. SUMMARY
# ============================================================

group_count = (
    inventory.loc[
        inventory["duplicate_group_size"] > 1,
        "duplicate_group_id"
    ].nunique()
)

grouped_images = int(
    (inventory["duplicate_group_size"] > 1).sum()
)

review_images = int(
    inventory["annotation_review_required"].sum()
)

print("=" * 60)
print("PRAD DUPLICATE GROUPING SUMMARY")
print("=" * 60)

print("Total images:", len(inventory))
print("Total unique groups:", len(groups))
print("Groups containing multiple images:", group_count)
print("Images in multi-image groups:", grouped_images)
print("Images flagged for annotation review:", review_images)

print("\nPair decisions:")
print(decision_df["visual_decision"].value_counts())

print("\nSaved files:")
print(decision_path)
print(manifest_path)

print("\nOriginal PRAD dataset: NOT MODIFIED")
print("No images deleted.")
print("No train/validation/test split created.")