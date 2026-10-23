# ============================================================
# C2 — SHUFFLED RCF SPATIAL SUPERVISION
# STAGE 1: CREATE AND AUDIT FIXED SHUFFLE MANIFEST
#
# Shuffle RCF masks within each AP/RCF subgroup.
# Preserve AP and Tooth labels.
# Preserve RCF presence and subgroup mask distribution.
#
# Manifest reused across seeds 42, 123, 2026.
#
# NO TRAINING
# NO MODEL MODIFICATION
# NO VALIDATION LABEL MODIFICATION
# NO TEST ACCESS
# ============================================================

import os
import json
import hashlib

import numpy as np
import pandas as pd
import torch

from pathlib import Path

# ============================================================
# 1. ENVIRONMENT AND PATHS
# ============================================================

required = [
    "REAL_MYDRIVE",
    "train_dataset",
    "val_dataset"
]

missing = [
    name for name in required
    if name not in globals()
]

assert not missing, f"Missing variables: {missing}"

assert len(train_dataset) == 3466
assert len(val_dataset) == 743

ROOT = (
    Path(REAL_MYDRIVE)
    / "RCF_Aware_AP_Segmentation_Research"
)

MODELS_DIR = ROOT / "04_Models"

M6_DIR = (
    MODELS_DIR
    / "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed42"
)

M6_CONFIG_FILE = (
    M6_DIR / "training_config.json"
)

C2_ROOT = (
    MODELS_DIR
    / "C02_Shuffled_RCF_Control"
)

MANIFEST_FILE = (
    C2_ROOT / "shuffle_manifest_v1.csv"
)

PROTOCOL_FILE = (
    C2_ROOT / "shuffle_protocol_v1.json"
)

assert M6_CONFIG_FILE.is_file(), (
    f"Missing original M6 configuration: "
    f"{M6_CONFIG_FILE}"
)

for path in [MANIFEST_FILE, PROTOCOL_FILE]:
    if path.exists():
        raise RuntimeError(
            f"Existing C2 protocol file: {path}. "
            "Do not overwrite it."
        )

SHUFFLE_SEED = 20261019
TRAINING_SEEDS = [42, 123, 2026]

EXPECTED_GROUPS = {
    "AP_RCF": 199,
    "AP_ONLY": 261,
    "RCF_ONLY": 1144,
    "NEITHER": 1862
}

print("=" * 78)
print("C2 SHUFFLED RCF — ENVIRONMENT VERIFIED")
print("=" * 78)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Shuffle seed:", SHUFFLE_SEED)
print("Training seeds:", TRAINING_SEEDS)

# ============================================================
# 2. VERIFY ORIGINAL M6 CONFIGURATION
# ============================================================

with open(M6_CONFIG_FILE) as f:
    m6_config = json.load(f)

assert int(m6_config["seed"]) == 42
assert m6_config["model"] == "DeepLabV3Plus"
assert m6_config["encoder"] == "resnet18"
assert m6_config["tasks"] == ["AP", "Tooth", "RCF"]

assert int(m6_config["output_channels"]) == 3
assert int(m6_config["image_size"]) == 512
assert int(m6_config["encoder_output_stride"]) == 16

assert abs(
    float(m6_config["tooth_weight"]) - 0.3
) < 1e-12

rcf_weight = m6_config.get(
    "rcf_weight",
    m6_config.get("RCF_weight")
)

if rcf_weight is not None:
    assert abs(float(rcf_weight) - 0.3) < 1e-12

CONFIG_FINGERPRINT = hashlib.sha256(
    json.dumps(
        m6_config,
        sort_keys=True
    ).encode("utf-8")
).hexdigest()

print("\nORIGINAL M6 CONFIGURATION: VERIFIED")
print("Tasks:", m6_config["tasks"])
print("Tooth weight:", m6_config["tooth_weight"])
print("RCF weight field:", rcf_weight)
print("M6 config SHA256:", CONFIG_FINGERPRINT[:16])

# ============================================================
# 3. VERIFY TRAINING DATAFRAME
# ============================================================

df = train_dataset.df.copy().reset_index(drop=True)

required_columns = [
    "image_id",
    "AP",
    "RCF"
]

for column in required_columns:
    assert column in df.columns, (
        f"Missing training column: {column}"
    )

assert len(df) == 3466

df["image_id"] = df["image_id"].astype(str)

assert df["image_id"].is_unique

df["AP"] = df["AP"].astype(int)
df["RCF"] = df["RCF"].astype(int)

assert df["AP"].isin([0, 1]).all()
assert df["RCF"].isin([0, 1]).all()

assert int(df["AP"].sum()) == 460
assert int(df["RCF"].sum()) == 1343

def get_group(row):

    ap = int(row["AP"])
    rcf = int(row["RCF"])

    if ap and rcf:
        return "AP_RCF"

    if ap:
        return "AP_ONLY"

    if rcf:
        return "RCF_ONLY"

    return "NEITHER"


df["group"] = df.apply(get_group, axis=1)

assert (
    df["group"].value_counts().to_dict()
    == EXPECTED_GROUPS
)

print("\n" + "=" * 78)
print("C2 — TRAINING SUBGROUPS VERIFIED")
print("=" * 78)

print(
    df["group"]
    .value_counts()
    .reindex(EXPECTED_GROUPS.keys())
    .to_string()
)

# ============================================================
# 4. GENERATE GROUP-PRESERVING DERANGEMENT
# ============================================================

# A derangement means no image receives
# its own original RCF mask.
#
# Every original mask is used exactly once.
#
# Shuffling is separately applied inside
# each of the four training subgroups.

rng = np.random.default_rng(SHUFFLE_SEED)

n_images = len(df)

source_indices = np.full(
    n_images,
    -1,
    dtype=np.int64
)

for group_name in EXPECTED_GROUPS:

    indices = np.flatnonzero(
        df["group"].to_numpy() == group_name
    )

    assert len(indices) == EXPECTED_GROUPS[group_name]
    assert len(indices) > 1

    permuted = rng.permutation(indices)

    # Form one complete cycle.
    # This guarantees no self-mapping.

    shuffled_sources = np.roll(permuted, 1)

    source_indices[permuted] = shuffled_sources

assert (source_indices >= 0).all()

# Every source is used exactly once.
assert len(np.unique(source_indices)) == n_images

# No image receives its own mask.
assert not np.any(
    source_indices == np.arange(n_images)
)

# Each mask remains within its original subgroup.
assert np.array_equal(
    df["group"].to_numpy(),
    df["group"].to_numpy()[source_indices]
)

# Exact RCF-positive status is preserved.
assert np.array_equal(
    df["RCF"].to_numpy(),
    df["RCF"].to_numpy()[source_indices]
)

# Exact AP subgroup status is preserved.
assert np.array_equal(
    df["AP"].to_numpy(),
    df["AP"].to_numpy()[source_indices]
)

# ============================================================
# 5. BUILD MANIFEST
# ============================================================

manifest = pd.DataFrame({
    "target_index": np.arange(
        n_images,
        dtype=np.int64
    ),
    "target_image_id": df["image_id"].to_numpy(),
    "target_group": df["group"].to_numpy(),
    "target_AP": df["AP"].to_numpy(),
    "target_RCF": df["RCF"].to_numpy(),

    "source_index": source_indices,

    "source_image_id": (
        df["image_id"]
        .iloc[source_indices]
        .to_numpy()
    ),

    "source_group": (
        df["group"]
        .iloc[source_indices]
        .to_numpy()
    ),

    "source_AP": (
        df["AP"]
        .iloc[source_indices]
        .to_numpy()
    ),

    "source_RCF": (
        df["RCF"]
        .iloc[source_indices]
        .to_numpy()
    )
})

assert len(manifest) == 3466
assert manifest["target_index"].is_unique
assert manifest["source_index"].is_unique

assert manifest["target_image_id"].is_unique
assert manifest["source_image_id"].is_unique

assert (
    manifest["target_index"]
    != manifest["source_index"]
).all()

assert (
    manifest["target_group"]
    == manifest["source_group"]
).all()

assert (
    manifest["target_RCF"]
    == manifest["source_RCF"]
).all()

assert (
    manifest["target_AP"]
    == manifest["source_AP"]
).all()

assert manifest["target_RCF"].sum() == 1343
assert manifest["source_RCF"].sum() == 1343

# ============================================================
# 6. AUDIT REAL DATASET MASKS
# ============================================================

# Check representative target/source samples.
# Do not modify any mask.
#
# The later training code must retrieve the RCF
# mask using source_index, while AP and Tooth
# masks come from target_index.

audit_rng = np.random.default_rng(77123)

audit_indices = []

for group_name in EXPECTED_GROUPS:

    indices = np.flatnonzero(
        df["group"].to_numpy() == group_name
    )

    selected = audit_rng.choice(
        indices,
        size=min(8, len(indices)),
        replace=False
    )

    audit_indices.extend(selected.tolist())

checked = 0
positive_pairs = 0
different_mask_pairs = 0

for target_index in audit_indices:

    source_index = int(
        source_indices[target_index]
    )

    target_sample = train_dataset[
        int(target_index)
    ]

    source_sample = train_dataset[
        source_index
    ]

    required_keys = [
        "image",
        "AP_mask",
        "Tooth_mask",
        "RCF_mask",
        "image_id"
    ]

    for key in required_keys:

        assert key in target_sample
        assert key in source_sample

    assert str(
        target_sample["image_id"]
    ) == str(
        df.loc[target_index, "image_id"]
    )

    assert str(
        source_sample["image_id"]
    ) == str(
        df.loc[source_index, "image_id"]
    )

    target_rcf = target_sample["RCF_mask"]
    source_rcf = source_sample["RCF_mask"]

    assert isinstance(target_rcf, torch.Tensor)
    assert isinstance(source_rcf, torch.Tensor)

    assert tuple(target_rcf.shape) == (
        1, 512, 512
    )

    assert tuple(source_rcf.shape) == (
        1, 512, 512
    )

    assert torch.isfinite(target_rcf).all()
    assert torch.isfinite(source_rcf).all()

    assert (
        (target_rcf >= 0)
        & (target_rcf <= 1)
    ).all()

    assert (
        (source_rcf >= 0)
        & (source_rcf <= 1)
    ).all()

    # RCF-negative groups must have empty masks.
    if int(df.loc[target_index, "RCF"]) == 0:

        assert not torch.any(target_rcf > 0.5)
        assert not torch.any(source_rcf > 0.5)

    else:

        positive_pairs += 1

        if not torch.equal(
            target_rcf,
            source_rcf
        ):

            different_mask_pairs += 1

    checked += 1

assert checked == 32
assert positive_pairs == 16

# At least one audited positive RCF mask
# must differ from its assigned source mask.

assert different_mask_pairs > 0

print("\n" + "=" * 78)
print("C2 — REAL MASK AUDIT PASSED")
print("=" * 78)

print("Target/source pairs audited:", checked)
print("RCF-positive pairs audited:", positive_pairs)
print(
    "RCF-positive pairs with different masks:",
    different_mask_pairs
)

print("RCF mask shape: (1, 512, 512)")
print("Dataset images and masks: NOT MODIFIED")

# ============================================================
# 7. VERIFY MANIFEST AS A PERMUTATION
# ============================================================

# Verify every row and each source position.
# This also ensures the mapping can be reused
# by index in the later training DataLoader.

assert np.array_equal(
    manifest["target_index"].to_numpy(),
    np.arange(n_images)
)

assert sorted(
    manifest["source_index"].tolist()
) == list(range(n_images))

for group_name, expected_count in (
    EXPECTED_GROUPS.items()
):

    subgroup = manifest[
        manifest["target_group"] == group_name
    ]

    assert len(subgroup) == expected_count

    assert (
        subgroup["source_group"]
        == group_name
    ).all()

print("\n" + "=" * 78)
print("C2 — SHUFFLE MANIFEST VERIFIED")
print("=" * 78)

print("Total mappings:", n_images)
print("Unique source masks:", n_images)
print("Self mappings:", 0)
print("Cross-group mappings:", 0)
print("RCF-positive targets:", 1343)
print("RCF-positive sources:", 1343)

# ============================================================
# 8. BUILD PROTOCOL METADATA
# ============================================================

manifest_csv = manifest.to_csv(
    index=False
)

manifest_sha = hashlib.sha256(
    manifest_csv.encode("utf-8")
).hexdigest()

metadata = {
    "experiment": "C02_Shuffled_RCF_Control",
    "protocol_version": "1.0",
    "control_type": (
        "group_preserving_spatial_RCF_label_shuffle"
    ),
    "shuffle_seed": SHUFFLE_SEED,
    "training_seeds": TRAINING_SEEDS,

    "source_model": (
        "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed42"
    ),

    "source_model_config_sha256": (
        CONFIG_FINGERPRINT
    ),

    "training_images": 3466,
    "validation_images": 743,
    "train_group_counts": EXPECTED_GROUPS,

    "shuffling_scope": "training_only",
    "shuffling_method": (
        "random_single_cycle_derangement_"
        "within_each_AP_RCF_subgroup"
    ),

    "AP_labels_changed": False,
    "Tooth_labels_changed": False,

    "RCF_spatial_masks_swapped": True,

    "RCF_presence_per_image_preserved": True,
    "AP_status_per_image_preserved": True,

    "RCF_mask_marginal_distribution_"
    "within_group_preserved": True,

    "every_mask_used_once": True,
    "self_assignments": 0,
    "cross_group_assignments": 0,

    "manifest_sha256": manifest_sha,

    "preprocessing": (
        "Original training dataset at 512x512; "
        "apply identical training augmentation "
        "to image, AP, Tooth and shuffled RCF mask."
    ),

    "validation_labels": "original_unshuffled",
    "model_selection": (
        "minimum_AP_validation_loss"
    ),

    "primary_evaluation": (
        "frozen_lesion_protocol_v1"
    ),

    "study_interpretation": (
        "Tests correspondence of RCF spatial "
        "supervision while retaining RCF presence "
        "and group-specific RCF mask distribution. "
        "Not a complete test of all RCF information."
    ),

    "patient_disjointness_verified": False,
    "held_out_test_used": False
}

# ============================================================
# 9. SAVE FIXED PROTOCOL
# ============================================================

assert not MANIFEST_FILE.exists()
assert not PROTOCOL_FILE.exists()

C2_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

manifest_tmp = (
    MANIFEST_FILE.with_suffix(".tmp")
)

protocol_tmp = (
    PROTOCOL_FILE.with_suffix(".tmp")
)

with open(
    manifest_tmp, "w", encoding="utf-8"
) as f:

    f.write(manifest_csv)

with open(
    protocol_tmp, "w", encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

# Verify saved temporary manifest and hash.

saved_bytes = manifest_tmp.read_bytes()

assert hashlib.sha256(
    saved_bytes
).hexdigest() == manifest_sha

saved_manifest = pd.read_csv(
    manifest_tmp,
    dtype={
        "target_image_id": str,
        "source_image_id": str
    }
)

assert len(saved_manifest) == 3466

assert np.array_equal(
    saved_manifest["source_index"].to_numpy(),
    source_indices
)

with open(protocol_tmp) as f:
    saved_metadata = json.load(f)

assert saved_metadata["manifest_sha256"] == manifest_sha
assert saved_metadata["held_out_test_used"] is False

os.replace(
    manifest_tmp,
    MANIFEST_FILE
)

os.replace(
    protocol_tmp,
    PROTOCOL_FILE
)

# ============================================================
# 10. FINAL REPORT
# ============================================================

assert MANIFEST_FILE.is_file()
assert PROTOCOL_FILE.is_file()

print("\n" + "=" * 78)
print("C2 SHUFFLED RCF PROTOCOL CREATED SUCCESSFULLY")
print("=" * 78)

print("Training images:", 3466)
print("Shuffled RCF mappings:", 3466)
print("Training seeds:", TRAINING_SEEDS)

print(
    "Positive RCF target/source pairs:",
    1343
)

print("Self mappings:", 0)
print("Cross-group mappings:", 0)

print(
    "Manifest SHA256:",
    manifest_sha
)

print("\nSaved:")
print(MANIFEST_FILE)
print(PROTOCOL_FILE)

print("\nOriginal AP masks: NOT MODIFIED")
print("Original Tooth masks: NOT MODIFIED")
print("Original RCF masks: NOT MODIFIED")
print("Original M6 checkpoints: NOT MODIFIED")
print("Validation labels: NOT MODIFIED")
print("Training performed: NONE")
print("Held-out Test accessed: NO")

print("\nFINAL STATUS: PASSED")