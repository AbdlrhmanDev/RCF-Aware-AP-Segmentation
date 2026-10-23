# ==============================================================
# PRE-TEST EVALUATION PLAN LOCK — VERSION 1
#
# Research:
# RCF-Aware Multi-Task Segmentation for
# Apical Periodontitis on Periapical Radiographs
#
# Freeze model/checkpoint selection, comparisons,
# endpoints and analysis procedures BEFORE Test.
#
# NO TRAINING
# NO INFERENCE
# NO TEST CSV READ
# NO TEST IMAGE/MASK ACCESS
# ==============================================================

import os
import json
import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================
# 1. RESTORE GOOGLE DRIVE IF NECESSARY
# ==============================================================

if "REAL_MYDRIVE" not in globals():

    from google.colab import drive

    drive.mount("/content/drive")

    REAL_MYDRIVE = Path("/content/drive/MyDrive")

ROOT = (
    Path(REAL_MYDRIVE)
    / "RCF_Aware_AP_Segmentation_Research"
)

ANALYSIS = ROOT / "06_Lesion_Level_Analysis"
MODELS_ROOT = ROOT / "04_Models"

assert ANALYSIS.is_dir()
assert MODELS_ROOT.is_dir()

PLAN_FILE = (
    ANALYSIS
    / "heldout_test_analysis_plan_frozen_v1.json"
)

SHA_FILE = (
    ANALYSIS
    / "heldout_test_analysis_plan_frozen_v1.sha256"
)

assert not PLAN_FILE.exists(), (
    "Pre-test plan already exists. Do not overwrite it."
)

assert not SHA_FILE.exists(), (
    "Pre-test SHA file already exists."
)

SEEDS = [42, 123, 2026]

MODEL_ORDER = ["M5", "M6", "C1", "C2"]

MODEL_DIRECTORIES = {
    "M5": "M05_DeepLabV3Plus_AP_Tooth_Seed{seed}",
    "M6": "M06_DeepLabV3Plus_AP_Tooth_RCF_Seed{seed}",
    "C1": (
        "C01_DeepLabV3Plus_AP_Tooth_AuxW06_Seed{seed}"
    ),
    "C2": (
        "C02_Shuffled_RCF_Control/"
        "C02_DeepLabV3Plus_AP_Tooth_ShuffledRCF_Seed{seed}"
    )
}

OUTPUT_CHANNELS = {
    "M5": 2,
    "M6": 3,
    "C1": 2,
    "C2": 3
}

REFERENCE_FILES = {
    42: "six_models_lesion_per_image_v1.csv",
    123: "four_models_seed123_lesion_per_image_v1.csv",
    2026: "four_models_seed2026_lesion_per_image_v1.csv"
}

EXPECTED_PRIMARY = {
    42: {
        "M5": (96, 358, 26),
        "M6": (94, 232, 28),
        "C1": (96, 219, 26),
        "C2": (97, 285, 25)
    },
    123: {
        "M5": (90, 164, 32),
        "M6": (90, 136, 32),
        "C1": (93, 319, 29),
        "C2": (94, 115, 28)
    },
    2026: {
        "M5": (95, 322, 27),
        "M6": (91, 161, 31),
        "C1": (99, 366, 23),
        "C2": (98, 365, 24)
    }
}

# ==============================================================
# 2. HASH FUNCTION
# ==============================================================

def file_sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()

# ==============================================================
# 3. VERIFY FROZEN VALIDATION ANALYSIS
# ==============================================================

SUMMARY_PREFIX = "three_seed_C2_M5_M6_C1"

REQUIRED_ANALYSIS = [
    f"{SUMMARY_PREFIX}_primary_per_seed_v1.csv",
    f"{SUMMARY_PREFIX}_model_summary_v1.csv",
    f"{SUMMARY_PREFIX}_effects_per_seed_v1.csv",
    f"{SUMMARY_PREFIX}_effect_summary_v1.csv",
    f"{SUMMARY_PREFIX}_bootstrap_primary_v1.csv",
    f"{SUMMARY_PREFIX}_sensitivity_v1.csv",
    f"{SUMMARY_PREFIX}_analysis_metadata_v1.json",
    "lesion_evaluation_protocol.json"
]

analysis_hashes = {}

for filename in REQUIRED_ANALYSIS:

    path = ANALYSIS / filename

    assert path.is_file(), (
        f"Missing completed analysis: {filename}"
    )

    analysis_hashes[filename] = file_sha256(path)

with open(
    ANALYSIS / "lesion_evaluation_protocol.json"
) as f:

    lesion_protocol = json.load(f)

assert lesion_protocol["version"] == "1.0"

assert lesion_protocol["dataset_split"] == (
    "validation_v1"
)

with open(
    ANALYSIS
    / f"{SUMMARY_PREFIX}_analysis_metadata_v1.json"
) as f:

    synthesis_metadata = json.load(f)

assert synthesis_metadata["seeds"] == SEEDS

assert set(
    synthesis_metadata["models"]
) == set(MODEL_ORDER)

assert (
    synthesis_metadata["held_out_test_used"]
    is False
)

FROZEN_PROTOCOL_SHA = (
    synthesis_metadata["frozen_protocol_sha256"]
)

assert FROZEN_PROTOCOL_SHA.startswith(
    "7381a8f2ac6364b1"
)

print("=" * 85)
print("PRE-TEST LOCK — VALIDATION ANALYSIS VERIFIED")
print("=" * 85)

print("Training seeds:", SEEDS)
print("Models:", MODEL_ORDER)

print("Completed synthesis files:", len(REQUIRED_ANALYSIS))

print(
    "Frozen lesion protocol:",
    FROZEN_PROTOCOL_SHA[:16]
)

# ==============================================================
# 4. VERIFY ORIGINAL DATA PREPROCESSING
# ==============================================================

DATALOADER_CONFIG = (
    ROOT
    / "03_Preprocessing_QC"
    / "dataloader_config_v1.json"
)

SPLIT_CONFIG = (
    ROOT
    / "02_Dataset_Splits"
    / "split_config_v1.json"
)

for path in [DATALOADER_CONFIG, SPLIT_CONFIG]:
    assert path.is_file()

with open(DATALOADER_CONFIG) as f:
    dataloader_config = json.load(f)

with open(SPLIT_CONFIG) as f:
    split_config = json.load(f)

assert dataloader_config["image_size"] == 512

assert dataloader_config["resize_method"] == (
    "aspect_ratio_preserving_padding"
)

assert dataloader_config["image_interpolation"] == (
    "bilinear"
)

assert dataloader_config["mask_interpolation"] == (
    "nearest"
)

assert dataloader_config["image_normalization"] == (
    "imagenet"
)

assert dataloader_config["classes"] == {
    "Tooth": 1,
    "RCF": 3,
    "AP": 9
}

assert split_config["patient_ids_available"] is False

# Do not read or inspect the Test split here.

# ==============================================================
# 5. VERIFY ALL 12 SAVED CHECKPOINTS
# ==============================================================

checkpoint_registry = []

input_evaluation_hashes = {}

for seed in SEEDS:

    reference_path = (
        ANALYSIS / REFERENCE_FILES[seed]
    )

    c1_path = (
        ANALYSIS
        / f"C01_Seed{seed}_lesion_per_image_v1.csv"
    )

    c2_path = (
        ANALYSIS
        / f"C02_Seed{seed}_lesion_per_image_v1.csv"
    )

    for path in [
        reference_path,
        c1_path,
        c2_path
    ]:
        assert path.is_file()
        input_evaluation_hashes[path.name] = (
            file_sha256(path)
        )

    reference = pd.read_csv(
        reference_path,
        dtype={"image_id": str}
    )

    c1_data = pd.read_csv(
        c1_path,
        dtype={"image_id": str}
    )

    c2_data = pd.read_csv(
        c2_path,
        dtype={"image_id": str}
    )

    sources = {
        "M5": reference,
        "M6": reference,
        "C1": c1_data,
        "C2": c2_data
    }

    print(f"\nVERIFYING SEED {seed}")

    for name in MODEL_ORDER:

        model_folder = (
            MODELS_ROOT
            / MODEL_DIRECTORIES[name].format(
                seed=seed
            )
        )

        checkpoint_path = (
            model_folder / "best_model.pt"
        )

        config_path = (
            model_folder / "training_config.json"
        )

        assert checkpoint_path.is_file(), (
            f"Missing checkpoint: {checkpoint_path}"
        )

        assert config_path.is_file(), (
            f"Missing config: {config_path}"
        )

        with open(config_path) as f:
            model_config = json.load(f)

        assert int(model_config["seed"]) == seed

        assert model_config["model"] == (
            "DeepLabV3Plus"
        )

        assert model_config["encoder"] == "resnet18"

        assert int(
            model_config["output_channels"]
        ) == OUTPUT_CHANNELS[name]

        # Read already-completed Validation records.
        # Never execute model inference here.

        source = sources[name]

        selected = source[
            (source["model"] == name)
            & np.isclose(
                source["IoU_threshold"], 0.10
            )
            & (source["min_pred_area"] == 1)
        ]

        assert len(selected) == 743

        assert selected["image_id"].is_unique

        actual = tuple(
            int(selected[col].sum())
            for col in ["TP", "FP", "FN"]
        )

        assert actual == (
            EXPECTED_PRIMARY[seed][name]
        ), (
            f"Validation mismatch: "
            f"{name} Seed{seed}"
        )

        expected_checkpoint_hashes = (
            source.loc[
                source["model"] == name,
                "checkpoint_sha256"
            ]
            .astype(str)
            .unique()
        )

        assert len(
            expected_checkpoint_hashes
        ) == 1

        recorded_hash = (
            expected_checkpoint_hashes[0]
        )

        # Verify that the checkpoint selected for Test
        # is EXACTLY the one used in Validation.

        actual_hash = file_sha256(
            checkpoint_path
        )

        assert actual_hash == recorded_hash, (
            f"Checkpoint differs from Validation: "
            f"{name} Seed{seed}. "
            "Stop before opening Test."
        )

        protocol_hashes = (
            source.loc[
                source["model"] == name,
                "protocol_sha256"
            ]
            .astype(str)
            .unique()
        )

        assert len(protocol_hashes) == 1

        assert protocol_hashes[0] == (
            FROZEN_PROTOCOL_SHA
        )

        checkpoint_registry.append({
            "model_id": name,
            "seed": seed,
            "architecture": "DeepLabV3Plus",
            "encoder": "resnet18",
            "output_channels": OUTPUT_CHANNELS[name],
            "checkpoint_relative_path": str(
                checkpoint_path.relative_to(ROOT)
            ),
            "checkpoint_sha256": actual_hash,
            "config_relative_path": str(
                config_path.relative_to(ROOT)
            ),
            "config_sha256": file_sha256(
                config_path
            ),
            "checkpoint_selection": (
                "minimum_AP_validation_loss"
            )
        })

        print(
            f"  {name}: CHECKPOINT VERIFIED | "
            f"{actual_hash[:16]}"
        )

assert len(checkpoint_registry) == 12

assert len({
    (
        row["model_id"],
        row["seed"]
    )
    for row in checkpoint_registry
}) == 12

print("\nALL 12 CHECKPOINTS: VERIFIED")

# ==============================================================
# 6. DEFINE FROZEN TEST ANALYSIS PLAN
# ==============================================================

PLAN = {

    "plan_id": (
        "RCF_Aware_AP_HeldOut_Test_Pretest_Lock_v1"
    ),

    "plan_status": (
        "FROZEN_BEFORE_HELD_OUT_TEST"
    ),

    "created_at_utc": (
        datetime.now(timezone.utc).isoformat()
    ),

    "project": (
        "RCF-Aware Multi-Task Segmentation "
        "for Apical Periodontitis on "
        "Periapical Radiographs"
    ),

    "source_dataset": "PRAD",

    "test_split": {
        "version": "v1",

        "relative_csv_path": (
            "02_Dataset_Splits/test_v1.csv"
        ),

        "expected_images": 743,

        "expected_strata": {
            "AP_RCF": 43,
            "AP_ONLY": 56,
            "RCF_ONLY": 245,
            "NEITHER": 399
        },

        "patient_ids_available": False,

        "patient_disjointness_verified": False,

        "test_csv_accessed_during_lock": False,

        "test_images_accessed_during_lock": False
    },

    "preprocessing": {
        "image_size": 512,

        "resize_method": (
            "aspect_ratio_preserving_padding"
        ),

        "image_interpolation": "bilinear",

        "mask_interpolation": "nearest",

        "normalization": "imagenet",

        "mask_class_ids": {
            "AP": 9,
            "Tooth": 1,
            "RCF": 3
        },

        "data_augmentation_during_test": False,

        "dataloader_config_sha256": (
            file_sha256(DATALOADER_CONFIG)
        ),

        "split_config_sha256": (
            file_sha256(SPLIT_CONFIG)
        )
    },

    "fixed_models": MODEL_ORDER,

    "training_seeds": SEEDS,

    "checkpoints": checkpoint_registry,

    "model_selection": {
        "selection_dataset": "validation_v1",

        "checkpoint_rule": (
            "minimum_AP_validation_loss"
        ),

        "test_based_checkpoint_selection": False,

        "test_based_seed_selection": False,

        "retraining_after_test": False,

        "threshold_tuning_after_test": False
    },

    "primary_analysis": {
        "comparison": "M6_minus_C2",

        "endpoint": "Lesion_F1",

        "direction": (
            "positive_difference_favors_M6"
        ),

        "evaluation_unit": (
            "connected_AP_component"
        ),

        "interpretation": (
            "Contrast correct RCF supervision "
            "with group-preserving spatially "
            "shuffled RCF supervision"
        ),

        "report_per_seed": True,

        "report_three_seed_mean_and_SD": True,

        "between_seed_inference": (
            "descriptive_only_n_equals_3"
        )
    },

    "secondary_comparisons": [
        {
            "comparison": "M6_minus_M5",

            "purpose": (
                "Incremental association of adding "
                "the RCF auxiliary task"
            )
        },
        {
            "comparison": "M6_minus_C1",

            "purpose": (
                "Comparison against increased "
                "Tooth auxiliary loss weight"
            )
        }
    ],

    "lesion_evaluation": {
        "probability_threshold": 0.50,

        "prediction_rule": (
            "sigmoid_probability_greater_or_equal_0.50"
        ),

        "connectivity": 8,

        "component_matching": (
            "Hungarian_maximum_cardinality_then_IoU"
        ),

        "primary_IoU_threshold": 0.10,

        "primary_min_predicted_area_pixels": 1,

        "sensitivity_settings": [
            {"IoU_threshold": 0.50, "min_area": 1},
            {"IoU_threshold": 0.10, "min_area": 16},
            {"IoU_threshold": 0.50, "min_area": 16}
        ],

        "primary_metrics": [
            "Lesion_F1"
        ],

        "additional_metrics": [
            "Lesion_Recall",
            "Lesion_Precision",
            "TP",
            "FP",
            "FN",
            "FP_Components_per_Image",
            "AP_Negative_FP_Image_Rate"
        ],

        "GT_components_are_proxy_lesions": True,

        "validation_protocol_sha256": (
            FROZEN_PROTOCOL_SHA
        )
    },

    "subgroup_analysis": {
        "groups": [
            "AP_RCF",
            "AP_ONLY",
            "RCF_ONLY",
            "NEITHER",
            "ALL_NEGATIVE"
        ],

        "ALL_NEGATIVE_definition": (
            "RCF_ONLY_plus_NEITHER"
        ),

        "subgroup_results": (
            "supportive_or_exploratory"
        )
    },

    "pixel_level_analysis": {
        "AP_Dice": (
            "mean_binary_Dice_on_AP_positive_images"
        ),

        "AP_IoU": (
            "mean_binary_IoU_on_AP_positive_images"
        ),

        "Tooth_Dice": (
            "mean_on_Tooth_positive_images"
        ),

        "RCF_Dice": (
            "mean_on_RCF_positive_masks_when_available"
        ),

        "AP_negative_FP_image_rate": True,

        "pixel_probability_threshold": 0.50
    },

    "bootstrap": {
        "method": (
            "paired_stratified_image_level_percentile"
        ),

        "resamples_per_seed": 5000,

        "rng_seed": 20261019,

        "resampling_unit": "test_image",

        "stratification": (
            "fixed_AP_RCF_AP_ONLY_RCF_ONLY_NEITHER"
        ),

        "same_resampled_images_across_models": True,

        "confidence_level": 0.95,

        "report_checkpoint_conditional_CIs": True,

        "pool_draws_across_training_seeds": False,

        "patient_level_bootstrap": False,

        "patient_level_inference": False
    },

    "reporting": {
        "all_12_checkpoints_reported": True,

        "three_seed_model_mean_SD": True,

        "paired_per_seed_differences": True,

        "three_seed_mean_paired_difference": True,

        "sensitivity_analyses_reported": True,

        "negative_subgroup_analysis_reported": True,

        "pixel_and_lesion_metrics_reported": True,

        "no_post_hoc_metric_selection": True,

        "no_claim_of_patient_independence": True,

        "no_unqualified_causal_claim": True
    },

    "validation_provenance": {
        "analysis_file_sha256": analysis_hashes,

        "per_image_evaluation_sha256": (
            input_evaluation_hashes
        )
    },

    "limitations": [
        "Only three training seeds",
        "No verified patient-disjoint splitting",
        "Connected components are proxy lesions",
        "Image bootstrap is not patient-level inference",
        "C2 shuffle preserves within-group prevalence",
        "Training effects may depend on initialization",
        "Pretest plan was developed using Validation results"
    ],

    "held_out_test_accessed": False
}

# ==============================================================
# 7. VERIFY PRE-TEST PLAN INTERNALLY
# ==============================================================

assert PLAN["primary_analysis"]["comparison"] == (
    "M6_minus_C2"
)

assert PLAN["primary_analysis"]["endpoint"] == (
    "Lesion_F1"
)

assert PLAN["training_seeds"] == SEEDS

assert len(PLAN["checkpoints"]) == 12

assert PLAN["bootstrap"]["resamples_per_seed"] == 5000

assert PLAN["test_split"]["expected_images"] == 743

assert sum(
    PLAN["test_split"]["expected_strata"].values()
) == 743

assert (
    PLAN["held_out_test_accessed"]
    is False
)

assert PLAN["model_selection"][
    "test_based_seed_selection"
] is False

assert PLAN["model_selection"][
    "threshold_tuning_after_test"
] is False

print("\n" + "=" * 85)
print("FROZEN TEST ANALYSIS PLAN — PRE-SAVE REVIEW")
print("=" * 85)

print("Primary comparison: M6 minus C2")
print("Primary endpoint: Lesion F1")

print("Secondary: M6 minus M5")
print("Secondary control: M6 minus C1")

print("Models:", len(MODEL_ORDER))
print("Training seeds:", len(SEEDS))
print("Frozen checkpoints:", len(checkpoint_registry))

print("Primary IoU: 0.10")
print("Primary minimum area: 1")
print("Probability threshold: 0.50")

print("Bootstrap per seed: 5000")
print("Held-out Test accessed: NO")

# ==============================================================
# 8. ATOMICALLY SAVE PLAN AND SHA256
# ==============================================================

with tempfile.TemporaryDirectory(
    prefix=".pretest_lock_stage_",
    dir=ANALYSIS
) as temp_directory:

    temporary_directory = Path(temp_directory)

    temp_plan = (
        temporary_directory / PLAN_FILE.name
    )

    temp_sha = (
        temporary_directory / SHA_FILE.name
    )

    with open(
        temp_plan,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            PLAN,
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(
        temp_plan,
        encoding="utf-8"
    ) as f:

        saved_plan = json.load(f)

    assert saved_plan == PLAN

    PLAN_SHA = file_sha256(temp_plan)

    with open(
        temp_sha,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{PLAN_SHA}  {PLAN_FILE.name}\n"
        )

    assert not PLAN_FILE.exists()
    assert not SHA_FILE.exists()

    os.replace(temp_plan, PLAN_FILE)
    os.replace(temp_sha, SHA_FILE)

# ==============================================================
# 9. FINAL FILE INTEGRITY VERIFICATION
# ==============================================================

assert PLAN_FILE.is_file()
assert SHA_FILE.is_file()

assert file_sha256(PLAN_FILE) == PLAN_SHA

with open(
    PLAN_FILE,
    encoding="utf-8"
) as f:

    final_plan = json.load(f)

assert final_plan["held_out_test_accessed"] is False

assert len(final_plan["checkpoints"]) == 12

print("\n" + "=" * 90)
print("PRE-TEST ANALYSIS PLAN FROZEN SUCCESSFULLY")
print("=" * 90)

print("Plan file:", PLAN_FILE.name)
print("SHA file:", SHA_FILE.name)

print("\nPLAN SHA256:")
print(PLAN_SHA)

print("\nPrimary comparison: M6 minus C2")
print("Secondary comparison: M6 minus M5")
print("Control comparison: M6 minus C1")

print("Frozen checkpoints:", 12)
print("Validation protocol: VERIFIED")

print("\nTraining: NONE")
print("Model inference: NONE")
print("Test CSV opened: NO")
print("Test images opened: NO")
print("Test masks opened: NO")
print("Previous results modified: NONE")

print("\nFINAL STATUS: PASSED")