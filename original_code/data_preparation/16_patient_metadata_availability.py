# SOURCE CELL 16 — original, method logic unchanged.
from pathlib import Path
import os
import re
import pandas as pd

# ============================================================
# STEP: PATIENT METADATA AVAILABILITY AUDIT
# Original PRAD dataset remains read-only.
# ============================================================

DRIVE_ROOT = Path(REAL_MYDRIVE)

DATASET_ROOT = (
    DRIVE_ROOT / "PRAD_MultiTask_Segmentation"
)

PROJECT_ROOT = (
    DRIVE_ROOT / "RCF_Aware_AP_Segmentation_Research"
)

ANALYSIS_DIR = PROJECT_ROOT / "01_Dataset_Analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = (
    ANALYSIS_DIR
    / "Duplicate_Check"
    / "PRAD_duplicate_group_manifest.csv"
)

# ============================================================
# 1. VERIFY PREVIOUS MANIFEST
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"image_id": str}
)

assert len(manifest) == 5000
assert manifest["image_id"].is_unique

print("=" * 60)
print("PRAD METADATA AVAILABILITY AUDIT")
print("=" * 60)

print("Total images:", len(manifest))
print(
    "Duplicate groups:",
    manifest["duplicate_group_id"].nunique()
)

# ============================================================
# 2. FIND DOCUMENTATION AND METADATA FILES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".csv", ".tsv", ".txt", ".json",
    ".xlsx", ".xls", ".xml", ".parquet", ".pdf"
}

metadata_files = []

for current, dirs, files in os.walk(DATASET_ROOT):

    # Do not scan original image or mask contents
    dirs[:] = [
        d for d in dirs
        if d.lower() not in {
            "image", "images",
            "label", "labels",
            "mask", "masks"
        }
    ]

    for filename in files:

        path = Path(current) / filename

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            metadata_files.append(path)

print("\nMetadata/documentation files found:",
      len(metadata_files))

# ============================================================
# 3. CHECK TABULAR COLUMN NAMES
# ============================================================

PATIENT_PATTERN = re.compile(
    r"patient|subject|person|participant|"
    r"patient.?id|subject.?id|case.?id",
    re.IGNORECASE
)

records = []

for path in sorted(metadata_files):

    relative_path = str(
        path.relative_to(DATASET_ROOT)
    )

    matched_columns = []
    status = "Not inspected"

    if path.suffix.lower() in {".csv", ".tsv"}:

        separator = (
            "\t" if path.suffix.lower() == ".tsv"
            else ","
        )

        try:
            header = pd.read_csv(
                path,
                sep=separator,
                nrows=0
            )

            matched_columns = [
                col for col in header.columns
                if PATIENT_PATTERN.search(str(col))
            ]

            status = "Header inspected"

        except Exception:
            status = "Could not inspect header"

    records.append({
        "relative_path": relative_path,
        "extension": path.suffix.lower(),
        "status": status,
        "possible_patient_id_columns": (
            ", ".join(matched_columns)
        )
    })

report = pd.DataFrame(records, columns=[
    "relative_path",
    "extension",
    "status",
    "possible_patient_id_columns"
])

# ============================================================
# 4. SAVE REPORT IN RESEARCH DIRECTORY
# ============================================================

OUTPUT_PATH = (
    ANALYSIS_DIR / "PRAD_metadata_availability_audit.csv"
)

report.to_csv(OUTPUT_PATH, index=False)

# ============================================================
# 5. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("METADATA FILE INVENTORY")
print("=" * 60)

if not report.empty:
    print(
        report[
            ["relative_path", "extension", "status"]
        ].to_string(index=False)
    )
else:
    print("No separate metadata files found.")

patient_candidates = report[
    report["possible_patient_id_columns"] != ""
]

print("\n" + "=" * 60)
print("PATIENT IDENTIFIER CHECK")
print("=" * 60)

print(
    "Files containing possible patient-ID columns:",
    len(patient_candidates)
)

if not patient_candidates.empty:
    print(
        patient_candidates[
            [
                "relative_path",
                "possible_patient_id_columns"
            ]
        ].to_string(index=False)
    )
else:
    print(
        "No patient-ID columns detected in inspected "
        "CSV/TSV headers."
    )
    print(
        "This does NOT prove patient identifiers "
        "are unavailable elsewhere."
    )

print("\nReport saved:", OUTPUT_PATH)
print("Original PRAD dataset: NOT MODIFIED")
print("No train/validation/test split created.")