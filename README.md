# RCF-aware multitask segmentation for apical periodontitis — code archive

**Associated manuscript:** *Root Canal Filling–Aware Multitask Segmentation for Apical Periodontitis on Periapical Radiographs: A Three-Seed Controlled Evaluation* (manuscript citation and DOI to be added following publication).

## Scope and scientific provenance

This repository provides the **historical experimental source code** supplied by the study authors for the four main configurations (M5, M6, C1, C2), each with seeds **42, 123, 2026**; the C2 RCF-shuffling manifest generator; the pre-test analysis-plan lock and audit scripts; held-out test inference, paired stratified image-level bootstrap, synthesis, exploratory analyses, manuscript plots, and supplementary table generation.

Files in `original_code/` preserve the source of the selected **executed Colab training cells**. Individual `notebooks/training/` notebooks contain those same source cells plus a separately marked release-time setup cell; their outputs have been removed. These are a cleaned **research code archive**, not a claim that the entire experiment is executable without its restricted inputs or can be rerun with a single command. Some experimental scripts support resuming from existing checkpoints and require a copy of the original project layout.

The archive does **not** contain clinical radiographs, image-level annotations, patient information, model weights, original PRAD files, or private Google Drive content. It also does **not** contain the original frozen evaluation-plan JSON, split manifests, or C2 shuffle assignment CSV: the source code refers to these original artifacts and reports their integrity checks where applicable. **Do not invent or regenerate a supposedly identical pre-test freeze after viewing held-out test data.** These original frozen files must be supplied separately by the corresponding author if authorized to share.

## Repository map

- `notebooks/data_preparation/` — historical image/mask inventory, duplicate review, patient metadata availability audit, provisional split and dataloader preparation.
- `notebooks/training/` — 12 code-only Colab notebooks, one per model/seed; contain historical source code and a separately labeled setup cell.
- `original_code/training/` — untouched historical source code for each of those 12 runs, exported as Python files (comment header added only).
- `notebooks/control_design/` — historical code to generate and audit the C2 group-preserving RCF-mask derangement.
- `notebooks/protocol/` — pre-test plan-freeze procedure and held-out test input integrity audit code.
- `original_code/data_preparation/` — matching original code extracted from the preprocessing cells.
- `scripts/evaluation/` — frozen checkpoint inference, paired bootstraps, summary, exploratory error analysis.
- `scripts/figures/` — quantitative visualizations, illustrative case generation, and boundary-redaction **draft** procedure.
- `scripts/tables/` — supplemental S1–S5 table assembly.
- `scripts/audits/` — frozen configuration and provenance checks.
- `docs/` — implementation order, original-code provenance, data-access requirements, release checklist, and statement template.

## Input access and prerequisites

**PRAD is access-controlled.** Obtain the **5,000-image researcher subset** and masks from the PRAD custodians, adhering to their data-use conditions. See `DATA_ACCESS.md`.

The historical scripts expect Google Colab with a mounted Drive and paths beginning with:

```text
/content/drive/MyDrive/PRAD_MultiTask_Segmentation
/content/drive/MyDrive/RCF_Aware_AP_Segmentation_Research
```

The latter contains processing manifests, annotation splits, checkpoints, the frozen test analysis plan, and previously generated per-image metric files. Code cannot replace those input files. The original dataset and project structure were not redistributed here.

Install `segmentation-models-pytorch==0.5.0`; the other Python packages are listed in `requirements.txt`. Compatible CUDA/PyTorch installation depends on the runtime. `pip install -r requirements.txt` is a starting point but **not** an exact, validated environment lock.

Run **one experiment per fresh Colab session** (mount Drive; set GPU runtime for training/inference), checking that private project paths exist before proceeding. The scripts were originally executed in Colab and are presented for methodological transparency; they have not been rerun as part of this publication archive preparation.

## Experimental design

| Model | Prediction targets | Supervision | Auxiliary weights |
|---|---|---|---|
| M5 | AP, tooth | spatially correct | tooth 0.3 |
| M6 | AP, tooth, RCF | spatially correct | tooth 0.3; RCF 0.3 |
| C1 | AP, tooth | spatially correct | tooth 0.6 |
| C2 | AP, tooth, RCF | training RCF mask shuffled within AP/RCF groups | tooth 0.3; RCF 0.3 |

All configurations use DeepLabV3+ with ResNet18/ImageNet pretrained weights at 512 x 512 input resolution. Original training settings are retained in the scripts. Image-level and lesion-level results in the manuscript were computed from a frozen internal hold-out set: 743 images, 99 AP-positive, and 116 annotated AP connected components. The test set **is not an externally acquired cohort**. Patient-level split independence cannot be verified without patient identifiers.

## Reproduction sequence

1. Obtain authorized PRAD access and construct or obtain the **original versioned dataset** and split manifests. Do not generate different splits and compare them as though they were the same experiment.
2. In Colab, set up the Drive mounts and install the library dependencies.
3. Run the original M5/M6/C1 scripts for the requested seed if full retraining is desired. For C2, use the **original frozen shuffled mask manifest** when reproducing the reported checkpoints; the manifest-generation script is included to explain the method.
4. Select model checkpoints using minimum AP validation loss as specified in the source code. Do not consult Test for checkpoint selection.
5. Use the **original** frozen pre-test analysis plan and checkpoint hashes. Run held-out evaluation and image-level bootstrap scripts only with the registered checkpoints and unchanged protocol.
6. Generate manuscript figures/tables and the supplementary tables from the saved analysis outputs.

The original frozen evaluation plan SHA256 was reported as:
`35dbf447bbabb1cec4bacebe5ade8c90d9d9fecf4a6e16f40db4ac1add635800`.
This value is **provenance metadata**, not a replacement for the JSON file.

## Disclosure and ethical considerations

- No PRAD radiographs, masks or patient data are included here. Do not redistribute them without authorization.
- Figure redaction is a **draft technical tool**, not proof of de-identification, consent, or permission to republish radiographs. Human review is required.
- Some source code retains project-specific Drive directory names to reflect the historical executed workflow, but contains no dataset bytes or model checkpoints.
- `CODE_AVAILABILITY_STATEMENT.md` provides submission text after the repository URL has been confirmed. Avoid claiming the code is publicly available until publication has actually occurred.
- An explicit open-source license is **not** supplied because the rights holder has not selected one; see `LICENSE_DECISION.md`.

## Citation

Please cite the associated manuscript once its final bibliographic record is available, and cite the PRAD data creators in accordance with their access conditions. See `CITATION.md`.
