# Suggested order of execution

1. **Prerequisites / dataset preprocessing** — obtain authorized PRAD images and mask files, original processing/split manifests and expected Drive project layout; review `DATA_ACCESS.md` and the historical notebooks in `notebooks/data_preparation/`. The split code was explicitly provisional: exact manuscript reproduction requires the original frozen manifests.
2. **Training** — use one training notebook for each combination of M5/M6/C1/C2 and seed 42/123/2026. These are historical Colab training cells, each exported independently from the original research notebook; do not execute all 12 blindly in one session.
3. **C2 manifest** — run `notebooks/control_design/C2_fixed_shuffle_manifest.ipynb` only when building a new controlled experiment. For exact reproduction of the reported runs, use the **original saved shuffle manifest** and verify its SHA256 (`70e348b70b54bbfd93d66db9f843f3f2d3cc45bfec66eb2d23cafc3bc563605a`).
4. **Pre-test plan** — `notebooks/protocol/Freeze_pretest_analysis_plan.ipynb` documents the original protocol-lock code. **Do not re-freeze the plan retrospectively**.
5. **Frozen held-out test** — with original source data, checkpoints, plan and integrity audit, use `scripts/evaluation/01_heldout_test_inference_v1.py` then `02_paired_stratified_bootstrap_v1.py` and `03_three_seed_final_synthesis_v1.py`.
6. **Exploratory analyses** — `scripts/evaluation/04_exploratory_error_analysis_v1.py`; these analyses are post-hoc and not confirmatory.
7. **Outputs** — `scripts/figures/05_manuscript_tables_figures_v1.py`, `scripts/tables/build_supplementary_tables_S1_S5_v1.py`, optional qualitative figures.

**Important:** The source scripts assume the original Drive directory hierarchy. Paths are intentionally not replaced by fabricated local test data. Use a separate private working copy to avoid overwriting actual research artifacts.
