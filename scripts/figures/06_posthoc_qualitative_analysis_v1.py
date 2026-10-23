# ============================================================================
# HELD-OUT TEST: POST-HOC QUALITATIVE RADIOGRAPH FIGURES — VERSION 1
# One standalone Google Colab cell. Requires GPU, smp==0.5.0, mounted Drive.
#
# This is ILLUSTRATION ONLY, after frozen Test evaluation:
#  - Use only the 36 already-selected case-manifest rows, without reselection.
#  - Reconstruct the exact frozen PIL / ImageNet preprocessing.
#  - Re-run M6 and C2 best checkpoints for selected images only.
#  - Verify TP / FP / FN against published test per-image results BEFORE saving.
#  - Save side-by-side original, ground truth, M6, C2 figures.
#  - NEVER recalculate model rankings, thresholds, or confirmatory endpoints.
#  - NO TRAINING, NO MODEL RETUNING, NO OVERWRITING PRIOR ARTIFACTS.
# ============================================================================

import os, gc, json, hashlib, tempfile, re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- 1. Frozen file locations and environment -------------------------------
MYDRIVE = Path('/content/drive/MyDrive')
if not MYDRIVE.is_dir():
    from google.colab import drive
    drive.mount('/content/drive')
assert MYDRIVE.is_dir(), 'Google Drive is not mounted.'
assert torch.cuda.is_available(), 'Select GPU in Colab Runtime settings.'
assert smp.__version__ == '0.5.0', 'This study used segmentation_models_pytorch==0.5.0.'

DEVICE = torch.device('cuda')
ROOT = MYDRIVE / 'RCF_Aware_AP_Segmentation_Research'
RAW = MYDRIVE / 'PRAD_MultiTask_Segmentation'
ANALYSIS = ROOT / '06_Lesion_Level_Analysis'
TEST_OUT = ROOT / '07_Heldout_Test_Evaluation_v1'
MANUSCRIPT = ROOT / '08_Manuscript_Tables_Figures_v1'
FIG_DIR = MANUSCRIPT / 'Qualitative_AP_M6_C2_Exploratory_v1'

PLAN_PATH = ANALYSIS / 'heldout_test_analysis_plan_frozen_v1.json'
TEST_AUDIT = ANALYSIS / 'heldout_test_input_integrity_audit_v1.json'
PER_PATH = TEST_OUT / 'heldout_test_12models_lesion_per_image_v1.csv'
CASE_PATH = TEST_OUT / 'heldout_test_error_analysis_exploratory_v1_illustrative_cases.csv'
ERROR_METADATA = TEST_OUT / 'heldout_test_error_analysis_exploratory_v1_metadata.json'
TEST_METADATA = TEST_OUT / 'heldout_test_12models_metadata_v1.json'
TEST_SPLIT = ROOT / '02_Dataset_Splits/test_v1.csv'
EXPECTED_PLAN_SHA = '35dbf447bbabb1cec4bacebe5ade8c90d9d9fecf4a6e16f40db4ac1add635800'

OUT_MANIFEST = FIG_DIR / 'qualitative_rendered_cases_v1.csv'
OUT_METADATA = FIG_DIR / 'qualitative_figures_metadata_v1.json'
OUT_LEGEND = FIG_DIR / 'qualitative_figure_legend_v1.txt'

for p in [PLAN_PATH, TEST_AUDIT, PER_PATH, CASE_PATH, ERROR_METADATA,
          TEST_METADATA, TEST_SPLIT]:
    assert p.is_file(), f'Missing frozen input: {p}'
assert MANUSCRIPT.is_dir(), 'Publication tables/figures folder has not been created.'
assert not FIG_DIR.exists(), (f'Target figure folder already exists: {FIG_DIR}. '
                              'Do not overwrite; review existing outputs first.')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


assert sha256_file(PLAN_PATH) == EXPECTED_PLAN_SHA, 'Frozen pre-test plan changed.'
with open(PLAN_PATH, encoding='utf-8') as f:
    plan = json.load(f)
with open(TEST_AUDIT, encoding='utf-8') as f:
    audit = json.load(f)
with open(ERROR_METADATA, encoding='utf-8') as f:
    error_meta = json.load(f)
with open(TEST_METADATA, encoding='utf-8') as f:
    test_meta = json.load(f)

assert plan['primary_analysis']['comparison'] == 'M6_minus_C2'
assert plan['primary_analysis']['endpoint'] == 'Lesion_F1'
assert plan['training_seeds'] == [42, 123, 2026]
assert plan['lesion_evaluation']['probability_threshold'] == 0.5
assert plan['lesion_evaluation']['primary_IoU_threshold'] == 0.10
assert plan['lesion_evaluation']['primary_min_predicted_area_pixels'] == 1
assert plan['lesion_evaluation']['connectivity'] == 8
assert plan['lesion_evaluation']['component_matching'] == 'Hungarian_maximum_cardinality_then_IoU'
assert audit['status'] == 'PASSED'
assert audit['frozen_plan_sha256'] == EXPECTED_PLAN_SHA
assert error_meta['frozen_plan_sha256'] == EXPECTED_PLAN_SHA
assert error_meta['case_selection_is_post_hoc'] is True
assert error_meta['illustrations_are_inferential_evidence'] is False
assert test_meta['frozen_pretest_plan_sha256'] == EXPECTED_PLAN_SHA
assert sha256_file(TEST_SPLIT) == audit['test_csv_sha256']

# -- 2. Audit the saved scenario manifest against published Test results ----
cases = pd.read_csv(CASE_PATH, dtype={'image_id': str})
published = pd.read_csv(PER_PATH, dtype={'image_id': str})
test_df = pd.read_csv(TEST_SPLIT, dtype={'image_id': str})

assert len(cases) == int(error_meta['rows']['illustrations']) == 36
assert len(published) == 35664
assert len(test_df) == 743 and test_df['image_id'].is_unique
assert set(cases['seed']) == {42, 123, 2026}
assert cases['selection_status'].eq('post_hoc_descriptive_illustration_only').all()
assert not cases.duplicated(['seed', 'Scenario', 'image_id']).any()
assert set(cases['image_id']).issubset(set(test_df['image_id']))
assert published['pretest_plan_sha256'].eq(EXPECTED_PLAN_SHA).all()
assert not published.duplicated(['seed', 'model', 'image_id', 'IoU_threshold', 'min_pred_area']).any()
primary = published[np.isclose(published['IoU_threshold'], 0.10) &
                    published['min_pred_area'].eq(1) &
                    published['model'].isin(['M6', 'C2'])].copy()
assert len(primary) == 2 * 3 * 743
published_index = primary.set_index(['seed', 'model', 'image_id'], verify_integrity=True)
group_by_id = test_df.set_index('image_id')['stratum']

for row in cases.itertuples(index=False):
    assert group_by_id.loc[str(row.image_id)] == row.group
    for model_name in ('M6', 'C2'):
        pub = published_index.loc[(int(row.seed), model_name, str(row.image_id))]
        assert int(pub['GT_components']) == int(row.GT_components)
        assert str(pub['group']) == str(row.group)
        for metric in ('TP', 'FP', 'FN'):
            assert int(pub[metric]) == int(getattr(row, model_name + '_' + metric))

registry = {(r['model_id'], int(r['seed'])): r for r in plan['checkpoints']}
assert len(registry) == 12
for seed in (42, 123, 2026):
    for model_name in ('M6', 'C2'):
        r = registry[(model_name, seed)]
        checkpoint_file = ROOT / r['checkpoint_relative_path']
        config_file = ROOT / r['config_relative_path']
        assert checkpoint_file.is_file() and config_file.is_file()
        assert sha256_file(checkpoint_file) == r['checkpoint_sha256']
        assert sha256_file(config_file) == r['config_sha256']
        assert int(r['output_channels']) == 3
        assert published_index.xs((seed, model_name), level=('seed', 'model'))[
            'checkpoint_sha256'].eq(r['checkpoint_sha256']).all()

print('=' * 94)
print('QUALITATIVE RADIOGRAPH FIGURES — FROZEN PROVENANCE VERIFIED')
print('=' * 94)
print('Pre-Test plan SHA256:', EXPECTED_PLAN_SHA)
print('Fixed, preselected POST-HOC illustration rows:', len(cases))
print('Test results: already published, unchanged')
print('6 checkpoint SHA256 hashes: VERIFIED | No metric/threshold changes')

# -- 3. EXACT frozen image and mask transform -------------------------------
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).reshape(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).reshape(3, 1, 1)
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def load_image_and_mask(image_id):
    image_path = RAW / 'image' / f'{image_id}.jpg'
    label_path = RAW / 'label' / f'{image_id}.png'
    assert image_path.is_file() and label_path.is_file()
    with Image.open(image_path) as src:
        img = src.convert('RGB')
    with Image.open(label_path) as src:
        assert src.mode in ('P', 'L'), (image_id, src.mode)
        label = src.copy()
    w, h = img.size
    assert (w, h) == label.size and w > 0 and h > 0
    scale = 512.0 / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    dx, dy = (512 - nw) // 2, (512 - nh) // 2
    im_canvas = Image.new('RGB', (512, 512), (0, 0, 0))
    gt_canvas = Image.new(label.mode, (512, 512), 0)
    im_canvas.paste(img.resize((nw, nh), Image.Resampling.BILINEAR), (dx, dy))
    gt_canvas.paste(label.resize((nw, nh), Image.Resampling.NEAREST), (dx, dy))
    image_uint8 = np.asarray(im_canvas, dtype=np.uint8).copy()
    gt_indices = np.asarray(gt_canvas, dtype=np.uint8).copy()
    tensor = torch.from_numpy(image_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = ((tensor - MEAN) / STD).contiguous()
    return image_uint8, tensor, gt_indices == 9, gt_indices == 3


# -- 4. Exact frozen Hungarian lesion matching (for image-level audit) -----
def frozen_lesion_counts(gt, pred):
    gt_l, ng = ndi.label(gt.astype(np.uint8), structure=STRUCTURE)
    pr_l, np_raw = ndi.label(pred.astype(np.uint8), structure=STRUCTURE)
    if ng == 0:
        return (0, int(np_raw), 0, 0, int(np_raw))
    if np_raw == 0:
        return (0, 0, int(ng), int(ng), 0)
    gt_area = np.bincount(gt_l.ravel(), minlength=ng + 1)
    pr_area = np.bincount(pr_l.ravel(), minlength=np_raw + 1)
    pair = gt_l.astype(np.int64).ravel() * (np_raw + 1) + pr_l.astype(np.int64).ravel()
    joint = np.bincount(pair, minlength=(ng + 1) * (np_raw + 1)).reshape(ng + 1, np_raw + 1)
    inter = joint[1:, 1:].astype(np.float64)
    union = gt_area[1:, None] + pr_area[None, 1:] - inter
    ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    valid = ious >= 0.10
    # Area filter = 1 pixel: every nonzero predicted component is kept.
    rewards = np.zeros((ng, np_raw + ng), dtype=np.float64)
    rewards[:, :np_raw] = np.where(valid, (ng + 1.0) + ious, -1.0)
    rr, cc = linear_sum_assignment(-rewards)
    tp = sum(int(c < np_raw and bool(valid[r, c])) for r, c in zip(rr, cc))
    return int(tp), int(np_raw - tp), int(ng - tp), int(ng), int(np_raw)


# -- 5. Frozen model loading and overlays ----------------------------------
def load_model(entry):
    model = smp.DeepLabV3Plus(
        encoder_name='resnet18', encoder_weights=None,
        encoder_output_stride=16, in_channels=3,
        classes=int(entry['output_channels']), activation=None
    ).to(DEVICE)
    checkpoint = torch.load(ROOT / entry['checkpoint_relative_path'],
                            map_location='cpu', weights_only=True)
    state = next((checkpoint[k] for k in ('model_state', 'model_state_dict', 'state_dict')
                  if isinstance(checkpoint, dict) and k in checkpoint and isinstance(checkpoint[k], dict)), None)
    if state is None and isinstance(checkpoint, dict) and checkpoint and all(
            torch.is_tensor(v) for v in checkpoint.values()):
        state = checkpoint
    assert state is not None, 'Unknown checkpoint structure; STOP without modifying evaluation.'
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def overlay(rgb, mask, color, alpha=0.48):
    arr = rgb.astype(np.float32).copy()
    arr[mask] = (1 - alpha) * arr[mask] + alpha * np.asarray(color, dtype=float)
    return arr.astype(np.uint8)


def gt_outline(ax, gt):
    if gt.any():
        ax.contour(gt.astype(float), levels=[0.5], colors=['#1aff66'], linewidths=1.25)


def render_case(row, rgb, ap_gt, rcf_gt, ap_m6, ap_c2, path):
    # Colors: AP GT=green; RCF GT=cyan; M6 AP=orange; C2 AP=magenta.
    base = rgb.copy()
    gt_view = overlay(base, rcf_gt, (25, 205, 235), alpha=0.58)
    gt_view = overlay(gt_view, ap_gt, (35, 240, 100), alpha=0.64)
    m6_view = overlay(base, ap_m6, (255, 128, 24), alpha=0.60)
    c2_view = overlay(base, ap_c2, (235, 55, 190), alpha=0.60)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), constrained_layout=True)
    titles = [
        'Input radiograph',
        'Ground truth\nAP=green, RCF=cyan',
        f'M6: correct RCF\nTP={int(row.M6_TP)}  FP={int(row.M6_FP)}  FN={int(row.M6_FN)}',
        f'C2: shuffled RCF\nTP={int(row.C2_TP)}  FP={int(row.C2_FP)}  FN={int(row.C2_FN)}',
    ]
    for idx, (ax, im, title) in enumerate(zip(axes, [base, gt_view, m6_view, c2_view], titles)):
        ax.imshow(im)
        if idx >= 2:
            gt_outline(ax, ap_gt)
        ax.set_title(title, fontsize=10, fontweight='medium')
        ax.axis('off')
    scenario = str(row.Scenario).replace('_', ' ')
    fig.suptitle(
        f'Post-hoc illustrative Test example | Seed {int(row.seed)} | {scenario}\n'
        'Predictions: sigmoid ≥ 0.50; green outlines in prediction panels = AP ground truth',
        fontsize=11
    )
    fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight', format='png')
    plt.close(fig)
    assert path.is_file() and path.stat().st_size > 15_000


# -- 6. Render all manifest rows, with exact count matching ------------------
# Use unique (seed, image_id) predictions once, then render every preserved
# scenario row from the original manifest. No post-hoc reselection happens here.
rows_sorted = cases.sort_values(['seed', 'Scenario', 'rank_by_image_id', 'image_id']).reset_index(drop=True)
results = []
base_hashes = {
    p.name: sha256_file(p)
    for p in [PLAN_PATH, TEST_AUDIT, PER_PATH, CASE_PATH, ERROR_METADATA, TEST_METADATA, TEST_SPLIT]
}

with tempfile.TemporaryDirectory(prefix='.qualitative_stage_', dir=MANUSCRIPT) as tmp:
    stage = Path(tmp)
    figure_path_by_case = {}
    n_unique = 0
    for seed in (42, 123, 2026):
        seed_cases = rows_sorted[rows_sorted['seed'].eq(seed)]
        if seed_cases.empty:
            continue
        print(f'\nRendering Seed {seed}: {len(seed_cases)} previously selected rows')
        models = {m: load_model(registry[(m, seed)]) for m in ('M6', 'C2')}
        predictions = {}
        for image_id in seed_cases['image_id'].drop_duplicates():
            image_id = str(image_id)
            rgb, tensor, gt, rcf_gt = load_image_and_mask(image_id)
            pred_masks = {}
            with torch.inference_mode():
                inputs = tensor.unsqueeze(0).to(DEVICE)
                for name, model in models.items():
                    with torch.amp.autocast('cuda'):
                        logits = model(inputs)
                    assert tuple(logits.shape) == (1, 3, 512, 512)
                    pred_masks[name] = (logits[0, 0].float() >= 0).cpu().numpy()
            for name in ('M6', 'C2'):
                tp, fp, fn, ng, n_pred = frozen_lesion_counts(gt, pred_masks[name])
                ref = published_index.loc[(seed, name, image_id)]
                expected = (int(ref.TP), int(ref.FP), int(ref.FN),
                            int(ref.GT_components), int(ref.Pred_components))
                assert (tp, fp, fn, ng, n_pred) == expected, (
                    f'Exact frozen lesion count mismatch: {seed}, {name}, {image_id}: '
                    f'{(tp, fp, fn, ng, n_pred)} vs {expected}. STOP.')
            predictions[image_id] = (rgb, gt, rcf_gt, pred_masks['M6'], pred_masks['C2'])
            n_unique += 1
        for _, row in seed_cases.iterrows():
            image_id = str(row['image_id'])
            figure_number = len(results) + 1
            filename = f'Qualitative_Case_{figure_number:02d}_Seed{seed}.png'
            figure = stage / filename
            rgb, gt, rcf_gt, m6, c2 = predictions[image_id]
            render_case(row, rgb, gt, rcf_gt, m6, c2, figure)
            r = dict(row)
            r['figure_filename'] = filename
            r['figure_sha256'] = sha256_file(figure)
            r['frozen_count_match_M6'] = True
            r['frozen_count_match_C2'] = True
            results.append(r)
        del models
        gc.collect()
        torch.cuda.empty_cache()
        print(f'  Seed {seed}: {len(seed_cases)} figures staged; image-level TP/FP/FN audit PASSED')

    assert len(results) == len(cases) == 36
    result_df = pd.DataFrame(results)
    assert not result_df.duplicated(['seed', 'Scenario', 'image_id']).any()

    legend = (
        'Figure type: post-hoc qualitative illustrations of already-published frozen Test evaluation.\n'
        'Left to right: original 512x512 padded radiograph; AP GT (green) and RCF GT (cyan); '
        'AP prediction from M6 correct-RCF training (orange); AP prediction from C2 shuffled-RCF '
        'control (magenta). Green outlines in the prediction panels indicate AP ground truth.\n'
        'Binary prediction rule: AP sigmoid probability >= 0.50. 8-connected components, '
        'Hungarian maximum-cardinality-then-IoU matching; IoU >= 0.10; '
        'minimum predicted component area 1 pixel. TP, FP, FN are per-image component counts.\n'
        'Only cases already in the frozen exploratory illustrative-case manifest are rendered. '
        'Selection occurred after Test evaluation and is illustrative, NOT a registered '
        'endpoint or representative estimate. Rendering re-runs inference solely to produce '
        'pixel overlays and verifies per-image lesion counts against the published results.\n'
        'No model selection, probability threshold tuning, checkpoint changes or statistical '
        'inference is performed here. Patient independence is not verified.\n'
        'Radiographs must be reviewed for patient identifiers and clinically assessed before '
        'public dissemination.\n'
    )
    (stage / OUT_LEGEND.name).write_text(legend, encoding='utf-8')
    result_df.to_csv(stage / OUT_MANIFEST.name, index=False)
    readback = pd.read_csv(stage / OUT_MANIFEST.name, dtype={'image_id': str})
    assert len(readback) == 36 and readback['frozen_count_match_M6'].all()
    assert readback['frozen_count_match_C2'].all()
    assert list(readback['figure_filename']) == list(result_df['figure_filename'])

    metadata = {
        'title': 'Post-hoc qualitative AP illustrations, M6 vs C2 on held-out Test',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'pretest_plan_sha256': EXPECTED_PLAN_SHA,
        'frozen_primary_IoU': 0.10,
        'frozen_min_area_pixels': 1,
        'frozen_probability_threshold': 0.50,
        'n_selected_case_manifest_rows': len(cases),
        'n_rendered_figures': len(results),
        'n_unique_seed_image_pairs': n_unique,
        'sources_sha256': base_hashes,
        'checkpoints_sha256': {
            f'{seed}_{m}': registry[(m, seed)]['checkpoint_sha256']
            for seed in (42, 123, 2026) for m in ('M6', 'C2')
        },
        'post_hoc_descriptive_only': True,
        'selected_cases_not_reselected_or_retuned': True,
        'new_inference': 'ONLY selected image pixel masks for overlays',
        'all_rendered_per_image_lesion_counts_equal_published': True,
        'test_statistical_metrics_recomputed': False,
        'patient_disjointness_verified': False,
        'pretest_plan_modified': False,
        'original_test_csv_and_results_modified': False,
    }
    (stage / OUT_METADATA.name).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    with open(stage / OUT_METADATA.name, encoding='utf-8') as f:
        assert json.load(f) == metadata
    assert sha256_file(PLAN_PATH) == EXPECTED_PLAN_SHA
    for p in [PLAN_PATH, TEST_AUDIT, PER_PATH, CASE_PATH, ERROR_METADATA, TEST_METADATA, TEST_SPLIT]:
        assert sha256_file(p) == base_hashes[p.name], f'Input changed unexpectedly: {p}'

    # All expected files have been validated, now publish to a NEW folder.
    assert not FIG_DIR.exists(), 'Another run created this folder; do not overwrite.'
    FIG_DIR.mkdir(parents=False, exist_ok=False)
    for p in sorted(stage.iterdir()):
        dest = FIG_DIR / p.name
        assert not dest.exists()
        os.replace(p, dest)

assert OUT_MANIFEST.is_file() and OUT_METADATA.is_file() and OUT_LEGEND.is_file()
assert len(list(FIG_DIR.glob('Qualitative_Case_*_Seed*.png'))) == 36
assert sha256_file(PLAN_PATH) == EXPECTED_PLAN_SHA
print('\n' + '=' * 98)
print('QUALITATIVE RADIOGRAPH FIGURES COMPLETED')
print('=' * 98)
print('New 4-panel radiograph figures:', len(results))
print('Unique seed-image pairs subjected to illustrative inference:', n_unique)
print('All M6/C2 per-image TP, FP, FN reproduced: YES')
print('Manifest CSV:', OUT_MANIFEST.name)
print('Figure legend:', OUT_LEGEND.name)
print('Provenance metadata:', OUT_METADATA.name)
print('New output directory:', FIG_DIR)
print('Original Test results modified: NO')
print('Original model checkpoints modified: NO')
print('Frozen Pre-Test plan modified: NO')
print('Threshold or model retuning: NO')
print('New statistical test: NO')
print('New inference was for illustration only: YES')
print('Patient independence: NOT VERIFIED')
print('FINAL STATUS: PASSED')
