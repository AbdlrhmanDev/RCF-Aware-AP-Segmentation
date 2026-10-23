# ================================================================
# PRAD — METHODS TRAINING / ETHICS PROVENANCE AUDIT (READ-ONLY)
# One self-contained Google Colab cell; CPU sufficient.
# Reads the 12 EXACT configuration files registered in the frozen
# pre-Test analysis plan, verifies each config SHA256, and reviews
# original training_history.csv files without model inference.
# Original sources and Test data remain untouched.
# ================================================================

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

# ----- 1. DRIVE AND LOCK ---------------------------------------------------
DRIVE = Path('/content/drive/MyDrive')
if not DRIVE.is_dir():
    from google.colab import drive
    drive.mount('/content/drive')

ROOT = DRIVE / 'RCF_Aware_AP_Segmentation_Research'
PLAN = ROOT / '06_Lesion_Level_Analysis/heldout_test_analysis_plan_frozen_v1.json'
PREPROC = ROOT / '03_Preprocessing_QC/dataloader_config_v1.json'
SHUFFLE = ROOT / '04_Models/C02_Shuffled_RCF_Control/shuffle_protocol_v1.json'
EXPECTED_PLAN_SHA = '35dbf447bbabb1cec4bacebe5ade8c90d9d9fecf4a6e16f40db4ac1add635800'
EXPECTED_MODELS = ('M5', 'M6', 'C1', 'C2')
EXPECTED_SEEDS = (42, 123, 2026)
KEYS = {(m, s) for m in EXPECTED_MODELS for s in EXPECTED_SEEDS}

assert ROOT.is_dir(), 'Research project not found in mounted Google Drive.'
assert PLAN.is_file(), f'Missing frozen plan: {PLAN}'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()

def rel(path):
    return str(path.relative_to(ROOT))

def flattened(x, prefix=''):
    if isinstance(x, dict):
        for k, v in x.items():
            p = f'{prefix}.{k}' if prefix else str(k)
            yield from flattened(v, p)
    elif isinstance(x, list):
        # Preserve list-valued hyperparameters without exploding the output.
        yield prefix, x
    else:
        yield prefix, x

def safe_value(v):
    """Do not reproduce secrets, personal identifiers, or absolute data paths in logs."""
    s = json.dumps(v, ensure_ascii=False, default=str)
    if len(s) > 400:
        s = s[:397] + '...'
    return s

plan_sha = sha256(PLAN)
assert plan_sha == EXPECTED_PLAN_SHA, (
    f'FROZEN PLAN HASH MISMATCH: {plan_sha}; STOP. Do not edit the plan.'
)
with PLAN.open(encoding='utf-8') as f:
    plan = json.load(f)
assert plan.get('plan_status') == 'FROZEN_BEFORE_HELD_OUT_TEST'
assert plan.get('training_seeds') == list(EXPECTED_SEEDS)
assert set(plan.get('fixed_models', [])) == set(EXPECTED_MODELS)
registry = plan.get('checkpoints')
assert isinstance(registry, list) and len(registry) == 12

# ----- 2. EXACT REGISTERED TRAINING CONFIGS --------------------------------
configs, config_files, statuses, issues = {}, {}, {}, []
for record in registry:
    key = (record['model_id'], int(record['seed']))
    assert key in KEYS and key not in configs, f'Unexpected or duplicate entry: {key}'
    cfg_rel = record['config_relative_path']
    cfg_file = (ROOT / cfg_rel).resolve()
    assert cfg_file.is_relative_to(ROOT.resolve()), 'Config path outside research root.'
    assert cfg_file.is_file(), f'Missing registered config for {key}: {cfg_file}'
    assert sha256(cfg_file) == record['config_sha256'], (
        f'CONFIG HASH MISMATCH FOR {key}: {cfg_rel}. STOP.'
    )
    ckpt = (ROOT / record['checkpoint_relative_path']).resolve()
    assert ckpt.is_relative_to(ROOT.resolve()) and ckpt.is_file(), (
        f'Missing checkpoint registered for {key}.'
    )
    with cfg_file.open(encoding='utf-8') as f:
        config = json.load(f)
    assert isinstance(config, dict), f'Non-object training config for {key}'
    configs[key], config_files[key] = config, cfg_file
    statuses[key] = {'model': key[0], 'seed': key[1],
                     'config_relative_path': cfg_rel,
                     'config_sha256': record['config_sha256'],
                     'checkpoint_file_present': True}
assert set(configs) == KEYS, 'The 12 registered configs were not all verified.'

# ----- 3. EXTRACT TRAINING RELEVANT FIELDS ---------------------------------
# Field names are kept exactly as found, with original values.
CATEGORIES = {
    'ARCHITECTURE': r'architecture|model|encoder|backbone|stride|pretrain|weight[s]?|image_size|input|output|num_class|channels',
    'AUGMENTATION': r'augment|flip|rotat|bright|contrast|gamma|jitter|affine|transfor[m]?|crop|elastic|blur',
    'OPTIMIZATION_SCHEDULER': r'optim|adam|learn|(^|[.\-_])lr($|[.\-_])|scheduler|schedul|reduce|factor|milestone|cosine|warm|decay|patience|early_stop|epoch|batch|mixed.precision|amp',
    'LOSSES': r'loss|dice|bce|cross.entropy|focal|pos.weight|smoothing|lambda|coefficient|auxiliary',
    'SAMPLING': r'sampl|oversampl|balanc|positive.frac|positive_ratio|class.frac|train_fraction|sampling_seed',
    'DATA_SPLITS_AND_SELECT': r'split|validation|selection|select|checkpoint|best_epoch|best_metric|monitor',
    'CONTROL_AND_SHUFFLE': r'shuffl|derang|manifest|source_label|source.mask|permut|rcf.*group|group.*rcf',
}
SENSITIVE_FIELD = re.compile(r'password|secret|credential|(^|[._-])token($|[._-])|api_key|auth|patient_id|person|email', re.I)
PATH_FIELD = re.compile(r'(?:^|[._-])(path|root|dir|folder|drive|directory)(?:$|[._-])', re.I)

extracted = {}
for key, cfg in configs.items():
    flat = dict(flattened(cfg))
    extracted[key] = {}
    for category, rx in CATEGORIES.items():
        extracted[key][category] = {
            k: v for k, v in flat.items()
            if re.search(rx, k, re.I)
            and not SENSITIVE_FIELD.search(k)
            and not PATH_FIELD.search(k)
        }

# ----- 4. HISTORY: LR TRAJECTORY + BEST EPOCH ------------------------------
history_summary = {}
for key, cfg_file in config_files.items():
    # Histories are expected beside the registered config.
    candidate = cfg_file.parent / 'training_history.csv'
    if not candidate.is_file():
        history_summary[key] = {'present': False, 'note': 'Training history not adjacent to config'}
        continue
    with candidate.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    def first_col(targets):
        lookup = {c.lower().replace(' ', '_'): c for c in cols}
        for t in targets:
            if t in lookup:
                return lookup[t]
        return None
    epoch_c = first_col(['epoch', 'epochs'])
    lr_c = first_col(['learning_rate', 'lr', 'current_lr', 'train_learning_rate'])
    ap_c = first_col(['val_ap_loss', 'val_loss', 'validation_ap_loss', 'validation_loss'])
    epochs = []
    for n, row in enumerate(rows):
        ep = row.get(epoch_c) if epoch_c else str(n + 1)
        ep = int(float(ep)) if ep not in (None, '') else n + 1
        item = {'epoch': ep}
        if lr_c and row.get(lr_c) not in (None, ''):
            try: item['lr'] = float(row[lr_c])
            except ValueError: pass
        if ap_c and row.get(ap_c) not in (None, ''):
            try: item['validation_ap_loss'] = float(row[ap_c])
            except ValueError: pass
        epochs.append(item)
    changes = []
    prev = None
    for row in epochs:
        if 'lr' in row and (prev is None or row['lr'] != prev):
            changes.append({'epoch': row['epoch'], 'lr': row['lr']})
            prev = row['lr']
    best = min((x for x in epochs if 'validation_ap_loss' in x),
               key=lambda a: a['validation_ap_loss'], default=None)
    history_summary[key] = {
        'present': True, 'relative_path': rel(candidate),
        'sha256': sha256(candidate), 'rows': len(rows), 'columns': cols,
        'lr_changes_observed': changes, 'best_val_loss_row': best,
    }

# ----- 5. PREPROCESSING + SHUFFLING SOURCE METADATA -------------------------
additional_sources = {}
for title, source in [('dataloader_config', PREPROC), ('C2_shuffle_protocol', SHUFFLE)]:
    if source.is_file():
        with source.open(encoding='utf-8') as f: obj = json.load(f)
        additional_sources[title] = {'present': True, 'path': rel(source),
                                     'sha256': sha256(source), 'content': obj}
    else:
        additional_sources[title] = {'present': False, 'path': rel(source)}

# Research-root metadata ONLY: never read actual patient images or masks.
ethics_hints = []
for top in [ROOT, ROOT / '00_Research_Protocol', ROOT / '01_Dataset', ROOT / '03_Preprocessing_QC']:
    if top.is_dir():
        for p in top.iterdir():
            if p.is_file() and re.search(r'ethic|IRB|approval|consent|license|licence|permission|data.use', p.name, re.I):
                ethics_hints.append(rel(p))
ethics_hints = sorted(set(ethics_hints))

# ----- 6. REPORT TO COPY BACK INTO CHAT ------------------------------------
lines = []
def out(s=''):
    print(s)
    lines.append(str(s))

def section(title):
    out('\n' + '=' * 95)
    out(title)
    out('=' * 95)

section('METHODS PROVENANCE AUDIT — FROZEN ORIGINAL CONFIGS VERIFIED')
out('Frozen plan SHA256: ' + plan_sha)
out('Original training configs: 12/12 EXACT SHA256 MATCHES')
out('Frozen checkpoint files: 12/12 PRESENT (checkpoint hashes not recomputed here)')
out('No images, masks, or Test CSV read. No inference or model training.')

section('1. ORIGINAL CONFIG FILE REGISTRY — ALL 12 RUNS')
for m in EXPECTED_MODELS:
    for s in EXPECTED_SEEDS:
        z = statuses[(m, s)]
        out(f'{m:<3} Seed {s:<4} | SHA {z["config_sha256"][:16]} | {z["config_relative_path"]}')

for cat, title in [('AUGMENTATION', '2. EXACT TRAIN AUGMENTATION SETTINGS'),
                   ('OPTIMIZATION_SCHEDULER', '3. OPTIMIZER / LEARNING-RATE SCHEDULER / EARLY STOPPING'),
                   ('LOSSES', '4. EXACT TASK-LOSS COMPOSITION AND WEIGHTS'),
                   ('SAMPLING', '5. CLASS-BALANCED / AP-POSITIVE SAMPLING'),
                   ('ARCHITECTURE', '6. MODEL ARCHITECTURE AND WEIGHT SETTINGS'),
                   ('CONTROL_AND_SHUFFLE', '7. C1/C2 CONTROL AND SHUFFLING SETTINGS'),
                   ('DATA_SPLITS_AND_SELECT', '8. MODEL SELECTION / VALIDATION SETTINGS')]:
    section(title + ' — VALUES COPIED FROM ORIGINAL CONFIGS')
    for m in EXPECTED_MODELS:
        out(f'\n[{m}]')
        for s in EXPECTED_SEEDS:
            fields = extracted[(m, s)][cat]
            out(f'  Seed {s}: {len(fields)} relevant fields')
            if not fields:
                out('    NOT DOCUMENTED in the registered training_config.json')
            for k, v in sorted(fields.items()):
                out(f'    {k}: {safe_value(v)}')

section('9. ACTUAL TRAINING HISTORY / OBSERVED LEARNING RATE')
for m in EXPECTED_MODELS:
    for s in EXPECTED_SEEDS:
        h = history_summary[(m, s)]
        out(f'{m} Seed {s}: ' + ('history AVAILABLE' if h['present'] else 'MISSING training_history.csv'))
        if h['present']:
            out(f'  Epoch rows: {h["rows"]}; observed LR changes: {safe_value(h["lr_changes_observed"])}')
            out(f'  Lowest recorded val AP loss row: {safe_value(h["best_val_loss_row"])}')
            out(f'  Raw history columns: {safe_value(h["columns"])}')
        else:
            issues.append(f'{m} Seed {s} original training history not located beside config')

section('10. PREPROCESSING AND SHUFFLE MANIFEST PROTOCOL FILES')
for title, obj in additional_sources.items():
    out(f'\n{title}: ' + ('FOUND' if obj['present'] else 'NOT FOUND'))
    out('  ' + obj['path'])
    if obj['present']:
        out('  SHA256: ' + obj['sha256'])
        for k, v in sorted(flattened(obj['content'])):
            if not SENSITIVE_FIELD.search(k) and not PATH_FIELD.search(k):
                out(f'  {k}: {safe_value(v)}')

section('11. HUMAN-VERIFIED ETHICS / PATIENT CONSENT / PRAD LICENSE')
out('Institutional ethics approval or exemption: NOT VERIFIED BY THIS CODE.')
out('Original PRAD permission, license, and figure-reuse conditions: REQUIRE HUMAN DOCUMENT REVIEW.')
out('Patient-disjoint train/validation/test splits: NOT VERIFIED (patient IDs unavailable).')
out('Potential ethics/license-related file names within selected metadata folders:')
for hint in ethics_hints:
    out('  ' + hint)
if not ethics_hints:
    out('  None identified; absence does not imply exemption or unrestricted reuse.')
out('DO NOT invent IRB identifier, informed-consent statement, or dataset redistribution rights.')

section('12. AUDIT STATUS / WHAT YOU SHOULD SEND TO THE MANUSCRIPT WRITER')
missing_categories = []
for cat in ('AUGMENTATION', 'OPTIMIZATION_SCHEDULER', 'LOSSES'):
    if not any(extracted[key][cat] for key in KEYS):
        missing_categories.append(cat)
if missing_categories:
    out('NOT DOCUMENTED IN REGISTERED CONFIGS: ' + ', '.join(missing_categories))
    out('If absent, locate the ORIGINAL Colab training cell/script; do not reconstruct methods from guesses.')
else:
    out('Augmentation, optimization, and loss-related settings were found in registered configs.')
out('Observed LR values cannot alone identify the exact scheduler type or trigger logic.')
out('Published Test data and original checkpoints: NOT MODIFIED.')
out('Privacy-sensitive metadata not exported. No test inference.')
out('FINAL STATUS: ORIGINAL CONFIG AND HISTORY AUDIT COMPLETED; ETHICS CLEARANCE IS SEPARATE')

# ----- 7. SAVE AUDIT, NEVER OVERWRITE AN EXISTING REPORT -------------------
OUT_DIR = ROOT / '10_Manuscript_Methods_Provenance_Audit_v1'
REPORT_TXT = OUT_DIR / 'methods_training_protocol_audit_v1.txt'
REPORT_JSON = OUT_DIR / 'methods_training_protocol_audit_v1.json'
report = {
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'frozen_plan_sha256': plan_sha,
    'source': 'exact config paths and SHA256 registered in pre-test analysis plan',
    'registered_configs': [statuses[(m, s)] for m in EXPECTED_MODELS for s in EXPECTED_SEEDS],
    'relevant_training_config_fields': {
        f'{m}_Seed{s}': extracted[(m, s)] for m in EXPECTED_MODELS for s in EXPECTED_SEEDS
    },
    'training_histories': {
        f'{m}_Seed{s}': history_summary[(m, s)] for m in EXPECTED_MODELS for s in EXPECTED_SEEDS
    },
    'supplementary_protocol_files': additional_sources,
    'ethics_document_names': ethics_hints,
    'ethics_verified': False,
    'patient_disjointness_verified': False,
    'test_inference': False,
    'source_files_modified': False,
}
if OUT_DIR.exists():
    out(f'\nOUTPUT FOLDER ALREADY EXISTS: {OUT_DIR}')
    out('EXISTING FILES WERE NOT REPLACED. If rerunning, use the printed report.')
else:
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with REPORT_JSON.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    assert REPORT_TXT.is_file() and REPORT_JSON.is_file()
    out('\nAUDIT SAVED (NEW FILES ONLY):')
    out('  ' + str(REPORT_TXT))
    out('  ' + str(REPORT_JSON))

out('\nSEND ME: Sections 2, 3, 4, 5, 9, 10, 11, and FINAL STATUS (or the TXT file).')
