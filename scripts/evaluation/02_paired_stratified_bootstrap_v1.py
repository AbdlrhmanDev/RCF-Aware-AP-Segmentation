# ============================================================
# HELD-OUT TEST — FROZEN PAIRED STRATIFIED BOOTSTRAP V1
# Run as ONE Colab cell; CPU is sufficient.
# Reads only the previously generated Test CSVs/metadata.
# All comparisons and thresholds were frozen before Test.
# Does not train, infer, retune, read raw images, or overwrite.
# ============================================================
import os
import json
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    REAL_MYDRIVE
except NameError:
    from google.colab import drive
    drive.mount('/content/drive')
    REAL_MYDRIVE = Path('/content/drive/MyDrive')

ROOT = Path(REAL_MYDRIVE) / 'RCF_Aware_AP_Segmentation_Research'
ANALYSIS = ROOT / '06_Lesion_Level_Analysis'
OUT = ROOT / '07_Heldout_Test_Evaluation_v1'
PLAN_PATH = ANALYSIS / 'heldout_test_analysis_plan_frozen_v1.json'
AUDIT_PATH = ANALYSIS / 'heldout_test_input_integrity_audit_v1.json'
INPUT_PER_IMAGE = OUT / 'heldout_test_12models_lesion_per_image_v1.csv'
INPUT_LESION = OUT / 'heldout_test_12models_lesion_summary_v1.csv'
INPUT_METADATA = OUT / 'heldout_test_12models_metadata_v1.json'
RESULT_CSV = OUT / 'heldout_test_paired_stratified_bootstrap_v1.csv'
RESULT_JSON = OUT / 'heldout_test_paired_stratified_bootstrap_metadata_v1.json'
FROZEN_SHA = '35dbf447bbabb1cec4bacebe5ade8c90d9d9fecf4a6e16f40db4ac1add635800'
SEEDS = [42, 123, 2026]
MODELS = ['M5', 'M6', 'C1', 'C2']
SETTINGS = [(0.10, 1), (0.50, 1), (0.10, 16), (0.50, 16)]
STRATA = {'AP_RCF': 43, 'AP_ONLY': 56, 'RCF_ONLY': 245, 'NEITHER': 399}
COMPARISONS = [('M6', 'C2'), ('M6', 'M5'), ('M6', 'C1')]
B = 5000
RNG_SEED = 20261019
EXPECTED_COUNTS = {
    42: {'M5': (97,301,19), 'M6': (90,205,26), 'C1': (91,231,25), 'C2': (93,275,23)},
    123: {'M5': (88,169,28), 'M6': (86,156,30), 'C1': (91,310,25), 'C2': (84,139,32)},
    2026: {'M5': (93,315,23), 'M6': (90,171,26), 'C1': (96,355,20), 'C2': (95,382,21)}
}

def sha256(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            d.update(c)
    return d.hexdigest()

for p in (PLAN_PATH, AUDIT_PATH, INPUT_PER_IMAGE, INPUT_LESION, INPUT_METADATA):
    assert p.is_file(), f'Missing frozen input: {p}'
for p in (RESULT_CSV, RESULT_JSON):
    assert not p.exists(), f'Bootstrap result already exists, DO NOT overwrite: {p}'
assert sha256(PLAN_PATH) == FROZEN_SHA, 'FROZEN PLAN CHANGED: STOP.'

with open(PLAN_PATH, encoding='utf-8') as f: plan = json.load(f)
with open(AUDIT_PATH, encoding='utf-8') as f: audit = json.load(f)
with open(INPUT_METADATA, encoding='utf-8') as f: source_meta = json.load(f)

assert plan['plan_status'] == 'FROZEN_BEFORE_HELD_OUT_TEST'
assert plan['primary_analysis']['comparison'] == 'M6_minus_C2'
assert plan['primary_analysis']['endpoint'] == 'Lesion_F1'
assert plan['training_seeds'] == SEEDS
assert set(plan['fixed_models']) == set(MODELS)
assert plan['lesion_evaluation']['probability_threshold'] == 0.50
assert plan['lesion_evaluation']['primary_IoU_threshold'] == 0.10
assert plan['lesion_evaluation']['primary_min_predicted_area_pixels'] == 1
assert int(plan['bootstrap']['resamples_per_seed']) == B
assert int(plan['bootstrap']['rng_seed']) == RNG_SEED
assert plan['bootstrap']['resampling_unit'] == 'test_image'
assert plan['bootstrap']['same_resampled_images_across_models'] is True
assert plan['bootstrap']['patient_level_bootstrap'] is False
assert plan['bootstrap']['pool_draws_across_training_seeds'] is False
assert audit['status'] == 'PASSED' and audit['frozen_plan_sha256'] == FROZEN_SHA
assert source_meta['status'] == 'PASSED'
assert source_meta['frozen_pretest_plan_sha256'] == FROZEN_SHA
assert source_meta['test_input_audit_sha256'] == sha256(AUDIT_PATH)
assert source_meta['test_split_sha256'] == audit['test_csv_sha256']
assert source_meta['ground_truth_AP_components'] == 116
assert source_meta['test_images'] == 743
assert source_meta['output_rows'] == {'per_image': 35664, 'pixel': 12, 'lesion': 48}
assert source_meta['image_bootstrap_performed'] is False
assert source_meta['test_based_model_selection'] is False
assert source_meta['test_based_threshold_tuning'] is False

print('='*88)
print('FROZEN HELD-OUT TEST PAIRED BOOTSTRAP — PRECHECK')
print('='*88)
print('Plan SHA256:', FROZEN_SHA)
print('Resamples per seed:', B)
print('Comparisons:', ', '.join(a+'-'+b for a,b in COMPARISONS))
print('No new Test inference or tuning.')

required_cols = {
    'model', 'seed', 'image_id', 'group', 'IoU_threshold', 'min_pred_area',
    'GT_components', 'Pred_components', 'TP', 'FP', 'FN',
    'checkpoint_sha256', 'validation_protocol_sha256', 'pretest_plan_sha256'
}
per_image = pd.read_csv(INPUT_PER_IMAGE, dtype={'image_id': str})
lesion = pd.read_csv(INPUT_LESION)
assert len(per_image) == 35664 and len(lesion) == 48
assert required_cols.issubset(per_image.columns)
assert not per_image[list(required_cols)].isna().any().any()
assert set(per_image['seed']) == set(SEEDS)
assert set(per_image['model']) == set(MODELS)
assert set(per_image['group']) == set(STRATA)
assert per_image['pretest_plan_sha256'].eq(FROZEN_SHA).all()
assert per_image['validation_protocol_sha256'].eq(plan['lesion_evaluation']['validation_protocol_sha256']).all()
assert not per_image.duplicated(['seed','model','image_id','IoU_threshold','min_pred_area']).any()
assert (per_image['TP']+per_image['FP']==per_image['Pred_components']).all()
assert (per_image['TP']+per_image['FN']==per_image['GT_components']).all()
assert (per_image[['TP','FP','FN','GT_components','Pred_components']] >= 0).all().all()
assert per_image.groupby('image_id')['group'].nunique().eq(1).all()
assert per_image.groupby(['image_id','IoU_threshold','min_pred_area'])['GT_components'].nunique().eq(1).all()
assert set((round(float(i),2),int(a)) for i,a in per_image[['IoU_threshold','min_pred_area']].drop_duplicates().itertuples(index=False,name=None)) == set(SETTINGS)

registry = {(r['model_id'],int(r['seed'])):r['checkpoint_sha256'] for r in plan['checkpoints']}
assert len(registry) == 12
for (model, seed), group_frame in per_image.groupby(['model','seed']):
    assert len(group_frame) == 743 * 4
    assert group_frame['checkpoint_sha256'].eq(registry[(model,seed)]).all(), f'Wrong checkpoint provenance: {model} {seed}'

primary_mask = np.isclose(per_image['IoU_threshold'],0.10) & (per_image['min_pred_area']==1)
canon = per_image[primary_mask & (per_image['model']=='M6') & (per_image['seed']==42)][['image_id','group','GT_components']].sort_values('image_id').reset_index(drop=True)
assert len(canon)==743 and canon['image_id'].is_unique
assert canon['group'].value_counts().to_dict()==STRATA
assert int(canon['GT_components'].sum())==116
assert set(canon.loc[canon['GT_components']==0,'group']).issubset(set(STRATA))
# The frozen ALL_NEGATIVE subgroup consists of RCF_ONLY and NEITHER.
assert canon.loc[canon['group'].isin(['RCF_ONLY','NEITHER']),'GT_components'].eq(0).all()
image_ids = canon['image_id'].tolist()
image_groups = canon['group'].to_numpy()
gt_components = canon['GT_components'].to_numpy(dtype=np.int64)

aligned = {}
for seed in SEEDS:
    for iou, area in SETTINGS:
        for model in MODELS:
            rows = per_image[(per_image['seed']==seed)&(per_image['model']==model)&np.isclose(per_image['IoU_threshold'],iou)&(per_image['min_pred_area']==area)]
            assert len(rows)==743 and rows['image_id'].is_unique
            a = rows.set_index('image_id').loc[image_ids]
            assert np.array_equal(a['group'].to_numpy(),image_groups)
            assert np.array_equal(a['GT_components'].to_numpy(dtype=np.int64),gt_components)
            assert int(a['GT_components'].sum())==116
            record=lesion[(lesion['seed']==seed)&(lesion['model']==model)&np.isclose(lesion['IoU_threshold'],iou)&(lesion['min_pred_area']==area)]
            assert len(record)==1
            r=record.iloc[0]
            assert all(int(a[col].sum())==int(r[col]) for col in ('TP','FP','FN'))
            for metric,col in [('Lesion_Recall','TP'),('FP_Components_per_Image','FP')]:
                calc = a['TP'].sum()/116 if metric=='Lesion_Recall' else a['FP'].sum()/743
                assert abs(calc-float(r[metric]))<1e-10
            t,f,n = (int(a[x].sum()) for x in ('TP','FP','FN'))
            assert abs((2*t/(2*t+f+n))-float(r['Lesion_F1']))<1e-10
            aligned[(seed,iou,area,model)] = {key: a[key].to_numpy(dtype=np.int64) for key in ('TP','FP','FN')}

for seed in SEEDS:
    for model in MODELS:
        a=aligned[(seed,0.10,1,model)]
        actual=tuple(int(a[c].sum()) for c in ('TP','FP','FN'))
        assert actual==EXPECTED_COUNTS[seed][model], (seed,model,actual)

print('Test data 743 x 12 checkpoints x 4 settings: VERIFIED')
print('GT components: 116; identical GT per image: VERIFIED')
print('Checkpoint hashes and published lesion summaries: VERIFIED')

# Metric helper: vectorized over bootstrap resamples, zero-safe.
def safe_ratio(num, den):
    num,den=np.asarray(num,dtype=float),np.asarray(den,dtype=float)
    return np.divide(num,den,out=np.zeros_like(num,dtype=float),where=den>0)

# One deterministic stratified sample per seed, SHARED by all models/settings.
# Same declared RNG seed is restarted for each seed (no pooling).
all_results=[]
metric_names = ['Lesion_F1','Lesion_Recall','Lesion_Precision',
                'FP_Components_per_Image','AP_Negative_FP_Image_Rate']

for seed in SEEDS:
    rng=np.random.default_rng(RNG_SEED)
    group_draws={}
    group_positions={}
    for group,expected_n in STRATA.items():
        positions=np.flatnonzero(image_groups==group)
        assert len(positions)==expected_n
        group_positions[group]=positions
        group_draws[group]=positions[rng.integers(0,expected_n,size=(B,expected_n))]

    def sampled_indices(included):
        return np.concatenate([group_draws[g] for g in included],axis=1)

    all_groups=list(STRATA)
    subgroups={
        'ALL': all_groups,
        'ALL_NEGATIVE':['RCF_ONLY','NEITHER'],
        'AP_RCF':['AP_RCF'],
        'AP_ONLY':['AP_ONLY'],
        'RCF_ONLY':['RCF_ONLY'],
        'NEITHER':['NEITHER']
    }

    for iou,area in SETTINGS:
        active_subgroups=['ALL'] if (iou,area)!=(0.10,1) else list(subgroups)
        for subgroup in active_subgroups:
            included=subgroups[subgroup]
            observed_positions=np.concatenate([group_positions[g] for g in included])
            draws=sampled_indices(included)
            n=len(observed_positions)
            assert draws.shape==(B,n)
            observed_gt=gt_components[observed_positions]
            boot_gt=gt_components[draws]
            n_neg=int((observed_gt==0).sum())
            boot_negative=boot_gt==0
            assert (boot_gt.sum(axis=1)>0).all() if observed_gt.sum()>0 else (boot_gt==0).all()
            assert (boot_negative.sum(axis=1)>0).all() if n_neg>0 else True

            chosen_metrics=metric_names if subgroup=='ALL' else (
                ['FP_Components_per_Image','AP_Negative_FP_Image_Rate'] if observed_gt.sum()==0
                else ['Lesion_F1','Lesion_Recall','Lesion_Precision','FP_Components_per_Image']
            )

            computed={}
            for model in MODELS:
                arrays=aligned[(seed,iou,area,model)]
                obs_tp,obs_fp,obs_fn=[int(arrays[k][observed_positions].sum()) for k in ('TP','FP','FN')]
                bs_tp,bs_fp,bs_fn=[arrays[k][draws].sum(axis=1) for k in ('TP','FP','FN')]
                fp_event=arrays['FP'][observed_positions]>0
                bs_fp_event=arrays['FP'][draws]>0
                obs={
                    'Lesion_F1': float(safe_ratio(2*obs_tp,2*obs_tp+obs_fp+obs_fn)),
                    'Lesion_Recall':float(safe_ratio(obs_tp,obs_tp+obs_fn)),
                    'Lesion_Precision':float(safe_ratio(obs_tp,obs_tp+obs_fp)),
                    'FP_Components_per_Image':obs_fp/n,
                    'AP_Negative_FP_Image_Rate':float(fp_event[observed_gt==0].mean()) if n_neg else np.nan
                }
                boot={
                    'Lesion_F1':safe_ratio(2*bs_tp,2*bs_tp+bs_fp+bs_fn),
                    'Lesion_Recall':safe_ratio(bs_tp,bs_tp+bs_fn),
                    'Lesion_Precision':safe_ratio(bs_tp,bs_tp+bs_fp),
                    'FP_Components_per_Image':bs_fp/n,
                    'AP_Negative_FP_Image_Rate':safe_ratio(np.logical_and(bs_fp_event,boot_negative).sum(axis=1),boot_negative.sum(axis=1))
                }
                computed[model]=(obs,boot)

            for candidate,baseline in COMPARISONS:
                obs_c,boot_c=computed[candidate]
                obs_b,boot_b=computed[baseline]
                for metric in chosen_metrics:
                    d=np.asarray(boot_c[metric]-boot_b[metric],dtype=float)
                    assert d.shape==(B,) and np.isfinite(d).all()
                    lo,hi=np.percentile(d,[2.5,97.5])
                    effect=float(obs_c[metric]-obs_b[metric])
                    all_results.append({
                        'seed':seed,'Comparison':f'{candidate}_minus_{baseline}',
                        'Group':subgroup,'Images':n,'IoU_threshold':iou,
                        'min_pred_area':area,'Metric':metric,
                        'Candidate':float(obs_c[metric]),'Baseline':float(obs_b[metric]),
                        'Difference':effect,'CI95_Low':float(lo),'CI95_High':float(hi),
                        'Bootstrap_Resamples':B
                    })
    print(f'Seed {seed}: PAIRED STRATIFIED BOOTSTRAP PASSED ({B} draws)',flush=True)

results=pd.DataFrame(all_results)
assert len(results)==306, f'Unexpected number of output rows: {len(results)}'
assert not results.duplicated(['seed','Comparison','Group','IoU_threshold','min_pred_area','Metric']).any()
assert np.isfinite(results[['Difference','CI95_Low','CI95_High','Candidate','Baseline']].to_numpy(dtype=float)).all()
assert (results['CI95_Low']<=results['CI95_High']).all()
assert results['Bootstrap_Resamples'].eq(B).all()

primary=results[(results['Group']=='ALL')&np.isclose(results['IoU_threshold'],0.10)&(results['min_pred_area']==1)]
assert len(primary)==45   # 3 seeds x 3 comparisons x 5 metrics
for seed in SEEDS:
    for candidate,baseline in COMPARISONS:
        for metric in ['Lesion_F1','Lesion_Recall','FP_Components_per_Image']:
            r=primary[(primary['seed']==seed)&(primary['Comparison']==f'{candidate}_minus_{baseline}')&(primary['Metric']==metric)]
            assert len(r)==1
            a=aligned[(seed,0.10,1,candidate)]
            b=aligned[(seed,0.10,1,baseline)]
            if metric=='Lesion_F1':
                def f1(x):
                    t,f,n=(int(x[c].sum()) for c in ('TP','FP','FN'))
                    return 2*t/(2*t+f+n)
                expected=f1(a)-f1(b)
            elif metric=='Lesion_Recall':
                expected=(a['TP'].sum()-b['TP'].sum())/116
            else:
                expected=(a['FP'].sum()-b['FP'].sum())/743
            assert abs(float(r['Difference'].iloc[0])-expected)<1e-12

print('\n'+'='*108)
print('FROZEN HELD-OUT TEST — PRIMARY PAIRED BOOTSTRAP (ALL SEEDS)')
print('='*108)
show=primary[primary['Metric'].isin(['Lesion_F1','Lesion_Recall','FP_Components_per_Image','AP_Negative_FP_Image_Rate'])]
print(show[['seed','Comparison','Metric','Difference','CI95_Low','CI95_High']].sort_values(['Comparison','seed','Metric']).round(4).to_string(index=False))
print('\n'+'='*108)
print('M6_minus_C2 — PRIMARY ALL_NEGATIVE SUPPORTIVE BOOTSTRAP')
print('='*108)
neg=results[(results['Comparison']=='M6_minus_C2')&(results['Group']=='ALL_NEGATIVE')&np.isclose(results['IoU_threshold'],0.10)&(results['min_pred_area']==1)]
assert len(neg)==6
print(neg[['seed','Group','Metric','Difference','CI95_Low','CI95_High']].sort_values(['seed','Metric']).round(4).to_string(index=False))

metadata={
    'analysis':'Frozen held-out Test paired stratified image bootstrap v1',
    'created_at_utc':datetime.now(timezone.utc).isoformat(),
    'frozen_pretest_plan_sha256':FROZEN_SHA,
    'test_input_audit_sha256':sha256(AUDIT_PATH),
    'test_inference_metadata_sha256':sha256(INPUT_METADATA),
    'test_lesion_per_image_sha256':sha256(INPUT_PER_IMAGE),
    'test_lesion_summary_sha256':sha256(INPUT_LESION),
    'seed_list':SEEDS,
    'comparisons':[f'{a}_minus_{b}' for a,b in COMPARISONS],
    'primary_comparison':'M6_minus_C2',
    'primary_endpoint':'Lesion_F1',
    'image_count':743,
    'ground_truth_components':116,
    'strata_counts':STRATA,
    'bootstrap_resamples_per_seed':B,
    'rng_seed_reset_for_each_seed':RNG_SEED,
    'resampling_unit':'test_image',
    'method':'paired image bootstrap stratified by frozen Test groups; percentile CI [2.5,97.5]',
    'confidence_level':0.95,
    'all_models_share_draws_within_seed':True,
    'bootstrap_draws_pooled_across_seeds':False,
    'captures_training_seed_uncertainty':False,
    'patient_level_bootstrap':False,
    'patient_independence_verified':False,
    'exploratory_subgroups':True,
    'frozen_settings':[{'IoU_threshold':i,'min_pred_area':a} for i,a in SETTINGS],
    'result_rows':len(results),
    'training_performed':False,
    'inference_performed':False,
    'thresholds_retuned':False,
    'frozen_plan_modified':False,
}

# Stage, reload, and publish two new files; preserve all original evidence.
with tempfile.TemporaryDirectory(dir=OUT,prefix='.frozen_test_bootstrap_stage_') as tmp:
    tmp=Path(tmp)
    temp_csv=tmp/RESULT_CSV.name
    temp_json=tmp/RESULT_JSON.name
    results.to_csv(temp_csv,index=False)
    with open(temp_json,'w',encoding='utf-8') as f:
        json.dump(metadata,f,indent=2,ensure_ascii=False,allow_nan=False)
    reread=pd.read_csv(temp_csv)
    with open(temp_json,encoding='utf-8') as f: reread_meta=json.load(f)
    assert len(reread)==306 and list(reread.columns)==list(results.columns)
    assert reread_meta==metadata
    assert sha256(PLAN_PATH)==FROZEN_SHA
    assert not RESULT_CSV.exists() and not RESULT_JSON.exists()
    os.replace(temp_csv,RESULT_CSV)
    os.replace(temp_json,RESULT_JSON)

assert RESULT_CSV.is_file() and RESULT_JSON.is_file()
assert len(pd.read_csv(RESULT_CSV))==306
assert sha256(PLAN_PATH)==FROZEN_SHA

print('\n'+'='*90)
print('HELD-OUT TEST PAIRED BOOTSTRAP COMPLETED')
print('='*90)
print('Seeds:',SEEDS)
print('Comparisons: M6_minus_C2, M6_minus_M5, M6_minus_C1')
print('Resamples per seed:',B)
print('Result rows:',len(results))
print('Saved:',RESULT_CSV.name)
print('Saved:',RESULT_JSON.name)
print('Inference performed: NO')
print('New model training: NO')
print('Threshold tuning: NO')
print('Original Test results overwritten: NO')
print('Frozen plan modified: NO')
print('Patient-level independence established: NO')
print('FINAL STATUS: PASSED')
