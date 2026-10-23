# ============================================================
# HELD-OUT TEST — ERROR PROFILE & ILLUSTRATIVE CASE MANIFEST
# One complete Google Colab cell. CPU only.
# Exploratory descriptive analysis AFTER frozen Test synthesis.
# NO NEW INFERENCE / TRAINING / MODEL SELECTION / RETUNING.
# ============================================================
import os, json, hashlib, tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

try:
    MYDRIVE = Path(REAL_MYDRIVE)
except NameError:
    from google.colab import drive
    drive.mount('/content/drive')
    MYDRIVE = Path('/content/drive/MyDrive')

ROOT = MYDRIVE / 'RCF_Aware_AP_Segmentation_Research'
VAL = ROOT / '06_Lesion_Level_Analysis'
TEST = ROOT / '07_Heldout_Test_Evaluation_v1'
PLAN_PATH = VAL / 'heldout_test_analysis_plan_frozen_v1.json'
PER_PATH = TEST / 'heldout_test_12models_lesion_per_image_v1.csv'
LESION_PATH = TEST / 'heldout_test_12models_lesion_summary_v1.csv'
SYNTH_META_PATH = TEST / 'heldout_test_three_seed_analysis_metadata_v1.json'
TEST_META_PATH = TEST / 'heldout_test_12models_metadata_v1.json'
EXPECTED_PLAN_SHA = '35dbf447bbabb1cec4bacebe5ade8c90d9d9fecf4a6e16f40db4ac1add635800'
SEEDS = (42, 123, 2026)
MODELS = ('M5','M6','C1','C2')
STRATA = {'AP_RCF':43,'AP_ONLY':56,'RCF_ONLY':245,'NEITHER':399}
GROUPS = ('ALL','AP_RCF','AP_ONLY','RCF_ONLY','NEITHER','ALL_NEGATIVE','ALL_POSITIVE')

PREFIX = 'heldout_test_error_analysis_exploratory_v1'
OUT = {
    'subgroups': TEST / (PREFIX + '_model_subgroups.csv'),
    'paired_images': TEST / (PREFIX + '_M6_vs_C2_per_image.csv'),
    'paired_subgroups': TEST / (PREFIX + '_M6_vs_C2_subgroups.csv'),
    'illustrations': TEST / (PREFIX + '_illustrative_cases.csv'),
    'metadata': TEST / (PREFIX + '_metadata.json'),
}

def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

for p in (PLAN_PATH, PER_PATH, LESION_PATH, SYNTH_META_PATH, TEST_META_PATH):
    assert p.is_file(), f'MISSING INPUT: {p}'
for p in OUT.values():
    assert not p.exists(), f'OUTPUT ALREADY EXISTS (DO NOT OVERWRITE): {p}'
assert sha(PLAN_PATH) == EXPECTED_PLAN_SHA, 'FROZEN PLAN HASH MISMATCH'
with open(PLAN_PATH, encoding='utf-8') as f: plan = json.load(f)
with open(SYNTH_META_PATH, encoding='utf-8') as f: synth_meta = json.load(f)
with open(TEST_META_PATH, encoding='utf-8') as f: test_meta = json.load(f)
assert plan['primary_analysis']['comparison']=='M6_minus_C2'
assert plan['primary_analysis']['endpoint']=='Lesion_F1'
assert tuple(plan['training_seeds'])==SEEDS
assert set(plan['fixed_models'])==set(MODELS)
assert plan['lesion_evaluation']['primary_IoU_threshold']==0.10
assert plan['lesion_evaluation']['primary_min_predicted_area_pixels']==1
assert synth_meta['frozen_pretest_plan_sha256']==EXPECTED_PLAN_SHA
assert synth_meta['input_hashes'][PER_PATH.name]==sha(PER_PATH)
assert synth_meta['input_hashes'][LESION_PATH.name]==sha(LESION_PATH)
assert test_meta['frozen_pretest_plan_sha256']==EXPECTED_PLAN_SHA

per = pd.read_csv(PER_PATH, dtype={'image_id':str})
lesion = pd.read_csv(LESION_PATH)
assert len(per)==35664 and len(lesion)==48
assert not per.duplicated(['seed','model','image_id','IoU_threshold','min_pred_area']).any()
assert (per['TP']+per['FP']==per['Pred_components']).all()
assert (per['TP']+per['FN']==per['GT_components']).all()
assert per['pretest_plan_sha256'].eq(EXPECTED_PLAN_SHA).all()
assert per['validation_protocol_sha256'].eq(plan['lesion_evaluation']['validation_protocol_sha256']).all()
assert set(per['model'])==set(MODELS) and set(per['seed'])==set(SEEDS)

primary = per[
    np.isclose(per['IoU_threshold'], 0.10) & (per['min_pred_area']==1)
].copy()
assert len(primary)==12*743
assert not primary.duplicated(['seed','model','image_id']).any()
assert primary.groupby(['seed','model']).size().eq(743).all()
assert primary.groupby('image_id')['group'].nunique().eq(1).all()
assert primary.groupby('image_id')['GT_components'].nunique().eq(1).all()
assert primary['GT_components'].ge(0).all()

canonical = primary[(primary['seed']==42)&(primary['model']=='M6')].copy()
assert len(canonical)==743 and canonical['image_id'].is_unique
assert canonical['group'].value_counts().to_dict()==STRATA
assert int(canonical['GT_components'].sum())==116
assert int(canonical['GT_components'].gt(0).sum())==99
assert int(canonical['GT_components'].eq(0).sum())==644
assert (canonical.loc[canonical['group'].isin(['AP_RCF','AP_ONLY']), 'GT_components']>0).all()
assert (canonical.loc[canonical['group'].isin(['RCF_ONLY','NEITHER']), 'GT_components']==0).all()

# Independently reconcile every primary model/seed aggregate with published summary.
for (seed, model), f in primary.groupby(['seed','model'],sort=False):
    ref = lesion[(lesion['seed']==seed)&(lesion['model']==model)&
        np.isclose(lesion['IoU_threshold'],0.10)&(lesion['min_pred_area']==1)]
    assert len(f)==743 and len(ref)==1
    for col in ('TP','FP','FN'):
        assert int(f[col].sum())==int(ref[col].iloc[0]), (seed,model,col)
    assert int(f['GT_components'].sum())==116

print('='*90)
print('FROZEN TEST ERROR ANALYSIS — INTEGRITY CHECK PASSED')
print('='*90)
print('Frozen plan SHA:',EXPECTED_PLAN_SHA)
print('Primary: IoU 0.10 / min area 1 / 8-connected components')
print('Images: 743 | Positive: 99 | Negative: 644 | GT components: 116')
print('Checkpoints: 12 | Test thresholds and model selection: UNCHANGED')

# ============================================================
# 1. EXPLORATORY PER-MODEL SUBGROUP ERROR PROFILES
# ============================================================
def choose_group(frame, group):
    if group=='ALL': return frame
    if group=='ALL_NEGATIVE': return frame[frame['GT_components']==0]
    if group=='ALL_POSITIVE': return frame[frame['GT_components']>0]
    return frame[frame['group']==group]

sub_rows = []
for seed in SEEDS:
    for model in MODELS:
        f = primary[(primary['seed']==seed)&(primary['model']==model)]
        for group in GROUPS:
            s=choose_group(f,group)
            n=len(s)
            assert n>0
            pos=s[s['GT_components']>0]
            neg=s[s['GT_components']==0]
            tp,fp,fn=(int(s[c].sum()) for c in ('TP','FP','FN'))
            gt=int(s['GT_components'].sum())
            assert tp+fn==gt
            sub_rows.append({
                'seed':seed,'model':model,'Group':group,'Images':n,
                'AP_Positive_Images':len(pos),'AP_Negative_Images':len(neg),
                'GT_components':gt,'TP':tp,'FP':fp,'FN':fn,
                'Lesion_Recall':(tp/gt if gt else np.nan),
                'Lesion_F1':(2*tp/(2*tp+fp+fn) if gt else np.nan),
                'FP_Components_per_Image':fp/n,
                'Negative_FP_Image_Rate':(float(neg['FP'].gt(0).mean()) if len(neg) else np.nan),
                'Positive_Any_FN_Image_Rate':(float(pos['FN'].gt(0).mean()) if len(pos) else np.nan),
                'Positive_All_Lesions_Missed_Image_Rate':(float(pos['TP'].eq(0).mean()) if len(pos) else np.nan),
                'Positive_Any_FP_Image_Rate':(float(pos['FP'].gt(0).mean()) if len(pos) else np.nan),
            })
sub = pd.DataFrame(sub_rows)
assert len(sub)==3*4*7
assert sub[sub['Group']=='ALL']['GT_components'].eq(116).all()
assert sub[sub['Group']=='ALL_NEGATIVE']['Images'].eq(644).all()
assert sub[sub['Group']=='ALL_POSITIVE']['Images'].eq(99).all()

# ============================================================
# 2. EXACT PAIRED IMAGE ERRORS: M6 MINUS C2
# ============================================================
paired_frames=[]
for seed in SEEDS:
    s=primary[primary['seed']==seed]
    a=s[s['model']=='M6'].set_index('image_id').sort_index()
    b=s[s['model']=='C2'].set_index('image_id').sort_index()
    assert a.index.equals(b.index) and len(a)==743
    assert np.array_equal(a['GT_components'].to_numpy(),b['GT_components'].to_numpy())
    assert np.array_equal(a['group'].to_numpy(),b['group'].to_numpy())
    paired=pd.DataFrame({
        'seed':seed, 'image_id':a.index.to_numpy(), 'group':a['group'].to_numpy(),
        'GT_components':a['GT_components'].to_numpy(),
        'M6_TP':a['TP'].to_numpy(), 'C2_TP':b['TP'].to_numpy(),
        'M6_FP':a['FP'].to_numpy(), 'C2_FP':b['FP'].to_numpy(),
        'M6_FN':a['FN'].to_numpy(), 'C2_FN':b['FN'].to_numpy(),
    })
    paired['M6_minus_C2_TP']=paired['M6_TP']-paired['C2_TP']
    paired['M6_minus_C2_FP']=paired['M6_FP']-paired['C2_FP']
    paired['M6_minus_C2_FN']=paired['M6_FN']-paired['C2_FN']
    paired['FP_advantage']=np.select(
        [paired['M6_minus_C2_FP']<0, paired['M6_minus_C2_FP']>0],
        ['M6_fewer_FP','C2_fewer_FP'],default='tie')
    paired['TP_advantage']=np.select(
        [paired['M6_minus_C2_TP']>0, paired['M6_minus_C2_TP']<0],
        ['M6_more_TP','C2_more_TP'],default='tie')
    assert (paired['M6_minus_C2_TP']+paired['M6_minus_C2_FN']==0).all()
    paired_frames.append(paired)
paired_all=pd.concat(paired_frames,ignore_index=True)
assert len(paired_all)==3*743
assert not paired_all.duplicated(['seed','image_id']).any()

# ============================================================
# 3. GROUP-SPECIFIC PAIRED ERROR COUNTS (DESCRIPTIVE)
# ============================================================
paired_summary_rows=[]
for seed in SEEDS:
    base=paired_all[paired_all['seed']==seed]
    for group in GROUPS:
        f=choose_group(base,group)
        n=len(f)
        assert n>0
        paired_summary_rows.append({
            'seed':seed,'Group':group,'Images':n,
            'M6_total_TP':int(f['M6_TP'].sum()),'C2_total_TP':int(f['C2_TP'].sum()),
            'M6_total_FP':int(f['M6_FP'].sum()),'C2_total_FP':int(f['C2_FP'].sum()),
            'M6_total_FN':int(f['M6_FN'].sum()),'C2_total_FN':int(f['C2_FN'].sum()),
            'Delta_TP_M6_minus_C2':int(f['M6_minus_C2_TP'].sum()),
            'Delta_FP_M6_minus_C2':int(f['M6_minus_C2_FP'].sum()),
            'Delta_FP_per_Image_M6_minus_C2':float(f['M6_minus_C2_FP'].mean()),
            'Images_M6_Fewer_FP':int((f['FP_advantage']=='M6_fewer_FP').sum()),
            'Images_C2_Fewer_FP':int((f['FP_advantage']=='C2_fewer_FP').sum()),
            'Images_FP_Tie':int((f['FP_advantage']=='tie').sum()),
            'Images_M6_More_TP':int((f['TP_advantage']=='M6_more_TP').sum()),
            'Images_C2_More_TP':int((f['TP_advantage']=='C2_more_TP').sum()),
            'Images_TP_Tie':int((f['TP_advantage']=='tie').sum()),
        })
paired_summary=pd.DataFrame(paired_summary_rows)
assert len(paired_summary)==3*7
assert (paired_summary[['Images_M6_Fewer_FP','Images_C2_Fewer_FP','Images_FP_Tie']].sum(axis=1)==paired_summary['Images']).all()
assert (paired_summary[['Images_M6_More_TP','Images_C2_More_TP','Images_TP_Tie']].sum(axis=1)==paired_summary['Images']).all()

# ============================================================
# 4. TRANSPARENT, POST-HOC ILLUSTRATION CANDIDATES
# Sort by image_id, take at most 2 per scenario/seed.
# These are NOT newly registered endpoints or evidence of effect.
# ============================================================
SCENARIOS = [
    ('RCF_ONLY_M6_avoids_FP', lambda x:(x['group']=='RCF_ONLY')&(x['M6_FP']==0)&(x['C2_FP']>0)),
    ('NEITHER_M6_avoids_FP', lambda x:(x['group']=='NEITHER')&(x['M6_FP']==0)&(x['C2_FP']>0)),
    ('RCF_ONLY_C2_avoids_FP', lambda x:(x['group']=='RCF_ONLY')&(x['C2_FP']==0)&(x['M6_FP']>0)),
    ('AP_RCF_M6_more_TP', lambda x:(x['group']=='AP_RCF')&(x['M6_TP']>x['C2_TP'])),
    ('AP_ONLY_M6_more_TP', lambda x:(x['group']=='AP_ONLY')&(x['M6_TP']>x['C2_TP'])),
    ('AP_RCF_C2_more_TP', lambda x:(x['group']=='AP_RCF')&(x['C2_TP']>x['M6_TP'])),
    ('AP_positive_both_miss', lambda x:(x['GT_components']>0)&(x['M6_FN']>0)&(x['C2_FN']>0)),
]
examples=[]
for seed in SEEDS:
    f=paired_all[paired_all['seed']==seed].sort_values('image_id')
    for scenario, selector in SCENARIOS:
        available=f[selector(f)]
        for rank, (_, row) in enumerate(available.head(2).iterrows(), start=1):
            examples.append({
                'seed':seed,'Scenario':scenario,'rank_by_image_id':rank,
                'image_id':str(row['image_id']),'group':str(row['group']),
                'GT_components':int(row['GT_components']),
                'M6_TP':int(row['M6_TP']),'M6_FP':int(row['M6_FP']),'M6_FN':int(row['M6_FN']),
                'C2_TP':int(row['C2_TP']),'C2_FP':int(row['C2_FP']),'C2_FN':int(row['C2_FN']),
                'selection_status':'post_hoc_descriptive_illustration_only',
            })
example_columns=['seed','Scenario','rank_by_image_id','image_id','group','GT_components',
                 'M6_TP','M6_FP','M6_FN','C2_TP','C2_FP','C2_FN','selection_status']
illustrations=pd.DataFrame(examples,columns=example_columns)
assert len(illustrations)<=3*len(SCENARIOS)*2
assert not illustrations.duplicated(['seed','Scenario','image_id']).any()

print('\n'+'='*105)
print('EXPLORATORY ERROR PROFILE — M6 AND C2 BY SEED AND GROUP')
print('='*105)
show=sub[(sub['model'].isin(['M6','C2'])) &
    (sub['Group'].isin(['AP_RCF','AP_ONLY','RCF_ONLY','NEITHER']))]
print(show[['seed','model','Group','Images','TP','FP','FN','FP_Components_per_Image',
    'Negative_FP_Image_Rate','Positive_All_Lesions_Missed_Image_Rate']].round(4).to_string(index=False))

print('\n'+'='*105)
print('M6 MINUS C2 — PAIRED ERROR PATTERNS (DESCRIPTIVE)')
print('='*105)
print(paired_summary[['seed','Group','Images','Delta_TP_M6_minus_C2',
    'Delta_FP_M6_minus_C2','Images_M6_Fewer_FP','Images_C2_Fewer_FP','Images_FP_Tie']].round(4).to_string(index=False))
print('\nILLUSTRATIVE CASE MANIFEST (POST-HOC):',len(illustrations),'scenario-seed rows')
print('No images opened and no model predictions generated.')

metadata={
    'analysis_id':'HeldOut_Test_Error_Profile_Exploratory_v1',
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'status':'EXPLORATORY_POST_TEST_DESCRIPTIVE_ONLY',
    'frozen_plan_sha256':EXPECTED_PLAN_SHA,
    'source_sha256':{p.name:sha(p) for p in (PER_PATH, LESION_PATH, SYNTH_META_PATH, TEST_META_PATH)},
    'models':list(MODELS),'seeds':list(SEEDS),'n_test_images':743,'GT_AP_components':116,
    'AP_positive_images':99,'AP_negative_images':644,
    'primary_iou':0.10,'primary_min_pred_area':1,
    'error_subgroups':list(GROUPS),
    'case_selection_rule':'For each seed and scenario: lexicographically earliest 2 image IDs satisfying descriptive condition; skip if none',
    'case_selection_is_post_hoc':True,
    'illustrations_are_inferential_evidence':False,
    'confirmatory_endpoints_changed':False,'models_or_thresholds_retuned':False,
    'test_image_contents_opened':False,'new_model_inference':False,
    'new_bootstrap':False,'training':False,
    'patient_independence_verified':False,
    'rows':{'subgroups':len(sub),'paired_images':len(paired_all),
            'paired_subgroups':len(paired_summary),'illustrations':len(illustrations)}
}
frames={'subgroups':sub,'paired_images':paired_all,'paired_subgroups':paired_summary,
        'illustrations':illustrations}
with tempfile.TemporaryDirectory(prefix='.test_error_stage_',dir=TEST) as tmp:
    d=Path(tmp)
    staged={}
    for k,f in frames.items():
        p=d/OUT[k].name
        f.to_csv(p,index=False)
        check=pd.read_csv(p,dtype={'image_id':str})
        assert len(check)==len(f) and list(check.columns)==list(f.columns)
        staged[k]=p
    mp=d/OUT['metadata'].name
    with open(mp,'w',encoding='utf-8') as f:
        json.dump(metadata,f,indent=2,ensure_ascii=False,allow_nan=False)
    with open(mp,encoding='utf-8') as f:
        assert json.load(f)==metadata
    staged['metadata']=mp
    assert sha(PLAN_PATH)==EXPECTED_PLAN_SHA
    for p in OUT.values(): assert not p.exists(), f'Unexpected prior output: {p}'
    for k,dest in OUT.items():
        os.replace(staged[k],dest)
for k,p in OUT.items():
    assert p.is_file()
    if k!='metadata': assert len(pd.read_csv(p))==metadata['rows'][k]
assert sha(PLAN_PATH)==EXPECTED_PLAN_SHA
print('\n'+'='*95)
print('HELD-OUT TEST ERROR PROFILE COMPLETED')
print('='*95)
print('Outputs saved:',len(OUT))
for k,p in OUT.items(): print(k+':',p.name)
print('Subgroup rows:',len(sub),'| Paired image rows:',len(paired_all),
      '| Paired subgroup rows:',len(paired_summary),'| Illustration candidates:',len(illustrations))
print('Frozen plan unchanged: YES | Checkpoints untouched: YES')
print('Training / Inference / Test retuning: NONE')
print('Analyses are descriptive and post-hoc, not confirmatory.')
print('FINAL STATUS: PASSED')
