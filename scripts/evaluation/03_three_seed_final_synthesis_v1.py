# ================================================================
# HELD-OUT TEST — FINAL THREE-SEED SYNTHESIS (FROZEN V1)
# One Colab cell, CPU only. No inference/training/tuning.
# Reads frozen Test evaluation and bootstrap outputs; no raw images.
# ================================================================
import os, json, hashlib, tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

if 'REAL_MYDRIVE' not in globals():
    from google.colab import drive
    drive.mount('/content/drive')
    REAL_MYDRIVE = Path('/content/drive/MyDrive')
ROOT = Path(REAL_MYDRIVE) / 'RCF_Aware_AP_Segmentation_Research'
VAL_DIR = ROOT / '06_Lesion_Level_Analysis'
TEST_DIR = ROOT / '07_Heldout_Test_Evaluation_v1'
PLAN = VAL_DIR / 'heldout_test_analysis_plan_frozen_v1.json'
TEST_META = TEST_DIR / 'heldout_test_12models_metadata_v1.json'
TEST_LESION = TEST_DIR / 'heldout_test_12models_lesion_summary_v1.csv'
TEST_PIXEL = TEST_DIR / 'heldout_test_12models_pixel_summary_v1.csv'
TEST_PER_IMAGE = TEST_DIR / 'heldout_test_12models_lesion_per_image_v1.csv'
BOOT = TEST_DIR / 'heldout_test_paired_stratified_bootstrap_v1.csv'
BOOT_META = TEST_DIR / 'heldout_test_paired_stratified_bootstrap_metadata_v1.json'
VAL_PRIMARY = VAL_DIR / 'three_seed_C2_M5_M6_C1_primary_per_seed_v1.csv'
FROZEN_PLAN_SHA = '35dbf447bbabb1cec4bacebe5ade8c90d9d9fecf4a6e16f40db4ac1add635800'
SEEDS = [42, 123, 2026]
MODELS = ['M5','M6','C1','C2']
SETTINGS = [(0.10,1),(0.50,1),(0.10,16),(0.50,16)]
COMPARISONS = [('M6','C2'),('M6','M5'),('M6','C1')]
METRICS = ['Lesion_F1','Lesion_Recall','Lesion_Precision','FP_Components_per_Image','AP_Negative_FP_Image_Rate']
PREFIX = 'heldout_test_three_seed'
OUTPUT = {
    'primary': TEST_DIR / (PREFIX+'_primary_per_seed_v1.csv'),
    'model_summary': TEST_DIR / (PREFIX+'_model_summary_v1.csv'),
    'effect_per_seed': TEST_DIR / (PREFIX+'_effects_per_seed_v1.csv'),
    'effect_summary': TEST_DIR / (PREFIX+'_effect_summary_v1.csv'),
    'bootstrap_primary': TEST_DIR / (PREFIX+'_bootstrap_primary_audited_v1.csv'),
    'sensitivity': TEST_DIR / (PREFIX+'_sensitivity_mean_sd_v1.csv'),
    'validation_test': TEST_DIR / (PREFIX+'_validation_vs_test_descriptive_v1.csv'),
    'metadata': TEST_DIR / (PREFIX+'_analysis_metadata_v1.json'),
}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

for path in [PLAN, TEST_META, TEST_LESION, TEST_PIXEL, TEST_PER_IMAGE, BOOT, BOOT_META, VAL_PRIMARY]:
    assert path.is_file(), f'MISSING INPUT: {path}'
for path in OUTPUT.values():
    assert not path.exists(), f'OUTPUT ALREADY EXISTS; DO NOT OVERWRITE: {path}'
assert sha256(PLAN)==FROZEN_PLAN_SHA, 'Frozen plan SHA mismatch; STOP'
with open(PLAN, encoding='utf-8') as f: plan=json.load(f)
with open(TEST_META,encoding='utf-8') as f: test_meta=json.load(f)
with open(BOOT_META,encoding='utf-8') as f: boot_meta=json.load(f)
assert plan['primary_analysis']['comparison']=='M6_minus_C2'
assert plan['primary_analysis']['endpoint']=='Lesion_F1'
assert plan['training_seeds']==SEEDS and set(plan['fixed_models'])==set(MODELS)
assert test_meta['status']=='PASSED' and test_meta['frozen_pretest_plan_sha256']==FROZEN_PLAN_SHA
assert test_meta['output_rows']=={'per_image':35664,'pixel':12,'lesion':48}
assert boot_meta['frozen_pretest_plan_sha256']==FROZEN_PLAN_SHA
assert boot_meta['test_lesion_per_image_sha256']==sha256(TEST_PER_IMAGE)
assert boot_meta['test_lesion_summary_sha256']==sha256(TEST_LESION)
assert boot_meta['test_inference_metadata_sha256']==sha256(TEST_META)
assert boot_meta['result_rows']==306 and boot_meta['bootstrap_resamples_per_seed']==5000
assert boot_meta['bootstrap_draws_pooled_across_seeds'] is False
assert boot_meta['captures_training_seed_uncertainty'] is False
assert boot_meta['patient_independence_verified'] is False

print('='*95)
print('FINAL TEST THREE-SEED SYNTHESIS — PRECHECK VERIFIED')
print('='*95)
print('Plan SHA256:',FROZEN_PLAN_SHA)
print('Models:',MODELS,'| Seeds:',SEEDS,'| 743 Test images | 116 GT components')
print('Image-level bootstrap: 5000 resamples/seed; no seed pooling')

lesion=pd.read_csv(TEST_LESION)
pixel=pd.read_csv(TEST_PIXEL)
per_image=pd.read_csv(TEST_PER_IMAGE,dtype={'image_id':str})
bootstrap=pd.read_csv(BOOT)
val=pd.read_csv(VAL_PRIMARY)
assert len(lesion)==48 and len(pixel)==12 and len(per_image)==35664
assert len(bootstrap)==306 and len(val)==12
assert set(lesion['model'])==set(MODELS) and set(lesion['seed'])==set(SEEDS)
assert not lesion.duplicated(['seed','model','IoU_threshold','min_pred_area']).any()
assert not pixel.duplicated(['seed','model']).any()
assert not val.duplicated(['seed','model']).any()
assert not per_image.duplicated(['seed','model','image_id','IoU_threshold','min_pred_area']).any()
assert per_image['pretest_plan_sha256'].eq(FROZEN_PLAN_SHA).all()
assert per_image['validation_protocol_sha256'].eq(plan['lesion_evaluation']['validation_protocol_sha256']).all()
assert (per_image['TP']+per_image['FP']==per_image['Pred_components']).all()
assert (per_image['TP']+per_image['FN']==per_image['GT_components']).all()
assert (per_image[['TP','FP','FN','GT_components','Pred_components']]>=0).all().all()
assert lesion['GT'].eq(116).all() and lesion['Images'].eq(743).all()
assert set((round(float(i),2),int(a)) for i,a in lesion[['IoU_threshold','min_pred_area']].itertuples(index=False,name=None))==set(SETTINGS)
assert pixel['AP_Negative_FP_Image_Rate'].notna().all()

# Independent audit of every Test lesion-summary row against per-image records.
for row in lesion.itertuples(index=False):
    sub=per_image[(per_image['model']==row.model)&(per_image['seed']==row.seed)&
                  np.isclose(per_image['IoU_threshold'],row.IoU_threshold)&
                  (per_image['min_pred_area']==row.min_pred_area)]
    assert len(sub)==743 and sub['image_id'].is_unique
    assert all(int(sub[col].sum())==int(getattr(row,col)) for col in ['TP','FP','FN'])
    assert int(sub['GT_components'].sum())==116
    t,p,n=int(row.TP),int(row.FP),int(row.FN)
    assert np.isclose(2*t/(2*t+p+n),float(row.Lesion_F1),atol=1e-12)
    assert np.isclose(t/(t+n),float(row.Lesion_Recall),atol=1e-12)
    assert np.isclose(p/743,float(row.FP_Components_per_Image),atol=1e-12)
    neg=sub[sub['GT_components']==0]
    assert len(neg)==644
    assert np.isclose(float(neg['FP'].gt(0).mean()),float(row.AP_Negative_FP_Image_Rate),atol=1e-12)

primary=lesion[np.isclose(lesion['IoU_threshold'],0.10)&(lesion['min_pred_area']==1)].copy()
assert len(primary)==12
primary=primary.sort_values(['seed','model']).reset_index(drop=True)
primary=primary.merge(pixel,how='left',on=['seed','model'],validate='one_to_one',suffixes=('','_pixel'))
assert len(primary)==12 and primary['AP_Dice'].notna().all()
assert np.allclose(primary['AP_Negative_FP_Image_Rate'],primary['AP_Negative_FP_Image_Rate_pixel'],atol=1e-12)
primary=primary.drop(columns=['AP_Negative_FP_Image_Rate_pixel'])

summaries=[]
for model in MODELS:
    rows=primary[primary['model']==model].sort_values('seed')
    assert rows['seed'].tolist()==SEEDS
    out={'model':model,'N_seeds':3}
    for m in METRICS+['AP_Dice','AP_IoU','Tooth_Dice']:
        arr=rows[m].to_numpy(dtype=float)
        out['Mean_'+m]=float(arr.mean())
        out['SD_'+m]=float(arr.std(ddof=1))
    if model in ['M6','C2']:
        rcf=rows['RCF_Dice'].to_numpy(dtype=float)
        out['Mean_RCF_Dice']=float(rcf.mean())
        out['SD_RCF_Dice']=float(rcf.std(ddof=1))
    else:
        out['Mean_RCF_Dice']=np.nan
        out['SD_RCF_Dice']=np.nan
    summaries.append(out)
model_summary=pd.DataFrame(summaries)

paired=[]
for seed in SEEDS:
    table=primary[primary['seed']==seed].set_index('model')
    for candidate,baseline in COMPARISONS:
        row={'seed':seed,'Comparison':f'{candidate}_minus_{baseline}'}
        for m in METRICS:
            row['Delta_'+m]=float(table.loc[candidate,m]-table.loc[baseline,m])
        paired.append(row)
effects=pd.DataFrame(paired)
assert len(effects)==9

effect_summary_rows=[]
for candidate,baseline in COMPARISONS:
    cname=f'{candidate}_minus_{baseline}'
    sub=effects[effects['Comparison']==cname].sort_values('seed')
    assert sub['seed'].tolist()==SEEDS
    row={'Comparison':cname,'N_seeds':3}
    for m in METRICS:
        x=sub['Delta_'+m].to_numpy(dtype=float)
        row['Mean_Delta_'+m]=float(x.mean())
        row['SD_Delta_'+m]=float(x.std(ddof=1))
        row['Positive_Seeds_'+m]=int((x>0).sum())
        row['Negative_Seeds_'+m]=int((x<0).sum())
    effect_summary_rows.append(row)
effect_summary=pd.DataFrame(effect_summary_rows)

# Audit ALL primary bootstrap metric differences against Test point estimates.
bootstrap_primary=bootstrap[(bootstrap['Group']=='ALL')&
   np.isclose(bootstrap['IoU_threshold'],0.10)&(bootstrap['min_pred_area']==1)].copy()
assert len(bootstrap_primary)==45
assert not bootstrap_primary.duplicated(['seed','Comparison','Metric']).any()
assert set(bootstrap_primary['Metric'])==set(METRICS)
assert bootstrap_primary['Bootstrap_Resamples'].eq(5000).all()
assert bootstrap_primary['CI95_Low'].le(bootstrap_primary['CI95_High']).all()
for row in bootstrap_primary.itertuples(index=False):
    assert row.Comparison in [f'{a}_minus_{b}' for a,b in COMPARISONS]
    cand,base=row.Comparison.split('_minus_')
    sub=primary[primary['seed']==row.seed].set_index('model')
    assert abs(float(row.Candidate)-float(sub.loc[cand,row.Metric]))<1e-10
    assert abs(float(row.Baseline)-float(sub.loc[base,row.Metric]))<1e-10
    assert abs(float(row.Difference)-(float(row.Candidate)-float(row.Baseline)))<1e-10

sens=[]
for iou,area in SETTINGS:
    for model in MODELS:
        s=lesion[(lesion['model']==model)&np.isclose(lesion['IoU_threshold'],iou)&(lesion['min_pred_area']==area)]
        assert len(s)==3 and set(s['seed'])==set(SEEDS)
        record={'model':model,'IoU_threshold':iou,'min_pred_area':area,'N_seeds':3}
        for m in METRICS:
            values=s[m].to_numpy(dtype=float)
            record['Mean_'+m]=float(values.mean())
            record['SD_'+m]=float(values.std(ddof=1))
        sens.append(record)
sensitivity=pd.DataFrame(sens)
assert len(sensitivity)==16

val_compare=primary[['seed','model']+METRICS].merge(
    val[['seed','model']+METRICS],on=['seed','model'],how='inner',validate='one_to_one',
    suffixes=('_Test','_Validation'))
assert len(val_compare)==12
for m in METRICS:
    val_compare['Delta_Test_minus_Validation_'+m]=val_compare[m+'_Test']-val_compare[m+'_Validation']

# Freeze analysis provenance, with no between-seed inference from n=3.
metadata={
  'version':'v1','analysis':'heldout test 3-seed descriptive synthesis and bootstrap audit',
  'created_utc':datetime.now(timezone.utc).isoformat(),
  'frozen_pretest_plan_sha256':FROZEN_PLAN_SHA,
  'input_hashes':{p.name:sha256(p) for p in [TEST_META,TEST_LESION,TEST_PIXEL,TEST_PER_IMAGE,BOOT,BOOT_META,VAL_PRIMARY]},
  'models':MODELS,'seeds':SEEDS,'test_images':743,'test_gt_components':116,
  'primary_comparison':'M6_minus_C2','primary_metric':'Lesion_F1',
  'primary_iou':0.10,'primary_min_area':1,
  'bootstrap':{'n_per_seed':5000,'conditional_on_checkpoints':True,'pool_draws_across_seeds':False},
  'seed_effects':{'mean_and_sample_sd_only':True,'n_seeds':3,'seed_level_inferential_CI':False},
  'validation_test_comparison':'descriptive only; distinct validation and test image sets',
  'patient_disjointness_verified':False,'frozen_plan_modified':False,
  'test_tuning_or_model_selection':False,'training':False,'inference':False,
  'output_rows':{'primary':12,'model_summary':4,'effect_per_seed':9,'effect_summary':3,
                 'bootstrap_primary':45,'sensitivity':16,'validation_test':12},
}

print('\n'+'='*100)
print('FROZEN TEST — THREE-SEED MODEL SUMMARY (MEAN ± SD)')
print('='*100)
show_cols=['model','Mean_Lesion_F1','SD_Lesion_F1','Mean_Lesion_Recall',
           'Mean_FP_Components_per_Image','Mean_AP_Negative_FP_Image_Rate','Mean_AP_Dice']
print(model_summary[show_cols].round(4).to_string(index=False))
print('\n'+'='*100)
print('FROZEN TEST — PAIRED EFFECTS BY SEED')
print('='*100)
print(effects[['seed','Comparison','Delta_Lesion_F1','Delta_Lesion_Recall',
   'Delta_FP_Components_per_Image','Delta_AP_Negative_FP_Image_Rate']].round(4).to_string(index=False))
print('\n'+'='*100)
print('FROZEN TEST — THREE-SEED EFFECT SUMMARY (DESCRIPTIVE)')
print('='*100)
print(effect_summary[['Comparison','Mean_Delta_Lesion_F1','SD_Delta_Lesion_F1',
  'Mean_Delta_FP_Components_per_Image','SD_Delta_FP_Components_per_Image',
  'Positive_Seeds_Lesion_F1','Negative_Seeds_Lesion_F1']].round(4).to_string(index=False))
print('\n'+'='*100)
print('FROZEN TEST — PRIMARY F1 BOOTSTRAP CIs (FIXED-CHECKPOINT)')
print('='*100)
print(bootstrap_primary[bootstrap_primary['Metric']=='Lesion_F1'][
  ['seed','Comparison','Difference','CI95_Low','CI95_High']].sort_values(
  ['Comparison','seed']).round(4).to_string(index=False))

outputs_df={'primary':primary,'model_summary':model_summary,'effect_per_seed':effects,
            'effect_summary':effect_summary,'bootstrap_primary':bootstrap_primary,
            'sensitivity':sensitivity,'validation_test':val_compare}
for k,df in outputs_df.items():
    assert len(df)==metadata['output_rows'][k]
assert sha256(PLAN)==FROZEN_PLAN_SHA
for path in OUTPUT.values():
    assert not path.exists(), f'Existing output; stopped to prevent overwrite: {path}'
with tempfile.TemporaryDirectory(dir=TEST_DIR,prefix='.three_seed_test_stage_') as tmp:
    staged={}
    tmp=Path(tmp)
    for k,df in outputs_df.items():
        p=tmp/OUTPUT[k].name
        df.to_csv(p,index=False)
        check=pd.read_csv(p)
        assert len(check)==len(df) and list(check.columns)==list(df.columns)
        staged[k]=p
    mp=tmp/OUTPUT['metadata'].name
    with open(mp,'w',encoding='utf-8') as f:
        json.dump(metadata,f,indent=2,ensure_ascii=False,allow_nan=False)
    with open(mp,encoding='utf-8') as f:
        assert json.load(f)==metadata
    staged['metadata']=mp
    assert sha256(PLAN)==FROZEN_PLAN_SHA
    for path in OUTPUT.values():
        assert not path.exists()
    for k,path in OUTPUT.items():
        os.replace(staged[k],path)
assert sha256(PLAN)==FROZEN_PLAN_SHA
assert all(p.is_file() for p in OUTPUT.values())
print('\n'+'='*100)
print('FROZEN HELD-OUT TEST THREE-SEED SYNTHESIS COMPLETED')
print('='*100)
print('Saved 8 NEW files:')
for name,path in OUTPUT.items():
    print(f'  {name}: {path.name}')
print('Training: NONE | Inference: NONE | New bootstrap: NONE')
print('Threshold retuning: NONE | Original outputs overwritten: NONE')
print('Frozen plan SHA256: VERIFIED | Patient independence: NOT VERIFIED')
print('FINAL STATUS: PASSED')
