# ================================================================
# HELD-OUT TEST — MANUSCRIPT TABLES AND FIGURES V1
# Single Google Colab cell. CPU only. Read-only frozen results.
# 6 tables (CSV), 4 figures (PNG 400 dpi + SVG), captions and provenance.
# NO inference / training / retuning / new significance tests / raw images.
# ================================================================
import os, json, hashlib, tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Locations and frozen analysis plan -------------------------
try:
    MYDRIVE = Path(REAL_MYDRIVE)
except NameError:
    from google.colab import drive
    drive.mount('/content/drive')
    MYDRIVE = Path('/content/drive/MyDrive')

ROOT = MYDRIVE / 'RCF_Aware_AP_Segmentation_Research'
VAL = ROOT / '06_Lesion_Level_Analysis'
TEST = ROOT / '07_Heldout_Test_Evaluation_v1'
DEST = ROOT / '08_Manuscript_Tables_Figures_v1'
PLAN_PATH = VAL / 'heldout_test_analysis_plan_frozen_v1.json'
SYNTH_META = TEST / 'heldout_test_three_seed_analysis_metadata_v1.json'
ERROR_META = TEST / 'heldout_test_error_analysis_exploratory_v1_metadata.json'
LESION = TEST / 'heldout_test_12models_lesion_summary_v1.csv'
PRIMARY = TEST / 'heldout_test_three_seed_primary_per_seed_v1.csv'
MODEL_SUMMARY = TEST / 'heldout_test_three_seed_model_summary_v1.csv'
EFFECTS = TEST / 'heldout_test_three_seed_effects_per_seed_v1.csv'
EFFECT_SUMMARY = TEST / 'heldout_test_three_seed_effect_summary_v1.csv'
BOOT = TEST / 'heldout_test_three_seed_bootstrap_primary_audited_v1.csv'
VAL_TEST = TEST / 'heldout_test_three_seed_validation_vs_test_descriptive_v1.csv'
SUBGROUPS = TEST / 'heldout_test_error_analysis_exploratory_v1_model_subgroups.csv'
PAIRED_SUBGROUPS = TEST / 'heldout_test_error_analysis_exploratory_v1_M6_vs_C2_subgroups.csv'
EXPECTED_PLAN_SHA = '35dbf447bbabb1cec4bacebe5ade8c90d9d9fecf4a6e16f40db4ac1add635800'
SEEDS = [42, 123, 2026]
MODELS = ['M5','M6','C1','C2']
COMPARISONS = ['M6_minus_C2','M6_minus_M5','M6_minus_C1']

input_paths = [PLAN_PATH, SYNTH_META, ERROR_META, LESION, PRIMARY,
               MODEL_SUMMARY, EFFECTS, EFFECT_SUMMARY, BOOT, VAL_TEST,
               SUBGROUPS, PAIRED_SUBGROUPS]
for p in input_paths:
    assert p.is_file(), f'MISSING FROZEN INPUT: {p}'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

before_hashes = {p.name: sha256(p) for p in input_paths}
assert before_hashes[PLAN_PATH.name] == EXPECTED_PLAN_SHA, 'PRE-TEST PLAN CHANGED: STOP'
with open(PLAN_PATH, encoding='utf-8') as f: plan = json.load(f)
with open(SYNTH_META, encoding='utf-8') as f: synth_meta = json.load(f)
with open(ERROR_META, encoding='utf-8') as f: error_meta = json.load(f)
assert plan['primary_analysis']['comparison'] == 'M6_minus_C2'
assert plan['primary_analysis']['endpoint'] == 'Lesion_F1'
assert plan['training_seeds'] == SEEDS
assert set(plan['fixed_models']) == set(MODELS)
assert synth_meta['frozen_pretest_plan_sha256'] == EXPECTED_PLAN_SHA
assert error_meta['frozen_plan_sha256'] == EXPECTED_PLAN_SHA
assert error_meta['status'] == 'EXPLORATORY_POST_TEST_DESCRIPTIVE_ONLY'
assert synth_meta['test_tuning_or_model_selection'] is False

print('='*86)
print('PUBLICATION TABLES & FIGURES — FROZEN TEST RESULTS VERIFIED')
print('='*86)
print('Pre-Test plan SHA256:', EXPECTED_PLAN_SHA)
print('Analysis: 743 Test images | 116 GT AP components | 3 training seeds')
print('Primary endpoint: Lesion F1 | Frozen IoU 0.10 | Min area 1 pixel')
print('Exploratory subgroup analysis will be labeled POST-HOC')

# ---- Load already-computed results; do NOT open images or masks --
primary = pd.read_csv(PRIMARY)
models = pd.read_csv(MODEL_SUMMARY)
effects = pd.read_csv(EFFECTS)
effect_summary = pd.read_csv(EFFECT_SUMMARY)
boot = pd.read_csv(BOOT)
valtest = pd.read_csv(VAL_TEST)
sub = pd.read_csv(SUBGROUPS)
paired_sub = pd.read_csv(PAIRED_SUBGROUPS)
lesion = pd.read_csv(LESION)

assert len(primary)==12 and len(models)==4 and len(effects)==9
assert len(effect_summary)==3 and len(boot)==45 and len(valtest)==12
assert len(sub)==84 and len(paired_sub)==21 and len(lesion)==48
assert not primary.duplicated(['seed','model']).any()
assert not boot.duplicated(['seed','Comparison','Metric']).any()
assert set(primary['seed'])==set(SEEDS) and set(primary['model'])==set(MODELS)
assert set(models['model'])==set(MODELS)
assert set(effect_summary['Comparison'])==set(COMPARISONS)
assert set(sub['seed'])==set(SEEDS) and set(sub['model'])==set(MODELS)
assert sub[(sub['Group']=='ALL_NEGATIVE')]['Images'].eq(644).all()
assert sub[(sub['Group']=='ALL_POSITIVE')]['Images'].eq(99).all()
assert (primary['GT']==116).all() and (primary['Images']==743).all()
assert np.allclose(primary['Lesion_F1'], 2*primary['TP']/(2*primary['TP']+primary['FP']+primary['FN']))
for m in MODELS:
    rows=primary[primary['model']==m].sort_values('seed')
    assert rows['seed'].tolist()==SEEDS
    saved=models[models['model']==m].iloc[0]
    assert np.isclose(float(rows['Lesion_F1'].mean()),float(saved['Mean_Lesion_F1']),atol=1e-12)
    assert np.isclose(float(rows['FP_Components_per_Image'].mean()),float(saved['Mean_FP_Components_per_Image']),atol=1e-12)
for comparison in COMPARISONS:
    e=effects[effects['Comparison']==comparison]
    assert len(e)==3
    saved=effect_summary[effect_summary['Comparison']==comparison].iloc[0]
    assert np.isclose(e['Delta_Lesion_F1'].mean(),saved['Mean_Delta_Lesion_F1'],atol=1e-12)
assert boot['Bootstrap_Resamples'].eq(5000).all()
assert boot['Group'].eq('ALL').all()
boot_f1=boot[boot['Metric']=='Lesion_F1'].copy()
assert len(boot_f1)==9 and set(boot_f1['Comparison'])==set(COMPARISONS)
for r in boot_f1.itertuples(index=False):
    ref=effects[(effects['seed']==r.seed)&(effects['Comparison']==r.Comparison)]
    assert len(ref)==1
    assert np.isclose(float(r.Difference),float(ref['Delta_Lesion_F1'].iloc[0]),atol=1e-10)
    assert r.CI95_Low<=r.CI95_High

# ---- Exactly six manuscript-ready source tables (CSV) ----------
def take_columns(frame, columns):
    absent=[c for c in columns if c not in frame.columns]
    assert not absent, f'MISSING TABLE COLUMNS: {absent}'
    return frame[columns].copy()

t1=take_columns(models,[
    'model','N_seeds','Mean_Lesion_F1','SD_Lesion_F1',
    'Mean_Lesion_Recall','SD_Lesion_Recall',
    'Mean_Lesion_Precision','SD_Lesion_Precision',
    'Mean_FP_Components_per_Image','SD_FP_Components_per_Image',
    'Mean_AP_Negative_FP_Image_Rate','SD_AP_Negative_FP_Image_Rate',
    'Mean_AP_Dice','SD_AP_Dice'])
t1['model']=pd.Categorical(t1['model'], categories=MODELS, ordered=True)
t1=t1.sort_values('model').reset_index(drop=True)
t1['model']=t1['model'].astype(str)

t2=take_columns(effect_summary,[
    'Comparison','N_seeds','Mean_Delta_Lesion_F1','SD_Delta_Lesion_F1',
    'Mean_Delta_Lesion_Recall','SD_Delta_Lesion_Recall',
    'Mean_Delta_FP_Components_per_Image','SD_Delta_FP_Components_per_Image',
    'Positive_Seeds_Lesion_F1','Negative_Seeds_Lesion_F1'])
t2=t2.set_index('Comparison').loc[COMPARISONS].reset_index()

t3=take_columns(primary,[
    'seed','model','GT','TP','FP','FN','Lesion_Recall',
    'Lesion_Precision','Lesion_F1','FP_Components_per_Image',
    'AP_Negative_FP_Image_Rate','AP_Dice','AP_IoU'])
t3=t3.sort_values(['seed','model']).reset_index(drop=True)

t4=take_columns(boot_f1,[
    'seed','Comparison','Difference','CI95_Low','CI95_High',
    'Bootstrap_Resamples'])
t4=t4.sort_values(['Comparison','seed']).reset_index(drop=True)

t5=take_columns(sub[
    sub['model'].isin(['M6','C2']) &
    sub['Group'].isin(['RCF_ONLY','NEITHER'])
],[
    'seed','model','Group','Images','FP','FP_Components_per_Image',
    'Negative_FP_Image_Rate'])
assert len(t5)==12 and set(t5['Images'])=={245,399}
t5=t5.sort_values(['seed','Group','model']).reset_index(drop=True)

t6=take_columns(paired_sub[
    paired_sub['Group'].isin(['RCF_ONLY','NEITHER'])
],[
    'seed','Group','Images','M6_total_FP','C2_total_FP',
    'Delta_FP_M6_minus_C2','Images_M6_Fewer_FP',
    'Images_C2_Fewer_FP','Images_FP_Tie'])
assert len(t6)==6
t6=t6.sort_values(['seed','Group']).reset_index(drop=True)
for r in t6.itertuples(index=False):
    assert r.M6_total_FP-r.C2_total_FP == r.Delta_FP_M6_minus_C2
    assert r.Images_M6_Fewer_FP+r.Images_C2_Fewer_FP+r.Images_FP_Tie==r.Images

tables={
    'Table_1_Test_ThreeSeed_Model_MeanSD.csv':t1,
    'Table_2_Test_Paired_Effect_MeanSD.csv':t2,
    'Table_3_Test_PerSeed_PerModel_Results.csv':t3,
    'Table_4_Test_F1_Bootstrap_PerSeed_95CI.csv':t4,
    'Table_S1_Exploratory_Negative_Subgroups.csv':t5,
    'Table_S2_Exploratory_Paired_FP_Counts.csv':t6,
}

# ---- Publication figures ---------------------------------------
plt.rcParams.update({
    'font.size':10,
    'axes.titlesize':11,
    'axes.labelsize':10,
    'legend.fontsize':9,
    'savefig.dpi':400,
    'figure.facecolor':'white',
    'axes.spines.top':False,
    'axes.spines.right':False,
    'svg.fonttype':'none',
})

STYLES={
    'M5': {'color':'#777777','marker':'D','linestyle':'--'},
    'M6': {'color':'#1D4E89','marker':'o','linestyle':'-'},
    'C1': {'color':'#D18D26','marker':'^','linestyle':':'},
    'C2': {'color':'#B24855','marker':'s','linestyle':'-.'},
}
figures={}

# Figure 1. F1 over seeds: all 12 checkpoints.
fig,ax=plt.subplots(figsize=(7.6,4.4), layout='constrained')
for m in MODELS:
    rows=t3[t3['model']==m].set_index('seed').loc[SEEDS]
    st=STYLES[m]
    ax.plot(range(3),rows['Lesion_F1'].to_numpy(),linewidth=2,markersize=7,
            marker=st['marker'],linestyle=st['linestyle'],color=st['color'],label=m)
ax.set_xticks(range(3),[str(x) for x in SEEDS])
ax.set_ylim(0,0.62)
ax.set_ylabel('Lesion-level F1')
ax.set_xlabel('Training seed')
ax.set_title('Held-out test: lesion F1 across three training seeds')
ax.grid(axis='y',alpha=.20)
ax.legend(ncol=4,loc='upper center',frameon=False)
figures['Figure_1_Test_F1_Across_Seeds']=fig

# Figure 2. Primary and secondary paired effect CIs (image bootstrap).
fig,ax=plt.subplots(figsize=(8.2,5.2), layout='constrained')
comp_label={'M6_minus_C2':'M6 − C2 (primary)',
            'M6_minus_M5':'M6 − M5',
            'M6_minus_C1':'M6 − C1'}
seed_style={42:('o','#255F85'),123:('s','#D18D26'),2026:('D','#99465A')}
ylabels=[]
for j,c in enumerate(COMPARISONS):
    for k,seed in enumerate(SEEDS):
        subrow=t4[(t4['Comparison']==c)&(t4['seed']==seed)].iloc[0]
        y=9-(j*3+k)
        d=float(subrow['Difference']); lo=float(subrow['CI95_Low']); hi=float(subrow['CI95_High'])
        marker,color=seed_style[seed]
        ax.errorbar(d,y,xerr=[[max(0,d-lo)],[max(0,hi-d)]],fmt=marker,
                    markersize=6.5,color=color,capsize=3,elinewidth=1.7)
        ylabels.append((y,f'{comp_label[c]}  |  Seed {seed}'))
ax.axvline(0,color='#444444',linestyle='--',linewidth=1)
for line_y in [6.5,3.5]:
    ax.axhline(line_y,color='#999999',linewidth=.8,alpha=.45)
ax.set_yticks([y for y,_ in ylabels],[s for _,s in ylabels])
ax.set_ylim(.25,9.8)
ax.set_xlabel('Difference in lesion F1 (positive favors M6)')
ax.set_title('Paired image-bootstrap 95% CIs within each training seed')
ax.grid(axis='x',alpha=.16)
figures['Figure_2_Test_Paired_F1_95CI']=fig

# Figure 3. Descriptive negative-subgroup FP burden (post-hoc).
fig,ax=plt.subplots(figsize=(9.0,4.6),layout='constrained')
keys=[(seed,grp) for seed in SEEDS for grp in ['RCF_ONLY','NEITHER']]
x=np.arange(len(keys)); width=.35
for offset,m in [(-width/2,'M6'),(width/2,'C2')]:
    vals=[]
    for seed,grp in keys:
        ss=t5[(t5['seed']==seed)&(t5['Group']==grp)&(t5['model']==m)]
        assert len(ss)==1
        vals.append(float(ss['FP_Components_per_Image'].iloc[0]))
    ax.bar(x+offset,vals,width=width,label=m,color=STYLES[m]['color'])
ax.set_xticks(x,[f'{seed}\n{grp.replace("_"," ")}' for seed,grp in keys])
ax.set_ylabel('False-positive components per image')
ax.set_title('AP-negative test subgroups: exploratory FP burden')
ax.grid(axis='y',alpha=.2)
ax.set_axisbelow(True)
ax.legend(frameon=False)
figures['Figure_3_Exploratory_AP_Negative_Subgroups']=fig

# Figure 4. Validation versus test; describe rather than claim generalization.
fig,ax=plt.subplots(figsize=(7.4,4.3),layout='constrained')
mean_by_split={}
for split in ['Validation','Test']:
    col='Lesion_F1_'+split
    assert col in valtest.columns
    mean_by_split[split]=valtest.groupby('model')[col].mean().reindex(MODELS)
    assert mean_by_split[split].notna().all()
width=.36; x=np.arange(len(MODELS))
ax.bar(x-width/2,mean_by_split['Validation'].to_numpy(),width,
       label='Validation',color='#A4ADB5')
ax.bar(x+width/2,mean_by_split['Test'].to_numpy(),width,
       label='Held-out test',color='#365E83')
ax.set_xticks(x,MODELS)
ax.set_ylim(0,0.65)
ax.set_ylabel('Mean lesion F1 (3 seeds)')
ax.set_title('Descriptive validation-to-test comparison')
ax.grid(axis='y',alpha=.2)
ax.set_axisbelow(True)
ax.legend(frameon=False)
figures['Figure_4_Validation_vs_Test_Descriptive']=fig

# ---- Figure/table notes ----------------------------------------
notes='''# Manuscript tables and figures — frozen held-out test v1

**Dataset:** 743 held-out test radiographs; 116 8-connected AP reference components across 99 AP-positive images. The other 644 images are AP-negative. The training seeds are 42, 123, and 2026.

**Frozen primary endpoint:** lesion F1, probability threshold 0.50, lesion matching IoU ≥ 0.10, minimum predicted-component area 1 pixel, Hungarian maximum-cardinality/IoU matching. All 12 pre-specified checkpoints are included.

**Table 1.** Mean ± sample SD across the three training seeds for each model on the held-out test. The SD reflects descriptive training-seed variability, not a confidence interval.

**Table 2.** Pre-specified paired contrasts (M6−C2 primary; M6−M5 and M6−C1 secondary), summarized across training seeds. The reported differences are in the original metric units.

**Table 3.** Per-seed, per-model primary lesion counts and pixel-level performance, to show the complete distribution rather than cherry-picking a seed.

**Table 4.** Per-seed paired, stratified image-bootstrap 95% percentile intervals for lesion F1 (5,000 draws per seed). These are conditional on the fixed trained checkpoints and do not account for training-seed or patient-level uncertainty.

**Supplementary Tables S1–S2.** Exploratory, post-hoc subgroup false-positive profiles and paired error patterns among AP-negative images. These analyses are descriptive and not additional confirmatory endpoints.

**Figure 1.** Lesion F1 across three seeds for M5, M6, C1 and C2 on the held-out test. Each point represents a frozen checkpoint; lines connect training seeds for visual inspection only.

**Figure 2.** M6 versus C2 (pre-specified primary comparison), M5 and C1 (secondary) in lesion F1, with separately computed image-level paired-bootstrap 95% intervals for each training seed. No pooled seed-level inferential CI is implied.

**Figure 3.** Exploratory AP-negative subgroup false-positive burden in RCF_ONLY and NEITHER, separately for each seed. Post-hoc visualization; interpret descriptively.

**Figure 4.** Validation versus held-out test mean lesion F1 across the same three frozen checkpoints per model. This is descriptive and not a paired comparison across the two image sets.

**Interpretation safeguards:** Do not infer statistical independence at the patient level: patient IDs are unavailable. GT connected components are proxy lesions. The C2 group-preserving RCF shuffle retains stratum-level prevalence. Lower false-positive burden can coexist with lower lesion recall; neither AP Dice nor any one metric should be used as a sole measure of clinical utility. All figures/tables are from already-frozen outputs; no model selection, new inference, or threshold optimization was performed.
'''

# ---- Stage all outputs, verify, then publish without overwrite --
DEST.mkdir(parents=True,exist_ok=True)
expected_names=list(tables)+[stem+ext for stem in figures for ext in ('.png','.svg')]+[
    'Figure_Table_Captions_and_Reporting_Notes.md',
    'publication_tables_figures_provenance_v1.json',
]
assert len(expected_names)==len(set(expected_names))
for name in expected_names:
    assert not (DEST/name).exists(), f'EXISTING OUTPUT — DO NOT OVERWRITE: {DEST/name}'

metadata={
    'version':'1.0','created_utc':datetime.now(timezone.utc).isoformat(),
    'purpose':'manuscript tables and figures from completed frozen test analyses',
    'frozen_plan_sha256':EXPECTED_PLAN_SHA,
    'input_file_sha256':before_hashes,
    'seeds':SEEDS,'models':MODELS,'n_test_images':743,'n_gt_AP_components':116,
    'primary_endpoint':'Lesion_F1','primary_IoU':0.10,'min_pred_area':1,
    'comparisons':COMPARISONS,
    'number_of_bootstrap_draws_per_seed':5000,
    'bootstrap_CIs_conditional_on_checkpoints':True,
    'between_seed_inference':False,
    'subgroup_analysis_is_post_hoc_exploratory':True,
    'patient_disjointness_verified':False,
    'raw_test_images_read':False,'new_model_inference':False,
    'training':False,'test_based_retuning':False,
    'expected_files':expected_names,
    'output_table_rows':{name:len(df) for name,df in tables.items()},
}

with tempfile.TemporaryDirectory(dir=DEST,prefix='.publication_stage_') as tmp:
    staged=Path(tmp)
    for name,frame in tables.items():
        frame.to_csv(staged/name,index=False,float_format='%.10f')
        checked=pd.read_csv(staged/name)
        assert len(checked)==len(frame) and list(checked.columns)==list(frame.columns)
    for stem,fig in figures.items():
        fig.savefig(staged/(stem+'.png'),dpi=400,bbox_inches='tight',facecolor='white')
        fig.savefig(staged/(stem+'.svg'),bbox_inches='tight',facecolor='white')
        assert (staged/(stem+'.png')).stat().st_size>15000
        assert (staged/(stem+'.svg')).stat().st_size>3000
        plt.close(fig)
    (staged/'Figure_Table_Captions_and_Reporting_Notes.md').write_text(notes,encoding='utf-8')
    (staged/'publication_tables_figures_provenance_v1.json').write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    assert sorted(p.name for p in staged.iterdir())==sorted(expected_names)
    for p in input_paths:
        assert sha256(p)==before_hashes[p.name], f'INPUT CHANGED DURING ANALYSIS: {p}'
    for name in expected_names:
        assert not (DEST/name).exists(), f'Output collision: {name}'
    for name in expected_names:
        os.replace(staged/name, DEST/name)

for name in expected_names:
    assert (DEST/name).is_file() and (DEST/name).stat().st_size>0
assert sha256(PLAN_PATH)==EXPECTED_PLAN_SHA
print('\n'+'='*98)
print('PUBLICATION TABLES AND FIGURES COMPLETED')
print('='*98)
print('Saved publication tables:',len(tables),'CSV files')
print('Saved figures:',len(figures),'(each PNG 400 dpi + SVG vector)')
print('Captions and reporting notes: 1')
print('Provenance metadata: 1')
print('Total output files:',len(expected_names))
print('Output directory:',DEST)
print('\nFROZEN TEST — MANUSCRIPT TABLE 1 (SELECTED COLUMNS)')
print(t1[['model','Mean_Lesion_F1','SD_Lesion_F1','Mean_Lesion_Recall',
          'Mean_FP_Components_per_Image','Mean_AP_Dice']].round(4).to_string(index=False))
print('\nFROZEN TEST — MANUSCRIPT TABLE 2 (SELECTED COLUMNS)')
print(t2[['Comparison','Mean_Delta_Lesion_F1','SD_Delta_Lesion_F1',
          'Mean_Delta_FP_Components_per_Image']].round(4).to_string(index=False))
print('\nNo raw images opened. No inference, training, threshold tuning or new bootstrap.')
print('Frozen plan unchanged: YES | Patient independence verified: NO')
print('FINAL STATUS: PASSED')
