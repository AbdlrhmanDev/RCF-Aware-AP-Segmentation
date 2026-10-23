# ================================================================
# PRAD — ANNOTATION-PRESERVING FIGURE CROPS V2 (ONE COLAB CELL)
# Uses ORIGINAL frozen illustrative overlays, NOT generative AI.
# No training, no inference, no threshold changes, no test analysis.
# IMPORTANT: independent visual privacy clearance is still required.
# ================================================================
from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --- 1. Drive and frozen provenance ---
MYDRIVE = Path('/content/drive/MyDrive')
if not MYDRIVE.is_dir():
    from google.colab import drive
    drive.mount('/content/drive')
assert MYDRIVE.is_dir(), 'Google Drive is not mounted.'
ROOT = MYDRIVE / 'RCF_Aware_AP_Segmentation_Research'
SOURCE = ROOT / '08_Manuscript_Tables_Figures_v1' / 'Qualitative_AP_M6_C2_Exploratory_v1'
MANIFEST = SOURCE / 'qualitative_rendered_cases_v1.csv'
PLAN = ROOT / '06_Lesion_Level_Analysis' / 'heldout_test_analysis_plan_frozen_v1.json'
DEST = ROOT / '09_Publication_Ready_Figures_v2_Privacy_DRAFT'
PLAN_SHA = '35dbf447bbabb1cec4bacebe5ade8c90d9d9fecf4a6e16f40db4ac1add635800'

# New output only: do not overwrite a prior publication package.
assert not DEST.exists(), (f'Output directory already exists: {DEST}. Do NOT delete or overwrite it; inspect it first.')
assert MANIFEST.is_file() and PLAN.is_file(), 'Missing original manifest or frozen plan.'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

assert sha256(PLAN) == PLAN_SHA, 'Frozen plan hash mismatch.'
df = pd.read_csv(MANIFEST, dtype={'image_id':str})
assert len(df) == 36 and df['image_id'].nunique() == 28
required = {'figure_filename','figure_sha256','seed','group','image_id',
            'M6_TP','M6_FP','M6_FN','C2_TP','C2_FP','C2_FN'}
assert required.issubset(df.columns)
parsed = df['figure_filename'].astype(str).str.extract(r'^Qualitative_Case_(\d+)_Seed(\d+)\.png$')
assert parsed.notna().all().all(), 'Unexpected original figure filename.'
df['case_number'] = parsed[0].astype(int)
assert df['case_number'].is_unique and set(df['case_number']) == set(range(1,37))
assert np.array_equal(df['seed'].astype(int).to_numpy(), parsed[1].astype(int).to_numpy())

CASES = [1,13,15,23,26,27]
GROUPS = {
    'Figure_05A_AP_Detection':[1,15],
    'Figure_05B_AP_Detection_Seed2026':[26,27],
    'Figure_06_False_Positive_Comparison':[13,23],
}
EXPECTED = {
    1:(42,'AP_ONLY',(1,0,0),(0,3,1)),
    13:(42,'RCF_ONLY',(0,0,0),(0,2,0)),
    15:(123,'AP_ONLY',(2,0,0),(1,1,1)),
    23:(123,'RCF_ONLY',(0,2,0),(0,0,0)),
    26:(2026,'AP_ONLY',(2,0,0),(1,0,1)),
    27:(2026,'AP_RCF',(1,1,1),(2,0,0))
}
selected = df[df['case_number'].isin(CASES)].sort_values('case_number').copy()
assert len(selected) == 6 and selected['image_id'].is_unique
assert selected['seed'].value_counts().to_dict() == {42:2,123:2,2026:2}
rows = {int(r.case_number):r for r in selected.itertuples(index=False)}
for n in CASES:
    row = rows[n]
    seed, group, m6, c2 = EXPECTED[n]
    assert (int(row.seed),str(row.group)) == (seed,group)
    assert (int(row.M6_TP),int(row.M6_FP),int(row.M6_FN)) == m6
    assert (int(row.C2_TP),int(row.C2_FP),int(row.C2_FN)) == c2
    source_file = SOURCE / str(row.figure_filename)
    assert source_file.is_file() and sha256(source_file) == str(row.figure_sha256)

print('='*86)
print('FROZEN FIGURE SOURCE / MODEL COUNTS / SIX UNIQUE IMAGE IDs: VERIFIED')
print('='*86)
print(selected[['case_number','seed','group']].to_string(index=False))

# --- 2. Extract exactly the original 4 panels; remove old crowded titles only. ---
def original_four_panels(source_file):
    with Image.open(source_file) as image:
        im = image.convert('RGB')
    a = np.asarray(im)
    h,w = a.shape[:2]
    detected = None
    for y in range(int(.13*h),int(.30*h)):
        dark = (a[y].min(axis=1) < 245)
        diffs = np.diff(np.r_[False,dark,False].astype(np.int8))
        bands = [(int(s),int(e)) for s,e in zip(np.where(diffs==1)[0],np.where(diffs==-1)[0])]
        bands = [(s,e) for s,e in bands if .20*w < e-s < .27*w]
        if len(bands)==4 and max(e-s for s,e in bands)-min(e-s for s,e in bands)<=5:
            detected=(y,bands)
            break
    assert detected is not None, f'Original four-panel geometry not found: {source_file.name}'
    top,bands = detected
    side=min(e-s for s,e in bands)
    assert side>=350 and top+side<=h
    return [np.asarray(im.crop((start,top,start+side,top+side))).copy() for start,end in bands]

# --- 3. Crop only when ALL reference and prediction overlays are protected. ---
# This is an ADAPTIVE conservative crop. Each case has its own crop limits.
# In case 13, RCF cyan pixels extend into the desired top crop; the crop is
# expanded to INCLUDE them. We do not disable the color-pixel safety check.
# Text/markings may remain after safe cropping. HUMAN VISUAL PRIVACY REVIEW
# IS STILL REQUIRED. NEVER PUBLISH DRAFT FILES WITHOUT THAT REVIEW.
# Small, visually localized *printed border* regions, in proportions of the
# ORIGINAL source tile. These are NOT lesions, not diagnostic pixels, and the
# code refuses to redact if any colored AP/RCF/prediction pixel is detected.
# One can inspect all exact black rectangles in the provenance JSON.
# The original source is never overwritten. Final human privacy QC remains mandatory.
PERIPHERAL_REDACTIONS = {
    13: [(0.085, 0.000, 0.385, 0.035),  # printed characters, top-left
         (0.79, 0.925, 0.96, 0.999)],  # laterality marker, lower right
    15: [(0.910, 0.430, 0.999, 0.999)], # vertical film strip code
    23: [(0.075, 0.930, 0.420, 0.999)], # bottom film code/date
    26: [(0.075, 0.930, 0.450, 0.999)], # bottom film code/date
    27: [(0.920, 0.430, 0.999, 0.999)]  # right-side film code
}

CROP = {'left_fraction': .045, 'right_fraction': .060,
        'top_of_content_fraction': .065, 'bottom_of_content_fraction': .085}
ANNOTATION_MARGIN_PIXELS = 8

def overlay_mask(a):
    # Conservative RGB chroma guard covering AP green/cyan/orange/magenta.
    rgb = a.astype(np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    return (spread > 35) & (rgb.max(axis=2) > 100)

def clean_panels(case_number):
    raw_panels = original_four_panels(SOURCE / str(rows[case_number].figure_filename))
    assert len(raw_panels) == 4
    side = raw_panels[0].shape[0]
    assert all(a.shape == (side,side,3) for a in raw_panels)

    # Determine the photographed portion of the original UNANNOTATED panel.
    orig = raw_panels[0]
    grey = orig.mean(axis=2)
    center = grey[:,int(side*.15):int(side*.85)]
    active = np.flatnonzero(np.quantile(center,.65,axis=1) > 40)
    assert len(active) > int(.65*side), f'Unusual image content: Case {case_number}'
    photo_y0,photo_y1 = int(active[0]),int(active[-1])+1
    photo_h = photo_y1-photo_y0

    proposed = [
        int(round(side*CROP['left_fraction'])),
        photo_y0 + int(round(photo_h*CROP['top_of_content_fraction'])),
        int(round(side*(1-CROP['right_fraction']))),
        photo_y1-int(round(photo_h*CROP['bottom_of_content_fraction']))
    ]
    masks = [overlay_mask(a) for a in raw_panels]
    union = np.logical_or.reduce(masks)
    yy,xx = np.where(union)
    assert len(xx) > 0, f'No annotation pixels detected: Case {case_number}'
    protected_box = [
        max(0,int(xx.min())-ANNOTATION_MARGIN_PIXELS),
        max(0,int(yy.min())-ANNOTATION_MARGIN_PIXELS),
        min(side,int(xx.max())+1+ANNOTATION_MARGIN_PIXELS),
        min(side,int(yy.max())+1+ANNOTATION_MARGIN_PIXELS)
    ]

    # Only EXPAND the requested crop when it would cut coloured ground-truth
    # or predictions. Keep same crop for all four panels (pixel alignment).
    x0=min(proposed[0],protected_box[0])
    y0=min(proposed[1],protected_box[1])
    x1=max(proposed[2],protected_box[2])
    y1=max(proposed[3],protected_box[3])
    assert 0 <= x0 < x1 <= side and 0 <= y0 < y1 <= side
    assert x1-x0 >= .70*side and y1-y0 >= .55*side

    edited=[]
    annotation_counts=[]
    performed_redactions=[]
    # Coordinates recorded before trimming, identical across all four panels.
    for box_frac in PERIPHERAL_REDACTIONS.get(case_number, []):
        bx0,by0,bx1,by1=[int(round(q*side)) for q in box_frac]
        assert 0 <= bx0 < bx1 <= side and 0 <= by0 < by1 <= side
        hits=sum(int(mask[by0:by1,bx0:bx1].sum()) for mask in masks)
        assert hits==0, (
            f'Case {case_number}: proposed text-redaction rectangle '
            f'{(bx0,by0,bx1,by1)} intersects {hits} colored overlay pixels. STOP.'
        )
        performed_redactions.append([bx0,by0,bx1,by1])
    for j,a in enumerate(raw_panels):
        coloured=masks[j]
        outside=coloured.copy()
        outside[y0:y1,x0:x1]=False
        lost=int(outside.sum())
        assert lost==0, (
            f'Case {case_number}, panel {j+1}: {lost} annotation pixels would be removed. STOP.'
        )
        # Work on a PRIVATE COPY only, never modify the source PNG.
        canvas=a.copy()
        for bx0,by0,bx1,by1 in performed_redactions:
            # Opaque black patch, rather than generative inpainting or blending.
            # This makes any privacy redaction transparent and auditable.
            canvas[by0:by1,bx0:bx1,:]=0
        tile=canvas[y0:y1,x0:x1].copy()
        # RGB overlay colours must be strictly unchanged everywhere retained.
        assert np.array_equal(tile[coloured[y0:y1,x0:x1]],
                              a[y0:y1,x0:x1][coloured[y0:y1,x0:x1]])
        annotation_counts.append(int(coloured.sum()))
        edited.append(Image.fromarray(tile))  # no deprecated mode parameter

    adaptation = tuple(proposed) != (x0,y0,x1,y1)
    audit = {
        'requested_crop_xyxy':proposed,
        'applied_crop_xyxy':[x0,y0,x1,y1],
        'annotation_union_bbox_xyxy':protected_box,
        'annotation_margin_pixels':ANNOTATION_MARGIN_PIXELS,
        'source_panel_side':side,
        'image_content_y0':photo_y0,
        'image_content_y1':photo_y1,
        'overlay_pixels_per_panel':annotation_counts,
        'annotated_pixels_removed':0,
        'explicit_redaction_rectangles_original_xyxy':performed_redactions,
        'redaction_overlap_annotation_pixels':0,
        'adaptive_crop_expanded':adaptation,
        'possible_printed_edge_text_remaining':True
    }
    return edited,audit

panels={}
crop_audit={}
for n in CASES:
    panels[n],crop_audit[n]=clean_panels(n)
    audit=crop_audit[n]
    applied=audit['applied_crop_xyxy']
    action='ADAPTIVE (annotation protected)' if audit['adaptive_crop_expanded'] else 'STANDARD'
    print(f'Case {n:02d}: {action}; annotations removed=0; crop={applied}')
    if audit['adaptive_crop_expanded']:
        print(f'  SAFETY NOTE: wider crop retained to protect masks. Check any visible edge text.')

# --- 4. Regenerate titles and layouts, with no identifying text on the figures. ---
def load_font(sz,bold=False):
    suffix='-Bold' if bold else ''
    path=f'/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf'
    try: return ImageFont.truetype(path,sz)
    except OSError: return ImageFont.load_default()

F_TITLE=load_font(25,True)
F_LABEL=load_font(19,True)
F_SMALL=load_font(17)
COLS=['Input radiograph','Reference masks','M6: true RCF labels','C2: shuffled RCF labels']
TITLES={
    'Figure_05A_AP_Detection':'AP lesion examples: training seeds 42 and 123',
    'Figure_05B_AP_Detection_Seed2026':'AP lesion examples: training seed 2026',
    'Figure_06_False_Positive_Comparison':'AP-negative examples: false-positive comparison',
}

def build_figure(numbers,heading):
    # All four tiles in a given case have identical sizes after exact cropping.
    # No image content is resized or recomputed.
    tile_w=max(panels[n][0].width for n in numbers)
    tile_h=max(panels[n][0].height for n in numbers)
    gutter=12; margin=22; row_label_h=72; title_h=72; gap_between_rows=22
    width=margin*2+4*tile_w+3*gutter
    row_height=row_label_h+tile_h+gap_between_rows
    height=title_h+len(numbers)*row_height+42
    canvas=Image.new('RGB',(width,height),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((margin,12),heading,font=F_TITLE,fill='#18202c')
    draw.line((margin,title_h-12,width-margin,title_h-12),fill='#ccd4df',width=2)
    for index,n in enumerate(numbers):
        r=rows[n]
        y=title_h+index*row_height
        draw.text((margin,y),f'Case {n:02d}   |   Seed {r.seed}   |   {r.group}',
                  fill='#152133',font=F_LABEL)
        for j,tile in enumerate(panels[n]):
            x=margin+j*(tile_w+gutter)
            draw.text((x,y+25),COLS[j],font=F_SMALL,fill='#1f2937')
            if j>=2:
                model='M6' if j==2 else 'C2'
                scores=f'TP {getattr(r,model+"_TP")} / FP {getattr(r,model+"_FP")} / FN {getattr(r,model+"_FN")}'
                draw.text((x,y+45),scores,font=F_SMALL,fill='#374151')
            canvas.paste(tile,(x,y+row_label_h))
        draw.line((margin,y+row_label_h+tile_h+10,width-margin,
                   y+row_label_h+tile_h+10),fill='#e2e8f0',width=1)
    return canvas

# --- 5. Stage all outputs and commit to NEW Drive directory only if all pass. ---
DEST.parent.mkdir(parents=True,exist_ok=True)
results={}
with tempfile.TemporaryDirectory(prefix='.figure_privacy_stage_',dir=DEST.parent) as td:
    stage=Path(td)
    for n in CASES:
        png=stage/f'Case_{n:02d}_Four_Panels.png'
        build_figure([n],f'Qualitative Test case {n:02d}').save(png,format='PNG',dpi=(300,300))
        results[png.name]=sha256(png)
    for name,numbers in GROUPS.items():
        composite=build_figure(numbers,TITLES[name])
        for ext in ('png','tiff'):
            p=stage/f'{name}.{ext}'
            composite.save(p,format='PNG' if ext=='png' else 'TIFF',dpi=(300,300))
            assert p.is_file()
            results[p.name]=sha256(p)

    selected[['case_number','seed','image_id','group',
              'M6_TP','M6_FP','M6_FN','C2_TP','C2_FP','C2_FN']].to_csv(
              stage/'figure_provenance_for_internal_use_v1.csv',index=False)
    (stage/'Figure_Captions_v1.txt').write_text(
        'FIGURE 5A — AP lesion examples (Cases 01 and 15).\n'
        'FIGURE 5B — AP lesion examples at seed 2026 (Cases 26 and 27).\n'
        'FIGURE 6 — AP-negative, RCF-positive examples (Cases 13 and 23).\n\n'
        'Column order: input radiograph; reference AP (green) and RCF (cyan); '
        'M6 AP prediction (orange); C2 AP prediction (magenta). '
        'Green contours within prediction panels denote reference AP. '
        'Reported TP, FP, and FN use frozen connected-component evaluation '
        '(IoU >= 0.10; predicted probability >= 0.50; min area 1 pixel). '
        'Selected post-hoc for illustration, not representative or confirmatory.\n'
        'WARNING: edge text may remain where scientific overlays require full field. Figure files need independent visual privacy screening '
        'and radiograph/data licensing review before journal submission.\n',
        encoding='utf-8')
    meta={
        'status':'DRAFT_V2_PENDING_HUMAN_PRIVACY_CLEARANCE',
        'method':'annotation_preserving_adaptive_crop_plus_audited_peripheral_opaque_redaction',
        'not_AI_generated':True,
        'unredacted_retained_radiograph_pixels_unchanged':True,
        'adaptive_cropping_for_annotation_safety':True,
        'edge_text_fully_redacted':'NOT_YET_HUMAN_VERIFIED',
        'annotation_pixels_removed':0,
        'cropping_parameters':CROP,
        'per_case_crop_coordinates':{str(k):v for k,v in crop_audit.items()},
        'selected_case_numbers':CASES,
        'source_manifest_sha256':sha256(MANIFEST),
        'frozen_plan_sha256':PLAN_SHA,
        'original_figure_sha256':{str(r.figure_filename):str(r.figure_sha256)
                                   for r in selected.itertuples(index=False)},
        'outputs_sha256':results,
        'patient_independence_verified':False,
        'new_inference':False,
        'test_metrics_modified':False,
        'human_privacy_clearance':False,
        'created_utc':datetime.now(timezone.utc).isoformat()
    }
    (stage/'Figure_Privacy_Provenance_v1.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    assert len(list(stage.glob('Case_*_Four_Panels.png')))==6
    assert len(list(stage.glob('Figure_*.png')))==3
    assert len(list(stage.glob('Figure_*.tiff')))==3
    for name,h in results.items(): assert sha256(stage/name)==h
    assert sha256(PLAN)==PLAN_SHA
    for r in selected.itertuples(index=False):
        assert sha256(SOURCE/r.figure_filename)==str(r.figure_sha256)
    # An atomic move preserves all original sources; no overwrite.
    assert not DEST.exists()
    os.rename(stage,DEST)

print('\n'+'='*90)
print('THREE FIGURES + SIX CASES — SAFE PRIVACY-REDACTION DRAFT CREATED')
print('='*90)
print('Three final-named figures, each PNG and TIFF, plus six single-case PNGs: SAVED')
print('No coloured AP/RCF GT or predictions cropped: VERIFIED (all cases)')
print('Peripheral printed edge markings treated with audited rectangles; MANUAL PRIVACY REVIEW REQUIRED.')
print('FROZEN TEST PLAN: UNCHANGED | INFERENCE: NONE | ORIGINAL FIGURES: UNCHANGED')
print('Google Drive folder:', DEST)
print('PUBLICATION PRIVACY APPROVAL: STILL REQUIRES HUMAN REVIEW')
print('FINAL STATUS: PASSED — V2 DRAFT ONLY; PRIVACY CLEARANCE PENDING')
