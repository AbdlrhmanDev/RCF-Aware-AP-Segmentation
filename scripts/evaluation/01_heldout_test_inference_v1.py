# ============================================================
# HELD-OUT TEST EVALUATION — 12 FROZEN CHECKPOINTS (V1)
# Run as ONE Google Colab cell after mounting Google Drive.
# Frozen protocol; no training, tuning or bootstrap in this cell.
# Per-checkpoint atomic progress files enable safe restart.
# ============================================================
import os, gc, json, hashlib, time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader, SequentialSampler

if "REAL_MYDRIVE" not in globals():
    from google.colab import drive
    drive.mount("/content/drive")
    REAL_MYDRIVE = Path("/content/drive/MyDrive")

assert torch.cuda.is_available(), "Enable a GPU runtime before running."
assert smp.__version__ == "0.5.0", "The frozen environment uses SMP 0.5.0."
DEVICE = torch.device("cuda")
ROOT = Path(REAL_MYDRIVE) / "RCF_Aware_AP_Segmentation_Research"
RAW = Path(REAL_MYDRIVE) / "PRAD_MultiTask_Segmentation"
ANALYSIS = ROOT / "06_Lesion_Level_Analysis"
SPLITS = ROOT / "02_Dataset_Splits"
PLAN_PATH = ANALYSIS / "heldout_test_analysis_plan_frozen_v1.json"
AUDIT_PATH = ANALYSIS / "heldout_test_input_integrity_audit_v1.json"
OUT = ROOT / "07_Heldout_Test_Evaluation_v1"
CACHE = OUT / "checkpoint_results"
FINAL_PER_IMAGE = OUT / "heldout_test_12models_lesion_per_image_v1.csv"
FINAL_PIXEL = OUT / "heldout_test_12models_pixel_summary_v1.csv"
FINAL_LESION = OUT / "heldout_test_12models_lesion_summary_v1.csv"
FINAL_META = OUT / "heldout_test_12models_metadata_v1.json"
FINAL_OUTPUTS = [FINAL_PER_IMAGE, FINAL_PIXEL, FINAL_LESION, FINAL_META]
EXPECTED_PLAN_SHA = "35dbf447bbabb1cec4bacebe5ade8c90d9d9fecf4a6e16f40db4ac1add635800"
EXPECTED_VAL_M6 = {
    "AP_Loss": 0.45712,
    "AP_Dice": 0.55684,
    "AP_IoU": 0.43479,
    "Tooth_Dice": 0.92030,
    "RCF_Dice": 0.81506,
    "AP_Negative_FP_Rate": 0.19535,
}
SETTINGS = [(0.10, 1), (0.50, 1), (0.10, 16), (0.50, 16)]
STRUCTURE = np.ones((3, 3), dtype=np.uint8)
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).reshape(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).reshape(3, 1, 1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    assert not path.exists(), f"Existing output must not be overwritten: {path}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(tmp, encoding="utf-8") as f:
        assert json.load(f) == data
    assert not path.exists()
    os.replace(tmp, path)


def atomic_csv(path, frame):
    assert not path.exists(), f"Existing output must not be overwritten: {path}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    verified = pd.read_csv(tmp)
    assert len(verified) == len(frame)
    assert list(verified.columns) == list(frame.columns)
    assert not path.exists()
    os.replace(tmp, path)

# ------------------------------------------------------------
# 1. VERIFY LOCK, INPUT AUDIT AND CHECKPOINT REGISTRY
# ------------------------------------------------------------
assert PLAN_PATH.is_file() and AUDIT_PATH.is_file()
assert sha256_file(PLAN_PATH) == EXPECTED_PLAN_SHA, "PRE-TEST LOCK CHANGED: STOP"
with open(PLAN_PATH, encoding="utf-8") as f:
    plan = json.load(f)
with open(AUDIT_PATH, encoding="utf-8") as f:
    audit = json.load(f)
assert plan["plan_status"] == "FROZEN_BEFORE_HELD_OUT_TEST"
assert plan["primary_analysis"]["comparison"] == "M6_minus_C2"
assert plan["primary_analysis"]["endpoint"] == "Lesion_F1"
assert plan["training_seeds"] == [42, 123, 2026]
assert set(plan["fixed_models"]) == {"M5", "M6", "C1", "C2"}
assert plan["lesion_evaluation"]["probability_threshold"] == 0.5
assert plan["lesion_evaluation"]["connectivity"] == 8
assert plan["lesion_evaluation"]["primary_IoU_threshold"] == 0.1
assert plan["lesion_evaluation"]["primary_min_predicted_area_pixels"] == 1
assert plan["lesion_evaluation"]["component_matching"] == "Hungarian_maximum_cardinality_then_IoU"
assert plan["test_split"]["expected_images"] == 743
assert len(plan["checkpoints"]) == 12
assert audit["status"] == "PASSED" and audit["frozen_plan_sha256"] == EXPECTED_PLAN_SHA
assert audit["test_predictions_generated"] is False
assert audit["test_metrics_computed"] is False
assert audit["patient_disjointness_verified"] is False

for file in FINAL_OUTPUTS:
    assert not file.exists(), f"Published Test results already exist: {file}"

TEST_CSV = ROOT / plan["test_split"]["relative_csv_path"]
VAL_CSV = SPLITS / "validation_v1.csv"
assert TEST_CSV.is_file() and VAL_CSV.is_file()
assert sha256_file(TEST_CSV) == audit["test_csv_sha256"], "Test CSV changed since audit."
test_df = pd.read_csv(TEST_CSV, dtype={"image_id": str}).reset_index(drop=True)
val_df = pd.read_csv(VAL_CSV, dtype={"image_id": str}).reset_index(drop=True)
assert len(test_df) == len(val_df) == 743
assert test_df["image_id"].is_unique and val_df["image_id"].is_unique
assert test_df["stratum"].value_counts().to_dict() == plan["test_split"]["expected_strata"]
assert not set(test_df["image_id"]) & set(val_df["image_id"])
assert set(test_df["image_id"]).isdisjoint(set(pd.read_csv(SPLITS / "train_v1.csv", dtype={"image_id": str})["image_id"]))

registry = {}
for r in plan["checkpoints"]:
    key = (r["model_id"], int(r["seed"]))
    assert key not in registry
    cpath = ROOT / r["checkpoint_relative_path"]
    cfgpath = ROOT / r["config_relative_path"]
    assert cpath.is_file() and cfgpath.is_file()
    assert sha256_file(cpath) == r["checkpoint_sha256"], f"Checkpoint changed: {key}"
    assert sha256_file(cfgpath) == r["config_sha256"], f"Config changed: {key}"
    registry[key] = r
assert len(registry) == 12

print("=" * 82)
print("FROZEN TEST PLAN, SPLITS AND ALL 12 CHECKPOINT HASHES: VERIFIED")
print("=" * 82)
print("GPU:", torch.cuda.get_device_name(0))
print("Test:", len(test_df), "images | Validation preflight:", len(val_df))
print("Plan SHA256:", EXPECTED_PLAN_SHA)

# ------------------------------------------------------------
# 2. DETERMINISTIC, ASPECT-PRESERVING PIL PREPROCESSING
# ------------------------------------------------------------
class FrozenPRADDataset(Dataset):
    def __init__(self, df, raw_root):
        self.df = df.reset_index(drop=True).copy()
        self.raw_root = Path(raw_root)
        assert "image_id" in self.df and "stratum" in self.df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        image_id = str(self.df.iloc[i]["image_id"])
        image_path = self.raw_root / "image" / (image_id + ".jpg")
        label_path = self.raw_root / "label" / (image_id + ".png")

        with Image.open(image_path) as source:
            img = source.convert("RGB")
        with Image.open(label_path) as source:
            if source.mode not in ("P", "L"):
                raise ValueError(f"Unexpected indexed-mask mode: {source.mode} for {image_id}")
            mask = source.copy()

        w, h = img.size
        assert (w, h) == mask.size and w > 0 and h > 0
        scale = 512.0 / max(w, h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        dx = (512 - nw) // 2
        dy = (512 - nh) // 2
        img_resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
        mask_resized = mask.resize((nw, nh), Image.Resampling.NEAREST)
        image_canvas = Image.new("RGB", (512, 512), (0, 0, 0))
        mask_canvas = Image.new(mask.mode, (512, 512), 0)
        image_canvas.paste(img_resized, (dx, dy))
        mask_canvas.paste(mask_resized, (dx, dy))

        image_arr = np.asarray(image_canvas, dtype=np.float32).copy() / 255.0
        indexed_mask = np.asarray(mask_canvas, dtype=np.uint8)
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
        image_tensor = (image_tensor - MEAN) / STD

        return {
            "image": image_tensor.contiguous(),
            "AP_mask": torch.from_numpy((indexed_mask == 9).astype(np.float32))[None],
            "Tooth_mask": torch.from_numpy((indexed_mask == 1).astype(np.float32))[None],
            "RCF_mask": torch.from_numpy((indexed_mask == 3).astype(np.float32))[None],
            "image_id": image_id,
            "group": str(self.df.iloc[i]["stratum"]),
        }

val_data = FrozenPRADDataset(val_df, RAW)
test_data = FrozenPRADDataset(test_df, RAW)
val_loader_frozen = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
test_loader_frozen = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
assert isinstance(val_loader_frozen.sampler, SequentialSampler)
assert isinstance(test_loader_frozen.sampler, SequentialSampler)
assert len(val_loader_frozen) == len(test_loader_frozen) == 186

# If original restored validation objects are still in this runtime,
# perform an extra exact-input equivalence check.
if "val_dataset" in globals() and len(val_dataset) == 743:
    for idx in (0, 1, 97, 371, 742):
        a, b = val_data[idx], val_dataset[idx]
        assert str(a["image_id"]) == str(b["image_id"])
        for key in ("image", "AP_mask", "Tooth_mask", "RCF_mask"):
            assert torch.allclose(a[key], b[key], atol=1e-6, rtol=0), (idx, key)
    print("Preprocessing input-equivalence to restored validation dataset: PASSED")

# ------------------------------------------------------------
# 3. LOAD CHECKPOINT EXACTLY AS FROZEN
# ------------------------------------------------------------
def load_frozen_model(entry):
    nheads = int(entry["output_channels"])
    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",
        encoder_weights=None,
        encoder_output_stride=16,
        in_channels=3,
        classes=nheads,
        activation=None,
    ).to(DEVICE)
    ckpt = torch.load(ROOT / entry["checkpoint_relative_path"], map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        state = next((ckpt[k] for k in ("model_state", "model_state_dict", "state_dict")
                      if k in ckpt and isinstance(ckpt[k], dict)), None)
        if state is None and len(ckpt) > 0 and all(torch.is_tensor(v) for v in ckpt.values()):
            state = ckpt
    else:
        state = None
    assert state is not None, "Unsupported checkpoint structure: stop without retuning."
    model.load_state_dict(state, strict=True)
    model.eval()
    del ckpt, state
    return model

# ------------------------------------------------------------
# 4. VALIDATION FIDELITY AUDIT, BEFORE TEST INFERENCE
# ------------------------------------------------------------
@torch.inference_mode()
def m6_validation_audit(model):
    ap_bce = 0.0
    ap_softdice = 0.0
    ap_dice, ap_iou, tooth_dice, rcf_dice = [], [], [], []
    neg_count, neg_fp = 0, 0
    gt_component_count = 0
    num_images = 0
    for batch in val_loader_frozen:
        x = batch["image"].to(DEVICE)
        a = batch["AP_mask"].to(DEVICE)
        with torch.amp.autocast("cuda"):
            output = model(x)
        ap_logits = output[:, 0:1].float()
        ap_bce += F.binary_cross_entropy_with_logits(ap_logits, a, reduction="none").mean((1,2,3)).sum().item()
        p = torch.sigmoid(ap_logits)
        soft_inter = (p * a).sum((1,2,3))
        soft_d = 1.0 - (2.0 * soft_inter + 1.0) / (p.sum((1,2,3)) + a.sum((1,2,3)) + 1.0)
        ap_softdice += soft_d[a.sum((1,2,3)) > 0].sum().item()
        preds = (output.float() >= 0).cpu().numpy()
        agt = batch["AP_mask"][:,0].numpy() >= 0.5
        tgt = batch["Tooth_mask"][:,0].numpy() >= 0.5
        rgt = batch["RCF_mask"][:,0].numpy() >= 0.5
        for i in range(len(batch["image_id"])):
            ap = preds[i,0]; tooth = preds[i,1]; rcf = preds[i,2]
            gt_component_count += int(ndi.label(agt[i], structure=STRUCTURE)[1])
            na = int(agt[i].sum()); npred = int(ap.sum())
            if na:
                inter = int(np.logical_and(agt[i], ap).sum())
                ap_dice.append(2*inter/(na+npred))
                ap_iou.append(inter/(na+npred-inter))
            else:
                neg_count += 1
                neg_fp += int(npred > 0)
            nt = int(tgt[i].sum())
            if nt:
                tooth_dice.append(2*np.logical_and(tgt[i], tooth).sum()/(nt + tooth.sum()))
            nr = int(rgt[i].sum())
            if nr:
                rcf_dice.append(2*np.logical_and(rgt[i], rcf).sum()/(nr + rcf.sum()))
            num_images += 1
    assert (num_images, len(ap_dice), neg_count, len(tooth_dice), len(rcf_dice), gt_component_count) == (743, 98, 645, 742, 287, 122)
    return {
        "AP_Loss": ap_bce/743 + ap_softdice/98,
        "AP_Dice": float(np.mean(ap_dice)),
        "AP_IoU": float(np.mean(ap_iou)),
        "Tooth_Dice": float(np.mean(tooth_dice)),
        "RCF_Dice": float(np.mean(rcf_dice)),
        "AP_Negative_FP_Rate": neg_fp/645,
    }

print("\n" + "=" * 82)
print("PRE-TEST VALIDATION PREPROCESSING AND M6 CHECKPOINT REPRODUCTION")
print("=" * 82)
m6_model = load_frozen_model(registry[("M6", 2026)])
preflight = m6_validation_audit(m6_model)
for name, expected in EXPECTED_VAL_M6.items():
    actual = preflight[name]
    difference = abs(actual - expected)
    print(f"{name:25s} observed={actual:.5f} expected={expected:.5f} delta={difference:.6f}")
    assert difference < 0.0002, f"Validation reproduction mismatch for {name}; STOP before Test inference."
del m6_model
gc.collect()
torch.cuda.empty_cache()
print("VALIDATION FIDELITY AUDIT: PASSED")

# ------------------------------------------------------------
# 5. FROZEN HUNGARIAN MATCHING; 8-CONNECTED COMPONENTS
# ------------------------------------------------------------
def lesion_counts(gt_mask, pred_mask, iou_threshold, minimum_area):
    gt_labels, n_gt = ndi.label(gt_mask, structure=STRUCTURE)
    pr_labels, n_pr_raw = ndi.label(pred_mask, structure=STRUCTURE)
    gt_areas = np.bincount(gt_labels.ravel(), minlength=n_gt + 1)
    pred_areas = np.bincount(pr_labels.ravel(), minlength=n_pr_raw + 1)
    kept = np.flatnonzero(pred_areas[1:] >= minimum_area) + 1
    n_pred = len(kept)
    if n_gt == 0:
        return 0, int(n_pred), 0, int(n_gt), int(n_pred)
    if n_pred == 0:
        return 0, 0, int(n_gt), int(n_gt), 0
    pair = gt_labels.astype(np.int64).ravel() * (n_pr_raw+1) + pr_labels.astype(np.int64).ravel()
    joint = np.bincount(pair, minlength=(n_gt+1)*(n_pr_raw+1)).reshape(n_gt+1, n_pr_raw+1)
    inter = joint[1:, kept].astype(np.float64)
    union = gt_areas[1:,None] + pred_areas[kept][None,:] - inter
    ious = np.divide(inter, union, out=np.zeros_like(inter), where=union>0)
    valid = ious >= iou_threshold
    rewards = np.zeros((n_gt, n_pred + n_gt), dtype=np.float64)
    rewards[:,:n_pred] = np.where(valid, (n_gt+1.0)+ious, -1.0)
    gt_indices, pred_indices = linear_sum_assignment(-rewards)
    tp = int(sum(pred_idx < n_pred and valid[gt_idx,pred_idx]
                 for gt_idx,pred_idx in zip(gt_indices,pred_indices)))
    return tp, int(n_pred-tp), int(n_gt-tp), int(n_gt), int(n_pred)


def binary_dice(gt, pred):
    return float(2 * np.logical_and(gt,pred).sum()/(int(gt.sum()) + int(pred.sum())))


def binary_iou(gt, pred):
    inter = int(np.logical_and(gt,pred).sum())
    return float(inter/(int(gt.sum())+int(pred.sum())-inter))

# ------------------------------------------------------------
# 6. ONE CHECKPOINT AT A TIME — ATOMIC IMMUTABLE CACHE
# ------------------------------------------------------------
@torch.inference_mode()
def evaluate_checkpoint(entry):
    model = load_frozen_model(entry)
    key_model, key_seed = entry["model_id"], int(entry["seed"])
    records = []
    ap_dices, ap_ious, tooth_dices, rcf_dices = [], [], [], []
    neg_images, neg_fp_images, image_count = 0, 0, 0
    gt_components_total = 0
    start = time.time()

    for batch_id, batch in enumerate(test_loader_frozen, start=1):
        images = batch["image"].to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
        assert outputs.shape[1] == int(entry["output_channels"])
        preds = (outputs.float() >= 0).cpu().numpy()
        agt = batch["AP_mask"][:,0].numpy() >= 0.5
        tgt = batch["Tooth_mask"][:,0].numpy() >= 0.5
        rgt = batch["RCF_mask"][:,0].numpy() >= 0.5
        for i, image_id in enumerate(batch["image_id"]):
            image_id = str(image_id)
            group = str(batch["group"][i])
            gt = agt[i]; ap = preds[i,0]
            n_gt_pixels, n_pred_pixels = int(gt.sum()), int(ap.sum())
            if n_gt_pixels > 0:
                ap_dices.append(binary_dice(gt, ap))
                ap_ious.append(binary_iou(gt, ap))
            else:
                neg_images += 1
                neg_fp_images += int(n_pred_pixels > 0)
            if tgt[i].any():
                tooth_dices.append(binary_dice(tgt[i], preds[i,1]))
            if rgt[i].any() and preds.shape[1] == 3:
                rcf_dices.append(binary_dice(rgt[i], preds[i,2]))
            image_gt_comp = None
            for iou, area in SETTINGS:
                tp, fp, fn, ngt, nprd = lesion_counts(gt, ap, iou, area)
                if image_gt_comp is None:
                    image_gt_comp = ngt
                else:
                    assert image_gt_comp == ngt
                records.append({
                    "model": key_model, "seed": key_seed, "image_id": image_id,
                    "group": group, "IoU_threshold": iou, "min_pred_area": area,
                    "GT_components": ngt, "Pred_components": nprd,
                    "TP": tp, "FP": fp, "FN": fn,
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                    "validation_protocol_sha256": plan["lesion_evaluation"]["validation_protocol_sha256"],
                    "pretest_plan_sha256": EXPECTED_PLAN_SHA,
                })
            gt_components_total += image_gt_comp
            image_count += 1
        if image_count % 120 < len(batch["image_id"]) or image_count == 743:
            print(f"  {key_model} Seed{key_seed}: {image_count}/743 images", flush=True)

    assert image_count == 743 and len(records) == 743 * 4
    assert neg_images + len(ap_dices) == 743
    assert len(tooth_dices) <= 743
    assert neg_images > 0 and len(ap_dices) > 0
    pixel = {
        "model": key_model, "seed": key_seed, "images": image_count,
        "AP_positive_images": len(ap_dices), "AP_negative_images": neg_images,
        "Tooth_positive_images": len(tooth_dices),
        "RCF_positive_images_evaluated": len(rcf_dices),
        "AP_Dice": float(np.mean(ap_dices)),
        "AP_IoU": float(np.mean(ap_ious)),
        "Tooth_Dice": float(np.mean(tooth_dices)),
        "RCF_Dice": (float(np.mean(rcf_dices)) if len(rcf_dices) else None),
        "AP_Negative_FP_Image_Rate": neg_fp_images/neg_images,
        "GT_AP_components": int(gt_components_total),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {key_model} Seed{key_seed}: completed in {(time.time()-start)/60:.1f} min | GT components={gt_components_total}")
    return records, pixel

OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

all_records, all_pixels = [], []
for seed in (42, 123, 2026):
    for model_name in ("M5", "M6", "C1", "C2"):
        entry = registry[(model_name, seed)]
        cache_path = CACHE / f"{model_name}_Seed{seed}_test_frozen_v1.json"
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                saved = json.load(f)
            assert saved["pretest_plan_sha256"] == EXPECTED_PLAN_SHA
            assert saved["test_csv_sha256"] == audit["test_csv_sha256"]
            assert saved["checkpoint_sha256"] == entry["checkpoint_sha256"]
            assert saved["model"] == model_name and saved["seed"] == seed
            assert len(saved["per_image_records"]) == 743 * 4
            assert saved["pixel_summary"]["images"] == 743
            print(f"REUSING VERIFIED IMMUTABLE CACHE: {model_name} Seed{seed}")
        else:
            print("\n" + "-"*80)
            print(f"TEST INFERENCE: {model_name} SEED {seed}")
            print("-"*80)
            records, pixel = evaluate_checkpoint(entry)
            saved = {
                "model": model_name, "seed": seed,
                "pretest_plan_sha256": EXPECTED_PLAN_SHA,
                "test_csv_sha256": audit["test_csv_sha256"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "per_image_records": records, "pixel_summary": pixel,
            }
            atomic_json(cache_path, saved)
            print("  ATOMIC CACHE SAVED:", cache_path.name)
        all_records.extend(saved["per_image_records"])
        all_pixels.append(saved["pixel_summary"])

# ------------------------------------------------------------
# 7. FINAL CROSS-MODEL, CROSS-SEED GROUND-TRUTH AUDIT
# ------------------------------------------------------------
per_image = pd.DataFrame(all_records)
pixel_summary = pd.DataFrame(all_pixels)
assert len(per_image) == 12 * 743 * 4
assert len(pixel_summary) == 12
assert not per_image.duplicated(["model", "seed", "image_id", "IoU_threshold", "min_pred_area"]).any()
assert (per_image["TP"]+per_image["FN"] == per_image["GT_components"]).all()
assert (per_image["TP"]+per_image["FP"] == per_image["Pred_components"]).all()
assert per_image.groupby("image_id")["group"].nunique().eq(1).all()
assert per_image.groupby(["image_id", "IoU_threshold", "min_pred_area"])["GT_components"].nunique().eq(1).all()
for seed in (42, 123, 2026):
    for name in ("M5", "M6", "C1", "C2"):
        frame = per_image[(per_image["seed"]==seed) & (per_image["model"]==name)]
        assert len(frame) == 743*4
        for iou, area in SETTINGS:
            one = frame[np.isclose(frame["IoU_threshold"], iou) & (frame["min_pred_area"]==area)]
            assert len(one)==743 and one["image_id"].is_unique
            assert set(one["image_id"]) == set(test_df["image_id"])
            assert one["GT_components"].sum() == pixel_summary.loc[(pixel_summary["model"]==name)&(pixel_summary["seed"]==seed),"GT_AP_components"].iloc[0]

# ------------------------------------------------------------
# 8. FROZEN LESION METRICS, ALL 4 SETTINGS
# ------------------------------------------------------------
lesion_rows = []
for seed in (42, 123, 2026):
    for name in ("M5", "M6", "C1", "C2"):
        for iou, area in SETTINGS:
            data = per_image[
                (per_image["seed"]==seed) & (per_image["model"]==name) &
                np.isclose(per_image["IoU_threshold"],iou) & (per_image["min_pred_area"]==area)
            ]
            gt, tp, fp, fn = (int(data[col].sum()) for col in ("GT_components","TP","FP","FN"))
            assert tp+fn == gt
            negative = data[data["GT_components"]==0]
            lesion_rows.append({
                "model": name, "seed": seed, "IoU_threshold": iou, "min_pred_area": area,
                "Images": 743, "GT": gt, "TP": tp, "FP": fp, "FN": fn,
                "Lesion_Recall": tp/gt if gt else None,
                "Lesion_Precision": tp/(tp+fp) if tp+fp else 0.0,
                "Lesion_F1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                "FP_Components_per_Image": fp/743,
                "AP_Negative_FP_Image_Rate": float(negative["FP"].gt(0).mean()),
                "AP_negative_images": len(negative),
            })
lesion_summary = pd.DataFrame(lesion_rows)
assert len(lesion_summary) == 48
assert lesion_summary["GT"].nunique() == 1, "All models must use identical Test GT."

print("\n" + "=" * 110)
print("FROZEN HELD-OUT TEST — PRIMARY LESION RESULTS (ALL 12 CHECKPOINTS)")
print("=" * 110)
primary = lesion_summary[np.isclose(lesion_summary["IoU_threshold"],0.10) & (lesion_summary["min_pred_area"]==1)]
print(primary[["seed","model","GT","TP","FP","FN","Lesion_Recall","Lesion_F1","FP_Components_per_Image","AP_Negative_FP_Image_Rate"]].round(4).to_string(index=False))
print("\nPIXEL METRICS — ALL 12 CHECKPOINTS")
print(pixel_summary[["seed","model","AP_Dice","AP_IoU","Tooth_Dice","RCF_Dice","AP_Negative_FP_Image_Rate"]].round(4).to_string(index=False))

# ------------------------------------------------------------
# 9. PUBLISH FROZEN TEST RESULTS (NO OVERWRITE)
# ------------------------------------------------------------
for path in FINAL_OUTPUTS:
    assert not path.exists(), f"Final output exists: {path}"
atomic_csv(FINAL_PER_IMAGE, per_image)
atomic_csv(FINAL_PIXEL, pixel_summary)
atomic_csv(FINAL_LESION, lesion_summary)
metadata = {
    "status": "PASSED", "version": "v1",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "frozen_pretest_plan_sha256": EXPECTED_PLAN_SHA,
    "test_input_audit_sha256": sha256_file(AUDIT_PATH),
    "test_split_sha256": audit["test_csv_sha256"],
    "validation_protocol_sha256": plan["lesion_evaluation"]["validation_protocol_sha256"],
    "checkpoints": [{"model": r["model_id"], "seed": r["seed"], "sha256": r["checkpoint_sha256"]}
                    for r in plan["checkpoints"]],
    "test_images": 743, "ground_truth_AP_components": int(lesion_summary["GT"].iloc[0]),
    "primary_comparison": "M6_minus_C2", "primary_endpoint": "Lesion_F1",
    "probability_threshold": 0.50, "primary_IoU": 0.10, "primary_min_pred_area": 1,
    "sensitivity": [{"IoU_threshold": iou, "min_pred_area": area} for iou,area in SETTINGS],
    "image_bootstrap_performed": False, "training_performed": False,
    "test_based_model_selection": False, "test_based_threshold_tuning": False,
    "patient_disjointness_verified": False,
    "output_rows": {"per_image": len(per_image), "pixel": len(pixel_summary), "lesion": len(lesion_summary)},
}
atomic_json(FINAL_META, metadata)
assert sha256_file(PLAN_PATH) == EXPECTED_PLAN_SHA
assert all(path.is_file() for path in FINAL_OUTPUTS)
print("\n" + "=" * 95)
print("HELD-OUT TEST INFERENCE AND LESION EVALUATION COMPLETED")
print("=" * 95)
print("Frozen checkpoints evaluated: 12/12")
print("Test images per checkpoint: 743")
print("Test GT components:", metadata["ground_truth_AP_components"])
print("Per-image records:", len(per_image))
print("Lesion summary rows:", len(lesion_summary))
print("Pixel summary rows:", len(pixel_summary))
print("Published Test outputs:")
for path in FINAL_OUTPUTS:
    print(" ", path)
print("\nOriginal checkpoints modified: NO")
print("Frozen pre-test plan modified: NO")
print("Threshold or model retuning: NONE")
print("Bootstrap: NEXT STEP")
print("Patient-level independence established: NO")
print("\nFINAL STATUS: PASSED")
