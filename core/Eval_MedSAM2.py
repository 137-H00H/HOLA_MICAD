import argparse
import json
import sys
import time
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion

MEDSAM2_DIR = "/gpfs/scratch/ec25141/medsam/MedSAM2"
sys.path.insert(0, MEDSAM2_DIR)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import aorta_data
import Clicksim as click_sim
from paths import LOCAL_RESULTS_DIR

RESULTS_DIR = str(LOCAL_RESULTS_DIR)
MODEL_CFG = "configs/sam2.1_hiera_t512.yaml"
CHECKPOINT = f"{MEDSAM2_DIR}/checkpoints/MedSAM2_latest.pt"
N_CLICKS = 10
SEED = 0


def load_medsam2(model_cfg=MODEL_CFG, checkpoint=CHECKPOINT):
    import os
    cwd = os.getcwd()
    os.chdir(MEDSAM2_DIR)
    try:
        model = build_sam2(model_cfg, checkpoint)
    finally:
        os.chdir(cwd)
    predictor = SAM2ImagePredictor(model)
    return predictor


def scale_intensity(vol, a_min=-175, a_max=250):
    return ((np.clip(vol, a_min, a_max) - a_min) / (a_max - a_min)).astype(np.float32)


def load_patient_volume(patient):
    ct, gt = aorta_data.load_patient_any(patient)
    ct_norm = scale_intensity(ct)
    gt_bin = (gt > 0.5).astype(np.float32)
    return ct_norm, gt_bin


def ct_slice_to_rgb(ct_slice_norm):
    """ct_slice_norm is already scaled to [0,1] via scale_intensity."""
    ct_uint8 = (ct_slice_norm * 255).astype(np.uint8)
    return np.stack([ct_uint8, ct_uint8, ct_uint8], axis=-1)


def medsam2_predict_slice(predictor, ct_rgb, pos_clicks, neg_clicks):
    predictor.set_image(ct_rgb)

    all_points, all_labels = [], []
    for (r, c) in pos_clicks:
        all_points.append([c, r])
        all_labels.append(1)
    for (r, c) in neg_clicks:
        all_points.append([c, r])
        all_labels.append(0)

    points = np.array(all_points, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)

    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
        )
    return masks[0].astype(np.float32)


def dice_score(pred, gt):
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (pred * gt).sum() / denom)


def iou_score(pred, gt):
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union == 0:
        return float("nan")
    return float(intersection / union)


def hausdorff_distance_95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    pred = (pred > 0.5)
    gt = (gt > 0.5)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    pred_surface = pred & ~binary_erosion(pred)
    gt_surface = gt & ~binary_erosion(gt)
    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    if len(all_d) == 0:
        return float("nan")
    return float(np.percentile(all_d, 95))


def read_fixed_split(path):
    splits = {"train": [], "val": [], "test": []}
    section = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "# train":
                section = "train"
            elif line == "# val":
                section = "val"
            elif line == "# test":
                section = "test"
            elif line.startswith("#"):
                section = None
            elif section and line:
                splits[section].append(line)
    return splits


def evaluate_one_patient(predictor, patient, n_clicks, rng):
    """
    MedSAM2 (non-3D) is a 2D, per-slice model with no native volume
    understanding -- each slice is segmented independently, driven by
    point clicks, matching your own system's click interaction style.
    To compare fairly against your own FULL-VOLUME dice, per-slice
    predictions are assembled into one 3D volume before scoring,
    rather than averaging per-slice dice values separately.
    """
    ct_norm, gt_bin = load_patient_volume(patient)
    shape = ct_norm.shape
    mid_z = shape[0] // 2
    seed_2d = click_sim.initial_seed_click(gt_bin[mid_z], rng=rng)
    pos = [(r, c) for r, c in seed_2d]
    neg = []

    pred_volume = np.zeros_like(gt_bin)
    ct_rgb = ct_slice_to_rgb(ct_norm[mid_z])
    pred_2d = medsam2_predict_slice(predictor, ct_rgb, pos, neg)
    pred_volume[mid_z] = pred_2d

    results = {}
    d = dice_score(pred_volume, gt_bin)
    results[0] = {"dice": d}

    for click_num in range(1, n_clicks + 1):
        click, ctype = click_sim.simulate_next_click(pred_2d, gt_bin[mid_z], rng=rng)
        if click is None:
            results[click_num] = results[click_num - 1]
            continue
        (pos if ctype == "positive" else neg).append(click)
        pred_2d = medsam2_predict_slice(predictor, ct_rgb, pos, neg)
        pred_volume[mid_z] = pred_2d
        d = dice_score(pred_volume, gt_bin)
        results[click_num] = {"dice": d}

    final_pred_volume = pred_volume
    final_dice = dice_score(final_pred_volume, gt_bin)
    final_iou = iou_score(final_pred_volume, gt_bin)
    final_hd95 = hausdorff_distance_95(final_pred_volume, gt_bin)

    return results, final_dice, final_iou, final_hd95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", default=MODEL_CFG)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--tag", default="medsam2_2d")
    parser.add_argument("--split_path", default=f"{RESULTS_DIR}/patient_split_3d.txt")
    parser.add_argument("--n_clicks", type=int, default=N_CLICKS)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    predictor = load_medsam2(args.model_cfg, args.checkpoint)

    splits = read_fixed_split(args.split_path)
    test_patients = splits["test"]
    print(f"evaluating MedSAM2 (2D, point-click) baseline on {len(test_patients)} test patients", flush=True)

    records = []
    for i, patient in enumerate(test_patients, 1):
        try:
            click_results, final_dice, final_iou, final_hd95 = evaluate_one_patient(
                predictor, patient, args.n_clicks, rng)
            records.append({
                "patient": patient,
                "click_dice": {str(k): v["dice"] for k, v in click_results.items()},
                "final_dice": final_dice, "final_iou": final_iou, "final_hd95": final_hd95,
            })
            print(f"{i}/{len(test_patients)} {patient}: "
                  f"dice@0={click_results[0]['dice']:.3f} final_dice={final_dice:.3f}", flush=True)
        except Exception as e:
            print(f"{patient}: FAILED - {e}")

    raw_out = f"{RESULTS_DIR}/medsam2_2d_raw_{args.tag}.json"
    with open(raw_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"saved -> {raw_out}")

    valid = [r for r in records if not np.isnan(r["final_dice"])]
    mean_dice = float(np.mean([r["final_dice"] for r in valid])) if valid else float("nan")
    mean_iou = float(np.mean([r["final_iou"] for r in valid])) if valid else float("nan")
    mean_hd95 = float(np.mean([r["final_hd95"] for r in valid if not np.isnan(r["final_hd95"])])) if valid else float("nan")

    print(f"\n=== MedSAM2 (2D, point-click) baseline summary ===")
    print(f"mean_dice={mean_dice:.4f} mean_iou={mean_iou:.4f} mean_hd95={mean_hd95:.2f} n={len(valid)}/{len(records)}")


if __name__ == "__main__":
    main()