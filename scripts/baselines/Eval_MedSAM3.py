import argparse
import json
import sys
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion

MEDSAM2_DIR = "/gpfs/scratch/ec25141/medsam/MedSAM2"
sys.path.insert(0, MEDSAM2_DIR)
from sam2.build_sam import build_sam2_video_predictor_npz

from pathlib import Path

# Locate shared utilities when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

import aorta_data
from paths import LOCAL_RESULTS_DIR

RESULTS_DIR = str(LOCAL_RESULTS_DIR)
MODEL_CFG = "configs/sam2.1_hiera_t512.yaml"
CHECKPOINT = f"{MEDSAM2_DIR}/checkpoints/MedSAM2_latest.pt"
BOX_MARGIN = 5


def scale_intensity(vol, a_min=-175, a_max=250):
    return ((np.clip(vol, a_min, a_max) - a_min) / (a_max - a_min)).astype(np.float32)


def load_patient_volume(patient):
    ct, gt = aorta_data.load_patient_any(patient)
    ct_norm = scale_intensity(ct)
    gt_bin = (gt > 0.5).astype(np.float32)
    return ct_norm, gt_bin


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


def mask_to_box(mask, margin=BOX_MARGIN):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y_min, y_max = max(0, ys.min() - margin), ys.max() + margin
    x_min, x_max = max(0, xs.min() - margin), xs.max() + margin
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


MODEL_IMG_SIZE = 512


def volume_to_video_frames(ct_norm, target_size=MODEL_IMG_SIZE):
    import cv2
    D, H, W = ct_norm.shape
    img_u8 = (ct_norm * 255).astype(np.uint8)
    frames = np.zeros((D, target_size, target_size, 3), dtype=np.uint8)
    for z in range(D):
        resized = cv2.resize(img_u8[z], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        frames[z, :, :, 0] = resized
        frames[z, :, :, 1] = resized
        frames[z, :, :, 2] = resized
    return frames


def segment_volume_medsam2_3d(predictor, ct_norm, gt_bin, device):
    D, H, W = ct_norm.shape
    z_with_fg = np.where(gt_bin.any(axis=(1, 2)))[0]
    if len(z_with_fg) == 0:
        return np.zeros_like(gt_bin, dtype=np.uint8)
    mid_z = int(z_with_fg[len(z_with_fg) // 2])

    seed_box = mask_to_box(gt_bin[mid_z])
    if seed_box is None:
        return np.zeros_like(gt_bin, dtype=np.uint8)

    scale_x = MODEL_IMG_SIZE / W
    scale_y = MODEL_IMG_SIZE / H
    seed_box_scaled = np.array([
        seed_box[0] * scale_x, seed_box[1] * scale_y,
        seed_box[2] * scale_x, seed_box[3] * scale_y,
    ], dtype=np.float32)

    frames = volume_to_video_frames(ct_norm)
    segs_3d = np.zeros((D, H, W), dtype=np.uint8)

    frames_tensor = torch.from_numpy(frames).float() / 255.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)

    inference_state = predictor.init_state(images=frames_tensor,
                                            video_height=MODEL_IMG_SIZE, video_width=MODEL_IMG_SIZE)
    predictor.reset_state(inference_state)

    def store_mask(frame_idx, mask_logits):
        import cv2
        mask_small = (mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8)
        mask_full = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
        segs_3d[frame_idx][mask_full > 0] = 1

    _, _, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=mid_z,
        obj_id=1,
        box=seed_box_scaled,
    )

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        store_mask(out_frame_idx, out_mask_logits)
    predictor.reset_state(inference_state)

    _, _, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=mid_z,
        obj_id=1,
        box=seed_box_scaled,
    )
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        store_mask(out_frame_idx, out_mask_logits)
    predictor.reset_state(inference_state)

    return segs_3d


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--model_cfg", default=MODEL_CFG)
    parser.add_argument("--tag", default="medsam2_3d")
    parser.add_argument("--split_path", default=f"{RESULTS_DIR}/patient_split_3d.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor_npz(args.model_cfg, args.checkpoint, device=device)

    splits = read_fixed_split(args.split_path)
    test_patients = splits["test"]
    print(f"evaluating MedSAM2-3D baseline on {len(test_patients)} test patients", flush=True)

    records = []
    for i, patient in enumerate(test_patients, 1):
        try:
            ct_norm, gt_bin = load_patient_volume(patient)
            seg_3d = segment_volume_medsam2_3d(predictor, ct_norm, gt_bin, device)
            d = dice_score(seg_3d, gt_bin)
            iou = iou_score(seg_3d, gt_bin)
            hd95 = hausdorff_distance_95(seg_3d, gt_bin)
            records.append({"patient": patient, "dice": d, "iou": iou, "hd95": hd95})
            print(f"{i}/{len(test_patients)} {patient}: dice={d:.3f} iou={iou:.3f} hd95={hd95:.2f}", flush=True)
        except Exception as e:
            print(f"{patient}: FAILED - {e}")

    raw_out = f"{RESULTS_DIR}/medsam2_3d_raw_{args.tag}.json"
    with open(raw_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"saved -> {raw_out}")

    valid = [r for r in records if not np.isnan(r["dice"])]
    mean_dice = float(np.mean([r["dice"] for r in valid])) if valid else float("nan")
    mean_iou = float(np.mean([r["iou"] for r in valid])) if valid else float("nan")
    mean_hd95 = float(np.mean([r["hd95"] for r in valid if not np.isnan(r["hd95"])])) if valid else float("nan")

    print(f"\n=== MedSAM2-3D baseline summary ===")
    print(f"mean_dice={mean_dice:.4f} mean_iou={mean_iou:.4f} mean_hd95={mean_hd95:.2f} n={len(valid)}/{len(records)}")


if __name__ == "__main__":
    main()