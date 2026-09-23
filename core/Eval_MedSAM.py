import argparse
import json
import sys
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion

sys.path.insert(0, "/gpfs/scratch/ec25141/medsam/MedSAM")
from segment_anything import sam_model_registry

import aorta_data
from paths import LOCAL_RESULTS_DIR

RESULTS_DIR = str(LOCAL_RESULTS_DIR)
CHECKPOINT = "/gpfs/scratch/ec25141/medsam/MedSAM/checkpoints/medsam_vit_b.pth"
IMG_SIZE = 1024
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


def mask_to_box(mask, margin=BOX_MARGIN, img_shape=None):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    if img_shape is not None:
        y_max = min(img_shape[0] - 1, y_max + margin)
        x_max = min(img_shape[1] - 1, x_max + margin)
    else:
        y_max += margin
        x_max += margin
    return np.array([x_min, y_min, x_max, y_max])


@torch.no_grad()
def medsam_infer_slice(medsam_model, slice_2d, box, device):
    H, W = slice_2d.shape
    img_3c = np.repeat(slice_2d[:, :, None], 3, axis=2)
    img_3c = (img_3c * 255).astype(np.uint8)

    import cv2
    img_resized = cv2.resize(img_3c, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    img_tensor = torch.tensor(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    box_scaled = box / np.array([W, H, W, H]) * IMG_SIZE
    box_tensor = torch.as_tensor(box_scaled, dtype=torch.float, device=device).unsqueeze(0)

    image_embedding = medsam_model.image_encoder(img_tensor)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_tensor, masks=None
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = torch.nn.functional.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False
    )
    mask = (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return mask


def segment_volume_from_middle(medsam_model, ct_norm, gt_bin, device):
    D, H, W = ct_norm.shape
    mid_z = D // 2
    z_with_fg = np.where(gt_bin.any(axis=(1, 2)))[0]
    if len(z_with_fg) == 0:
        return np.zeros_like(gt_bin)
    if mid_z not in z_with_fg:
        mid_z = int(z_with_fg[len(z_with_fg) // 2])

    seg_3d = np.zeros_like(gt_bin, dtype=np.uint8)

    seed_box = mask_to_box(gt_bin[mid_z], img_shape=(H, W))
    if seed_box is None:
        return seg_3d

    current_box = seed_box
    mask = medsam_infer_slice(medsam_model, ct_norm[mid_z], current_box, device)
    seg_3d[mid_z] = mask

    current_box = mask_to_box(mask, img_shape=(H, W))
    for z in range(mid_z + 1, D):
        if current_box is None:
            break
        mask = medsam_infer_slice(medsam_model, ct_norm[z], current_box, device)
        seg_3d[z] = mask
        current_box = mask_to_box(mask, img_shape=(H, W))

    current_box = mask_to_box(seg_3d[mid_z], img_shape=(H, W))
    for z in range(mid_z - 1, -1, -1):
        if current_box is None:
            break
        mask = medsam_infer_slice(medsam_model, ct_norm[z], current_box, device)
        seg_3d[z] = mask
        current_box = mask_to_box(mask, img_shape=(H, W))

    return seg_3d


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
    parser.add_argument("--tag", default="medsam")
    parser.add_argument("--split_path", default=f"{RESULTS_DIR}/patient_split_3d.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    splits = read_fixed_split(args.split_path)
    test_patients = splits["test"]
    print(f"evaluating MedSAM baseline on {len(test_patients)} test patients", flush=True)

    records = []
    for i, patient in enumerate(test_patients, 1):
        try:
            ct_norm, gt_bin = load_patient_volume(patient)
            seg_3d = segment_volume_from_middle(medsam_model, ct_norm, gt_bin, device)
            d = dice_score(seg_3d, gt_bin)
            iou = iou_score(seg_3d, gt_bin)
            hd95 = hausdorff_distance_95(seg_3d, gt_bin)
            records.append({"patient": patient, "dice": d, "iou": iou, "hd95": hd95})
            print(f"{i}/{len(test_patients)} {patient}: dice={d:.3f} iou={iou:.3f} hd95={hd95:.2f}", flush=True)
        except Exception as e:
            print(f"{patient}: FAILED - {e}")

    raw_out = f"{RESULTS_DIR}/medsam_raw_{args.tag}.json"
    with open(raw_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"saved -> {raw_out}")

    valid = [r for r in records if not np.isnan(r["dice"])]
    mean_dice = float(np.mean([r["dice"] for r in valid])) if valid else float("nan")
    mean_iou = float(np.mean([r["iou"] for r in valid])) if valid else float("nan")
    mean_hd95 = float(np.mean([r["hd95"] for r in valid if not np.isnan(r["hd95"])])) if valid else float("nan")

    print(f"\n=== MedSAM baseline summary ===")
    print(f"mean_dice={mean_dice:.4f} mean_iou={mean_iou:.4f} mean_hd95={mean_hd95:.2f} n={len(valid)}/{len(records)}")


if __name__ == "__main__":
    main()