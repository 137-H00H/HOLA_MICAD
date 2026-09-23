import argparse
import os
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label as connected_components, zoom
from monai.transforms import SpatialPad
from monai.networks.nets import DynUNet

import aorta_data
import Clicksim as click_sim
from paths import LOCAL_RESULTS_DIR

RESULTS_DIR = str(LOCAL_RESULTS_DIR)
SAMPLE_OUT_DIR = f"{RESULTS_DIR}/completeness_samples"
SEED = 0
CLICK_COUNTS = [0, 1, 3, 5, 8, 10]
PATCH_SIZE = (64, 128, 128)
OVERLAP_RATIO = 0.5
WINDOW_BATCH_SIZE = 16
SIGMA = 3
MIN_COMPONENT_VOXELS = 500

CROP_PADDING_VOXELS = 20
CNN_CROP_SIZE = (96, 96, 96)  


def scale_intensity(vol, a_min=-175, a_max=250):
    return ((np.clip(vol, a_min, a_max) - a_min) / (a_max - a_min)).astype(np.float32)


def load_patient_volume(patient):
    ct, gt = aorta_data.load_patient_any(patient)
    ct_norm = scale_intensity(ct)
    gt_bin = (gt > 0.5).astype(np.float32)
    return ct_norm, gt_bin


def make_click_channel_3d(shape, clicks, sigma=SIGMA):
    signal = np.zeros(shape, dtype=np.float32)
    for (z, y, x) in clicks:
        z = max(0, min(int(z), shape[0] - 1))
        y = max(0, min(int(y), shape[1] - 1))
        x = max(0, min(int(x), shape[2] - 1))
        signal[z, y, x] = 1.0
    if signal.max() > 0:
        signal = gaussian_filter(signal, sigma=sigma)
        signal = (signal - signal.min()) / (signal.max() - signal.min())
    return signal


def build_input_3d(ct_norm, pos_clicks, neg_clicks, shape):
    pos_ch = make_click_channel_3d(shape, pos_clicks)
    neg_ch = make_click_channel_3d(shape, neg_clicks)
    return np.stack([ct_norm, pos_ch, neg_ch], axis=0)


def build_dynunet_3d(device):
    kernels = [[3, 3, 3]] * 6
    strides = [[1, 1, 1]] + [[2, 2, 2]] * 5
    net = DynUNet(spatial_dims=3, in_channels=3, out_channels=2, kernel_size=kernels,
                  strides=strides, upsample_kernel_size=strides[1:],
                  norm_name="instance", deep_supervision=False, res_block=True)
    return net.to(device)


def load_checkpoint(model_path, device):
    net = build_dynunet_3d(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()
    return net


def dice_score(pred, gt):
    pred, gt = (pred > 0.5).astype(np.float32), (gt > 0.5).astype(np.float32)
    denom = pred.sum() + gt.sum()
    return float(2 * (pred * gt).sum() / denom) if denom > 0 else float("nan")


def filter_small_components(pred, min_voxels=MIN_COMPONENT_VOXELS):
    if pred.sum() == 0:
        return pred
    labeled, n = connected_components(pred > 0.5)
    if n <= 1:
        return pred
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == int(np.argmax(sizes))).astype(pred.dtype)


def predict_volume(net, ct_norm, pos_clicks, neg_clicks, shape, device,
                   patch_dhw=PATCH_SIZE, overlap=OVERLAP_RATIO):
    D0, H0, W0 = shape
    pd, ph, pw = patch_dhw
    pad_d, pad_h, pad_w = max(0, pd - D0), max(0, ph - H0), max(0, pw - W0)
    if pad_d or pad_h or pad_w:
        padder = SpatialPad(spatial_size=(max(pd, D0), max(ph, H0), max(pw, W0)), mode="constant")
        ct_norm = np.asarray(padder(ct_norm[None]))[0]
    D, H, W = ct_norm.shape
    sd, sh, sw = max(1, int(pd * (1 - overlap))), max(1, int(ph * (1 - overlap))), max(1, int(pw * (1 - overlap)))
    pred_acc, weight_acc = np.zeros((D, H, W), dtype=np.float32), np.zeros((D, H, W), dtype=np.float32)
    gz = np.exp(-((np.linspace(-1, 1, pd)) ** 2) / 0.5)
    gy = np.exp(-((np.linspace(-1, 1, ph)) ** 2) / 0.5)
    gx = np.exp(-((np.linspace(-1, 1, pw)) ** 2) / 0.5)
    weight_map = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    windows = []
    for z in range(0, max(1, D - pd + 1), sd):
        z = min(z, D - pd)
        for y in range(0, max(1, H - ph + 1), sh):
            y = min(y, H - ph)
            for x in range(0, max(1, W - pw + 1), sw):
                x = min(x, W - pw)
                ct_patch = ct_norm[z:z+pd, y:y+ph, x:x+pw]
                pos_patch = [(cz-z, cy-y, cx-x) for (cz, cy, cx) in pos_clicks if z <= cz < z+pd and y <= cy < y+ph and x <= cx < x+pw]
                neg_patch = [(cz-z, cy-y, cx-x) for (cz, cy, cx) in neg_clicks if z <= cz < z+pd and y <= cy < y+ph and x <= cx < x+pw]
                windows.append(((z, y, x), build_input_3d(ct_patch, pos_patch, neg_patch, ct_patch.shape)))
    net.eval()
    with torch.no_grad():
        for start in range(0, len(windows), WINDOW_BATCH_SIZE):
            batch = windows[start:start + WINDOW_BATCH_SIZE]
            t = torch.from_numpy(np.stack([w[1] for w in batch])).float().to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = net(t)
                probs = torch.softmax(out, dim=1)[:, 1].float().cpu().numpy()
            for i, ((wz, wy, wx), _) in enumerate(batch):
                pred_acc[wz:wz+pd, wy:wy+ph, wx:wx+pw] += probs[i] * weight_map
                weight_acc[wz:wz+pd, wy:wy+ph, wx:wx+pw] += weight_map
    weight_acc[weight_acc == 0] = 1.0
    final_pred = (pred_acc / weight_acc)[:D0, :H0, :W0]
    return filter_small_components((final_pred > 0.5).astype(np.float32))


def crop_with_context(ct_norm, pred, padding=CROP_PADDING_VOXELS, target_size=CNN_CROP_SIZE):
    D, H, W = ct_norm.shape
    fg = pred > 0.5

    if fg.sum() > 0:
        nz = np.where(fg)
        mins = [max(0, nz[i].min() - padding) for i in range(3)]
        maxs = [min([D, H, W][i], nz[i].max() + 1 + padding) for i in range(3)]
    else:
        center = [D // 2, H // 2, W // 2]
        half = max(padding, 32)
        mins = [max(0, c - half) for c in center]
        maxs = [min([D, H, W][i], center[i] + half) for i in range(3)]

    ct_crop = ct_norm[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    mask_crop = pred[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    factors = [t / s for t, s in zip(target_size, ct_crop.shape)]
    ct_resized = zoom(ct_crop, factors, order=1)
    mask_resized = zoom(mask_crop, factors, order=0)

    def fit_to_target(arr):
        result = np.zeros(target_size, dtype=np.float32)
        slices = tuple(slice(0, min(r, t)) for r, t in zip(arr.shape, target_size))
        result[slices] = arr[slices]
        return result

    return fit_to_target(ct_resized).astype(np.float32), fit_to_target(mask_resized).astype(np.float32)


def read_fixed_split(path):
    splits, section = {"train": [], "val": [], "test": []}, None
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


def process_patient(net, patient, device, rng, sample_idx_start, click_counts=CLICK_COUNTS):
    ct_norm, gt_bin = load_patient_volume(patient)
    shape = ct_norm.shape
    mid_z = shape[0] // 2
    seed_2d = click_sim.initial_seed_click(gt_bin[mid_z], rng=rng)
    pos, neg = [(mid_z, r, c) for r, c in seed_2d], []

    manifest_rows = []
    sample_idx = sample_idx_start

    def save_sample(pred, n_clicks):
        nonlocal sample_idx
        ct_crop, mask_crop = crop_with_context(ct_norm, pred)
        sample = np.stack([ct_crop, mask_crop], axis=0)  # (2, D, H, W)
        sample_path = f"{SAMPLE_OUT_DIR}/sample_{sample_idx:06d}.npy"
        np.save(sample_path, sample)
        manifest_rows.append({"patient": patient, "n_clicks": n_clicks,
                              "actual_dice": dice_score(pred, gt_bin), "sample_path": sample_path})
        sample_idx += 1

    pred = predict_volume(net, ct_norm, pos, neg, shape, device)
    if 0 in click_counts:
        save_sample(pred, 0)

    max_clicks = max(click_counts)
    for click_num in range(1, max_clicks + 1):
        errors = np.abs(pred - gt_bin)
        best_z = int(errors.sum(axis=(1, 2)).argmax())
        click, ctype = click_sim.simulate_next_click(pred[best_z], gt_bin[best_z], rng=rng)
        if click is None:
            continue
        row, col = click
        (pos if ctype == "positive" else neg).append((best_z, row, col))
        pred = predict_volume(net, ct_norm, pos, neg, shape, device)
        if click_num in click_counts:
            save_sample(pred, click_num)

    return manifest_rows, sample_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split_path", default=f"{RESULTS_DIR}/patient_split_3d.txt")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="which splits to extract from -- NEVER include 'test' here")
    args = parser.parse_args()

    if "test" in args.splits:
        raise ValueError("Refusing to extract completeness-CNN training data from the "
                         "test split -- this would contaminate final evaluation.")

    os.makedirs(SAMPLE_OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_checkpoint(args.model_path, device)

    splits = read_fixed_split(args.split_path)
    patients = []
    for split_name in args.splits:
        patients.extend(splits[split_name])

    print(f"extracting completeness-CNN training samples (CT+mask, cropped with "
          f"{CROP_PADDING_VOXELS}vx context padding, resized to {CNN_CROP_SIZE}) "
          f"from {len(patients)} patients (splits: {args.splits})", flush=True)

    rng = np.random.default_rng(SEED)
    all_manifest = []
    sample_idx = 0
    for i, patient in enumerate(patients, 1):
        try:
            rows, sample_idx = process_patient(net, patient, device, rng, sample_idx)
            all_manifest.extend(rows)
            dice_vals = [r["actual_dice"] for r in rows]
            print(f"{i}/{len(patients)} {patient}: {len(rows)} samples "
                  f"(dice range {min(dice_vals):.3f}-{max(dice_vals):.3f})", flush=True)
        except Exception as e:
            print(f"{patient}: FAILED - {e}")

    manifest_path = f"{RESULTS_DIR}/completeness_training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_manifest, f, indent=2)
    print(f"\nsaved {len(all_manifest)} training samples")
    print(f"samples -> {SAMPLE_OUT_DIR}/")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()