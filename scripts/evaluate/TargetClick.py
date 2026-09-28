import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.transforms import SpatialPad
from monai.networks.nets import DynUNet
from scipy.ndimage import gaussian_filter, label as connected_components

from pathlib import Path
import sys

# Locate shared utilities when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

import aorta_data
import Clicksim as click_sim
from paths import LOCAL_RESULTS_DIR

RESULTS_DIR = str(LOCAL_RESULTS_DIR)
SEED = 0
MAX_CLICKS = 12
TARGET_DICE_LEVELS = [0.70, 0.80, 0.85, 0.90, 0.95]
PATCH_SIZE = (64, 128, 128)
OVERLAP_RATIO = 0.5
WINDOW_BATCH_SIZE = 16
SIGMA = 3
MIN_COMPONENT_VOXELS = 500


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
    net = DynUNet(
        spatial_dims=3, in_channels=3, out_channels=2,
        kernel_size=kernels, strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance", deep_supervision=False, res_block=True,
    )
    return net.to(device)


def load_checkpoint(model_path, device):
    net = build_dynunet_3d(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()
    return net


def dice_score(pred, gt):
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (pred * gt).sum() / denom)


def filter_small_components(pred, min_voxels=MIN_COMPONENT_VOXELS):
    if pred.sum() == 0:
        return pred
    labeled, n_components = connected_components(pred > 0.5)
    if n_components <= 1:
        return pred
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    return (labeled == largest_label).astype(pred.dtype)


def predict_volume(net, ct_norm, pos_clicks, neg_clicks, shape, device,
                   patch_dhw=PATCH_SIZE, overlap=OVERLAP_RATIO):
    D0, H0, W0 = shape
    pd, ph, pw = patch_dhw

    pad_d = max(0, pd - D0)
    pad_h = max(0, ph - H0)
    pad_w = max(0, pw - W0)
    if pad_d or pad_h or pad_w:
        padder = SpatialPad(spatial_size=(max(pd, D0), max(ph, H0), max(pw, W0)), mode="constant")
        ct_norm = np.asarray(padder(ct_norm[None]))[0]

    D, H, W = ct_norm.shape
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    pred_acc   = np.zeros((D, H, W), dtype=np.float32)
    weight_acc = np.zeros((D, H, W), dtype=np.float32)

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
                pos_patch = [(cz-z, cy-y, cx-x) for (cz, cy, cx) in pos_clicks
                             if z <= cz < z+pd and y <= cy < y+ph and x <= cx < x+pw]
                neg_patch = [(cz-z, cy-y, cx-x) for (cz, cy, cx) in neg_clicks
                             if z <= cz < z+pd and y <= cy < y+ph and x <= cx < x+pw]
                windows.append(((z, y, x), build_input_3d(ct_patch, pos_patch, neg_patch, ct_patch.shape)))

    net.eval()
    with torch.no_grad():
        for start in range(0, len(windows), WINDOW_BATCH_SIZE):
            window_batch = windows[start:start + WINDOW_BATCH_SIZE]
            t = torch.from_numpy(np.stack([w[1] for w in window_batch])).float().to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = net(t)
                probs = torch.softmax(out, dim=1)[:, 1].float().cpu().numpy()
            for i, ((wz, wy, wx), _) in enumerate(window_batch):
                pred_acc[wz:wz+pd, wy:wy+ph, wx:wx+pw] += probs[i] * weight_map
                weight_acc[wz:wz+pd, wy:wy+ph, wx:wx+pw] += weight_map

    weight_acc[weight_acc == 0] = 1.0
    final_pred = pred_acc / weight_acc
    final_pred = (final_pred[:D0, :H0, :W0] > 0.5).astype(np.float32)
    return filter_small_components(final_pred)


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


def evaluate_one_patient(net, patient, max_clicks, target_levels, device, rng):
    ct_norm, gt_bin = load_patient_volume(patient)
    shape = ct_norm.shape
    mid_z = shape[0] // 2
    seed_2d = click_sim.initial_seed_click(gt_bin[mid_z], rng=rng)
    pos = [(mid_z, r, c) for r, c in seed_2d]
    neg = []

    clicks_to_reach = {level: None for level in target_levels}
    click_breakdown_at_target = {level: None for level in target_levels}
    dice_history = []

    pred = predict_volume(net, ct_norm, pos, neg, shape, device)
    d = dice_score(pred, gt_bin)
    dice_history.append(d)
    for level in target_levels:
        if clicks_to_reach[level] is None and not np.isnan(d) and d >= level:
            clicks_to_reach[level] = 0
            click_breakdown_at_target[level] = {"positive": len(pos), "negative": len(neg)}

    for click_num in range(1, max_clicks + 1):
        errors = np.abs(pred - gt_bin)
        best_z = int(errors.sum(axis=(1, 2)).argmax())
        click, ctype = click_sim.simulate_next_click(pred[best_z], gt_bin[best_z], rng=rng)
        if click is None:
            break
        row, col = click
        (pos if ctype == "positive" else neg).append((best_z, row, col))
        pred = predict_volume(net, ct_norm, pos, neg, shape, device)
        d = dice_score(pred, gt_bin)
        dice_history.append(d)
        print(f"    click {click_num}/{max_clicks} dice={d:.3f}", flush=True)
        for level in target_levels:
            if clicks_to_reach[level] is None and not np.isnan(d) and d >= level:
                clicks_to_reach[level] = click_num
                click_breakdown_at_target[level] = {"positive": len(pos), "negative": len(neg)}

    return clicks_to_reach, click_breakdown_at_target, dice_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--split_path", default=f"{RESULTS_DIR}/patient_split_3d.txt")
    parser.add_argument("--max_clicks", type=int, default=MAX_CLICKS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_checkpoint(args.model_path, device)

    splits = read_fixed_split(args.split_path)
    test_patients = splits["test"]
    print(f"evaluating clicks-to-target on {len(test_patients)} test patients, "
          f"targets={TARGET_DICE_LEVELS}, max_clicks={args.max_clicks}", flush=True)

    rng = np.random.default_rng(SEED)
    records = []

    for i, patient in enumerate(test_patients, 1):
        try:
            clicks_to_reach, click_breakdown, dice_history = evaluate_one_patient(
                net, patient, args.max_clicks, TARGET_DICE_LEVELS, device, rng)
            records.append({"patient": patient, "clicks_to_reach": clicks_to_reach,
                            "click_breakdown_at_target": click_breakdown,
                            "dice_history": dice_history})
            reached_090 = clicks_to_reach.get(0.90)
            breakdown_090 = click_breakdown.get(0.90)
            breakdown_str = f"pos={breakdown_090['positive']}/neg={breakdown_090['negative']}" if breakdown_090 else "n/a"
            print(f"{i}/{len(test_patients)} {patient}: "
                  f"clicks_to_0.90={'never' if reached_090 is None else reached_090} ({breakdown_str}) "
                  f"final_dice={dice_history[-1]:.3f}", flush=True)
        except Exception as e:
            print(f"{patient}: FAILED - {e}")

    raw_out = f"{RESULTS_DIR}/clicks_to_target_raw_{args.tag}.json"
    with open(raw_out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"saved -> {raw_out}")

    print(f"\n=== Clicks needed to reach each target dice ===")
    summary_rows = []
    for level in TARGET_DICE_LEVELS:
        reached = [r["clicks_to_reach"][level] for r in records if r["clicks_to_reach"][level] is not None]
        n_never = len(records) - len(reached)
        mean_clicks = float(np.mean(reached)) if reached else float("nan")
        median_clicks = float(np.median(reached)) if reached else float("nan")

        breakdowns = [r["click_breakdown_at_target"][level] for r in records
                      if r["click_breakdown_at_target"][level] is not None]
        mean_pos = float(np.mean([b["positive"] for b in breakdowns])) if breakdowns else float("nan")
        mean_neg = float(np.mean([b["negative"] for b in breakdowns])) if breakdowns else float("nan")

        print(f"dice>={level:.2f}: mean_clicks={mean_clicks:.2f} median_clicks={median_clicks:.1f} "
              f"reached={len(reached)}/{len(records)} never_reached={n_never} "
              f"(mean_positive={mean_pos:.1f} mean_negative={mean_neg:.1f})")
        summary_rows.append((level, mean_clicks, median_clicks, len(reached), n_never))

    latex_lines = [r"\begin{table}[h]", r"\centering",
                   f"\\caption{{Clicks needed to reach target Dice - {args.tag}}}",
                   r"\begin{tabular}{lcccc}", r"\toprule",
                   r"Target Dice & Mean Clicks & Median Clicks & Reached & Never Reached \\", r"\midrule"]
    for level, mean_c, median_c, n_reached, n_never in summary_rows:
        latex_lines.append(f"{level:.2f} & {mean_c:.2f} & {median_c:.1f} & "
                           f"{n_reached}/{len(records)} & {n_never} \\\\")
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex_out = f"{RESULTS_DIR}/clicks_to_target_table_{args.tag}.tex"
    with open(latex_out, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"saved -> {latex_out}")

    fig, ax = plt.subplots(figsize=(8, 5))
    levels = [row[0] for row in summary_rows]
    means = [row[1] for row in summary_rows]
    ax.bar([str(l) for l in levels], means, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Target Dice")
    ax.set_ylabel("Mean clicks needed")
    ax.set_title(f"Clicks needed to reach target Dice - {args.tag}")
    ax.grid(True, alpha=0.3, axis="y")
    for i, m in enumerate(means):
        ax.annotate(f"{m:.1f}", (i, m), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/clicks_to_target_{args.tag}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()