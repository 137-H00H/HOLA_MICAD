import argparse
import json
import os
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.transforms import SpatialPad
from monai.networks.nets import DynUNet
from scipy.ndimage import gaussian_filter, label as connected_components, distance_transform_edt, binary_erosion

import aorta_data
import Clicksim as click_sim
from paths import LOCAL_RESULTS_DIR

RESULTS_DIR = str(LOCAL_RESULTS_DIR)
SEED = 0
N_CLICKS = 10
PATCH_SIZE = (64, 128, 128)
OVERLAP_RATIO = 0.5
WINDOW_BATCH_SIZE = 16
SIGMA = 3
MIN_COMPONENT_VOXELS = 500

CORRECT_RATES = [0.90, 0.70, 0.50, 0.30, 0.10, 0.00]


def scale_intensity(vol, a_min=-175, a_max=250):
    return ((np.clip(vol, a_min, a_max) - a_min) / (a_max - a_min)).astype(np.float32)


def load_patient_volume(patient):
    ct, gt = aorta_data.load_patient_any(patient)
    ct_norm = scale_intensity(ct)
    gt_bin = (gt > 0.5).astype(np.float32)
    return ct_norm, gt_bin


def get_real_spacing(patient):
    dataset_name, raw_id = patient.split("::", 1)
    img_path, _ = aorta_data._patient_paths(dataset_name, raw_id)
    img = sitk.ReadImage(img_path)
    return img.GetSpacing()[::-1]


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
    """Keeps only the single largest connected component -- the aorta
    trunk is one continuous structure."""
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


def load_perfect_click_baseline(tag):
    path = f"{RESULTS_DIR}/test_raw_results_{tag}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path} -- run EvaluateTest.py with --tag {tag} first. "
            f"The click-quality experiment reuses its 100%-correct baseline "
            f"instead of recomputing it."
        )
    with open(path) as f:
        records = json.load(f)

    by_click = {}
    for rec in records:
        c = rec["n_clicks_requested"]
        by_click.setdefault(c, []).append(rec["dice"])

    baseline = {}
    for c in range(N_CLICKS + 1):
        vals = [d for d in by_click.get(c, []) if not np.isnan(d)]
        baseline[c] = float(np.mean(vals)) if vals else float("nan")
    return baseline


def run_click_sequence_with_errors(net, ct_norm, gt_bin, device, rng, n_clicks, correct_rate):
    shape = ct_norm.shape
    mid_z = shape[0] // 2
    seed_2d = click_sim.initial_seed_click(gt_bin[mid_z], rng=rng)
    pos = [(mid_z, r, c) for r, c in seed_2d]
    neg = []

    history = {}
    pred = predict_volume(net, ct_norm, pos, neg, shape, device)
    history[0] = {"dice": dice_score(pred, gt_bin)}

    for click_num in range(1, n_clicks + 1):
        errors = np.abs(pred - gt_bin)
        best_z = int(errors.sum(axis=(1, 2)).argmax())
        click, correct_ctype = click_sim.simulate_next_click(pred[best_z], gt_bin[best_z], rng=rng)
        if click is None:
            history[click_num] = history[click_num - 1]
            continue
        row, col = click

        is_wrong = rng.random() >= correct_rate
        ctype = ("negative" if correct_ctype == "positive" else "positive") if is_wrong else correct_ctype

        (pos if ctype == "positive" else neg).append((best_z, row, col))
        pred = predict_volume(net, ct_norm, pos, neg, shape, device)
        history[click_num] = {"dice": dice_score(pred, gt_bin)}

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--split_path", default=f"{RESULTS_DIR}/patient_split_3d.txt")
    parser.add_argument("--n_clicks", type=int, default=N_CLICKS)
    args = parser.parse_args()

    print(f"loading perfect-click (100%) baseline from EvaluateTest.py's output for tag={args.tag}...")
    baseline_100 = load_perfect_click_baseline(args.tag)
    print(f"baseline loaded: dice@0={baseline_100[0]:.3f} dice@{args.n_clicks}={baseline_100[args.n_clicks]:.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_checkpoint(args.model_path, device)

    splits = read_fixed_split(args.split_path)
    test_patients = splits["test"]
    print(f"evaluating click-error robustness on {len(test_patients)} test patients, "
          f"correct_rates={CORRECT_RATES} (plus reused 1.00 baseline)", flush=True)

    results = {rate: {c: [] for c in range(args.n_clicks + 1)} for rate in CORRECT_RATES}

    for i, patient in enumerate(test_patients, 1):
        try:
            ct_norm, gt_bin = load_patient_volume(patient)
        except Exception as e:
            print(f"{patient}: FAILED to load - {e}")
            continue

        for rate in CORRECT_RATES:
            rng = np.random.default_rng(SEED)
            history = run_click_sequence_with_errors(net, ct_norm, gt_bin, device, rng,
                                                      args.n_clicks, rate)
            for c in range(args.n_clicks + 1):
                results[rate][c].append(history[c]["dice"])

        print(f"{i}/{len(test_patients)} {patient} done", flush=True)

    summary = {1.00: baseline_100}
    for rate in CORRECT_RATES:
        summary[rate] = {}
        for c in range(args.n_clicks + 1):
            vals = [d for d in results[rate][c] if not np.isnan(d)]
            summary[rate][c] = float(np.mean(vals)) if vals else float("nan")

    all_rates = [1.00] + CORRECT_RATES

    print("\n=== Click-error robustness results ===")
    for rate in all_rates:
        row = "  ".join(f"c{c}={summary[rate][c]:.3f}" for c in range(args.n_clicks + 1))
        print(f"{int(rate*100)}% correct: {row}")

    fig, ax = plt.subplots(figsize=(9, 6))
    clicks = list(range(args.n_clicks + 1))
    for rate in all_rates:
        ys = [summary[rate][c] for c in clicks]
        ax.plot(clicks, ys, "o-", linewidth=2, markersize=6, label=f"{int(rate*100)}% correct")
    ax.set_xlabel("Number of clicks")
    ax.set_ylabel("Mean Dice")
    ax.set_title(f"Robustness to incorrect user clicks - {args.tag}")
    ax.set_ylim(0, 1)
    ax.legend(title="Click accuracy", loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/click_error_robustness_{args.tag}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"saved -> {out_path}")

    latex_lines = [r"\begin{table}[h]", r"\centering",
                   f"\\caption{{Robustness to incorrect clicks - {args.tag}}}",
                   "\\begin{tabular}{l" + "c" * (args.n_clicks + 1) + "}", r"\toprule"]
    header = ["\\% Correct clicks"] + [f"c={c}" for c in clicks]
    latex_lines.append(" & ".join(header) + r" \\")
    latex_lines.append(r"\midrule")
    for rate in all_rates:
        row = [f"{int(rate*100)}\\%"] + [f"{summary[rate][c]:.3f}" for c in clicks]
        latex_lines.append(" & ".join(row) + r" \\")
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex_out = f"{RESULTS_DIR}/click_error_robustness_table_{args.tag}.tex"
    with open(latex_out, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"saved -> {latex_out}")


if __name__ == "__main__":
    main()