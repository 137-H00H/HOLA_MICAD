import os
import json
import time
import argparse
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


def get_real_spacing(patient):
    dataset_name, raw_id = patient.split("::", 1)
    img_path, _ = aorta_data._patient_paths(dataset_name, raw_id)
    img = sitk.ReadImage(img_path)
    spacing_xyz = img.GetSpacing()
    return spacing_xyz[::-1]

RESULTS_DIR = str(LOCAL_RESULTS_DIR)
SEED = 0
CLICK_COUNTS = [0, 1, 3, 5, 8, 10]
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


def filter_small_components(pred, min_voxels=MIN_COMPONENT_VOXELS):
    if pred.sum() == 0:
        return pred, {"n_components": 0, "kept_sizes": [], "discarded_sizes": []}
    labeled, n_components = connected_components(pred > 0.5)
    if n_components <= 1:
        return pred, {"n_components": n_components, "kept_sizes": [int(pred.sum())], "discarded_sizes": []}
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    keep_labels = np.array([largest_label])
    discard_labels = np.array([lbl for lbl in range(1, len(sizes)) if lbl != largest_label and sizes[lbl] > 0])
    filtered = np.isin(labeled, keep_labels).astype(pred.dtype)
    info = {
        "n_components": n_components,
        "kept_sizes": sorted(sizes[keep_labels].tolist(), reverse=True),
        "discarded_sizes": sorted(sizes[discard_labels].tolist(), reverse=True),
    }
    return filtered, info


def predict_volume(net, ct_norm, pos_clicks, neg_clicks, shape, device,
                   patch_dhw=PATCH_SIZE, overlap=OVERLAP_RATIO, apply_component_filter=True):
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

    if apply_component_filter:
        final_pred, component_info = filter_small_components(final_pred)
    else:
        component_info = {"n_components": None, "kept_sizes": [], "discarded_sizes": []}

    return final_pred, component_info


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


def check_mid_slice_has_foreground(gt_bin):
    """Diagnostic: does the geometric mid-slice actually contain any
    foreground? If not, the initial seed click starts from nothing."""
    mid_z = gt_bin.shape[0] // 2
    return bool(gt_bin[mid_z].sum() > 0), mid_z


def evaluate_one_patient(net, patient, click_counts, device, rng):
    ct_norm, gt_bin = load_patient_volume(patient)
    shape = ct_norm.shape
    spacing = get_real_spacing(patient)

    mid_has_fg, mid_z = check_mid_slice_has_foreground(gt_bin)
    if not mid_has_fg:
        # Fall back to the nearest slice that actually contains
        # foreground, rather than seeding from an empty slice.
        z_with_fg = np.where(gt_bin.any(axis=(1, 2)))[0]
        if len(z_with_fg) > 0:
            mid_z = int(z_with_fg[len(z_with_fg) // 2])

    seed_2d = click_sim.initial_seed_click(gt_bin[mid_z], rng=rng)
    pos = [(mid_z, r, c) for r, c in seed_2d]
    neg = []

    results = {}
    component_log = {}

    t0 = time.perf_counter()
    pred, comp_info = predict_volume(net, ct_norm, pos, neg, shape, device)
    inference_time = time.perf_counter() - t0

    def record(inference_time, comp_info):
        d = dice_score(pred, gt_bin)
        return {
            "dice": d,
            "iou": iou_score(pred, gt_bin),
            "hd95": hausdorff_distance_95(pred, gt_bin, spacing=spacing),
            "inference_time_sec": inference_time,
            "n_clicks": len(pos) + len(neg),
            "mid_slice_had_foreground": mid_has_fg,
            "n_components": comp_info["n_components"],
            "discarded_component_sizes": comp_info["discarded_sizes"],
            "pred_voxel_count": int(pred.sum()),
            "gt_voxel_count": int(gt_bin.sum()),
            "intersection_voxel_count": int(((pred > 0.5) & (gt_bin > 0.5)).sum()),
            "positive_clicks": [[int(z), int(r), int(c)] for z, r, c in pos],
            "negative_clicks": [[int(z), int(r), int(c)] for z, r, c in neg],
        }

    if 0 in click_counts:
        results[0] = record(inference_time, comp_info)

    for click_num in range(1, max(click_counts) + 1):
        errors = np.abs(pred - gt_bin)
        best_z = int(errors.sum(axis=(1, 2)).argmax())
        click, ctype = click_sim.simulate_next_click(pred[best_z], gt_bin[best_z], rng=rng)
        if click is None:
            if click_num in click_counts:
                results[click_num] = results[max(k for k in results if k < click_num)]
            continue
        row, col = click
        (pos if ctype == "positive" else neg).append((best_z, row, col))

        t0 = time.perf_counter()
        pred, comp_info = predict_volume(net, ct_norm, pos, neg, shape, device)
        inference_time = time.perf_counter() - t0

        if click_num in click_counts:
            results[click_num] = record(inference_time, comp_info)

    return results


def latex_table_full(model_name, summary):
    n_cols = 1 + len(CLICK_COUNTS) + 3
    col_spec = "l" + "c" * (n_cols - 1)
    lines = [
        r"\begin{table}[h]", r"\centering",
        f"\\caption{{Test set performance for {model_name}}}",
        f"\\begin{{tabular}}{{{col_spec}}}", r"\toprule",
    ]
    header = ["Model"] + [f"Dice@{k}" for k in CLICK_COUNTS] + ["IoU (final)", "HD95 (final)", "N"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    row = [model_name] + [f"{summary[k]['dice']:.3f}" for k in CLICK_COUNTS]
    row += [f"{summary[CLICK_COUNTS[-1]]['iou']:.3f}",
            f"{summary[CLICK_COUNTS[-1]]['hd95']:.2f}",
            str(summary['n_patients'])]
    lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--split_path", default=f"{RESULTS_DIR}/patient_split_3d.txt")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_checkpoint(args.model_path, device)

    splits = read_fixed_split(args.split_path)
    test_patients = splits["test"]
    print(f"evaluating {len(test_patients)} test patients", flush=True)

    per_patient = {}
    all_patient_results = []
    raw_records = []
    mid_slice_empty_count = 0
    discarded_component_flags = 0

    for i, patient in enumerate(test_patients, 1):
        try:
            results = evaluate_one_patient(net, patient, CLICK_COUNTS, device, rng)
            per_patient[patient] = results
            all_patient_results.append((patient, results))
            for k in CLICK_COUNTS:
                raw_records.append({"patient": patient, "n_clicks_requested": k, **results[k]})
            if not results[0]["mid_slice_had_foreground"]:
                mid_slice_empty_count += 1
            if any(results[k]["discarded_component_sizes"] for k in CLICK_COUNTS):
                discarded_component_flags += 1
            print(f"{i}/{len(test_patients)} {patient}: "
                  f"dice@0={results[0]['dice']:.3f} dice@{CLICK_COUNTS[-1]}={results[CLICK_COUNTS[-1]]['dice']:.3f} "
                  f"hd95@{CLICK_COUNTS[-1]}={results[CLICK_COUNTS[-1]]['hd95']:.2f}", flush=True)
        except Exception as e:
            print(f"{patient}: FAILED - {e}")

    print(f"\n=== Diagnostics ===")
    print(f"patients where geometric mid-slice was empty (claim 1): {mid_slice_empty_count}/{len(test_patients)}")
    print(f"patients where connected-component filter discarded something (claim 4): "
          f"{discarded_component_flags}/{len(test_patients)}")

    raw_json_out = f"{RESULTS_DIR}/test_raw_results_{args.tag}.json"
    with open(raw_json_out, "w") as f:
        json.dump(raw_records, f, indent=2)
    print(f"saved per-patient, per-click raw results -> {raw_json_out}")

    summary = {}
    for k in CLICK_COUNTS:
        all_dice = [r[k]["dice"] for _, r in all_patient_results if not np.isnan(r[k]["dice"])]
        all_iou = [r[k]["iou"] for _, r in all_patient_results if not np.isnan(r[k]["iou"])]
        all_hd95 = [r[k]["hd95"] for _, r in all_patient_results if not np.isnan(r[k]["hd95"])]
        summary[k] = {
            "dice": float(np.mean(all_dice)) if all_dice else float("nan"),
            "iou": float(np.mean(all_iou)) if all_iou else float("nan"),
            "hd95": float(np.mean(all_hd95)) if all_hd95 else float("nan"),
        }
    summary["n_patients"] = len(all_patient_results)

    micro_summary = {}
    for k in CLICK_COUNTS:
        total_pred = sum(r[k]["pred_voxel_count"] for _, r in all_patient_results)
        total_gt = sum(r[k]["gt_voxel_count"] for _, r in all_patient_results)
        total_intersection = sum(r[k]["intersection_voxel_count"] for _, r in all_patient_results)
        denom = total_pred + total_gt
        micro_summary[k] = (2 * total_intersection / denom) if denom > 0 else float("nan")

    print("\n=== Macro vs Micro averaging (Dice) ===")
    for k in CLICK_COUNTS:
        print(f"clicks={k}: macro={summary[k]['dice']:.4f}  micro={micro_summary[k]:.4f}")

    click_efficiency_per_patient = [
        r[CLICK_COUNTS[-1]]["dice"] - r[0]["dice"]
        for _, r in all_patient_results
        if not np.isnan(r[0]["dice"]) and not np.isnan(r[CLICK_COUNTS[-1]]["dice"])
    ]
    mean_click_efficiency = float(np.mean(click_efficiency_per_patient)) if click_efficiency_per_patient else float("nan")
    print(f"\n=== Click efficiency (dice@{CLICK_COUNTS[-1]} minus dice@0) ===")
    print(f"mean improvement from clicking: {mean_click_efficiency:+.4f}")
    n_worse_after_clicks = sum(1 for e in click_efficiency_per_patient if e < 0)
    print(f"patients where clicking made things WORSE: {n_worse_after_clicks}/{len(click_efficiency_per_patient)}")

    PLAUSIBLE_ANATOMY_MIN_VOXELS = 5000
    large_discards = []
    for patient, r in all_patient_results:
        for k in CLICK_COUNTS:
            for size in r[k]["discarded_component_sizes"]:
                if size >= PLAUSIBLE_ANATOMY_MIN_VOXELS:
                    large_discards.append((patient, k, size))
    print(f"\n=== Component filter stability ===")
    print(f"discarded components >= {PLAUSIBLE_ANATOMY_MIN_VOXELS} voxels "
          f"(possibly real anatomy, not noise): {len(large_discards)}")

    latex_out = f"{RESULTS_DIR}/metrics_table_{args.tag}.tex"
    with open(latex_out, "w") as f:
        f.write(latex_table_full(args.tag, summary))
    print(latex_table_full(args.tag, summary))

    cohort_results = {}
    for patient, results in all_patient_results:
        cohort = patient.split("::", 1)[0] if "::" in patient else "unknown"
        cohort_results.setdefault(cohort, []).append((patient, results))

    print("\n=== Per-cohort breakdown ===")
    cohort_lines = ["Cohort & N & " + " & ".join(f"Dice@{k}" for k in CLICK_COUNTS) + r" \\"]
    for cohort in sorted(cohort_results):
        entries = cohort_results[cohort]
        row = [cohort, str(len(entries))]
        for k in CLICK_COUNTS:
            vals = [r[k]["dice"] for _, r in entries if not np.isnan(r[k]["dice"])]
            row.append(f"{float(np.mean(vals)) if vals else float('nan'):.3f}")
        print(f"{cohort} (n={len(entries)}): " +
              "  ".join(f"dice@{k}={row[2+i]}" for i, k in enumerate(CLICK_COUNTS)))
        cohort_lines.append(" & ".join(row) + r" \\")

    cohort_out = f"{RESULTS_DIR}/metrics_table_{args.tag}_percohort.tex"
    with open(cohort_out, "w") as f:
        f.write("\n".join(cohort_lines))
    print(f"per-cohort table saved -> {cohort_out}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ds = [summary[k]["dice"] for k in CLICK_COUNTS]
    ax.plot(CLICK_COUNTS, ds, "o-", linewidth=2, markersize=8, color="tab:blue")
    ax.set_xlabel("Number of clicks")
    ax.set_ylabel("Mean Dice (test patients)")
    ax.set_title(f"Click curve on test patients - {args.tag}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    for k, d in zip(CLICK_COUNTS, ds):
        ax.annotate(f"{d:.3f}", (k, d), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/test_click_curve_{args.tag}.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.13
    x = np.arange(len(test_patients))
    for i, k in enumerate(CLICK_COUNTS):
        vals = [per_patient[p][k]["dice"] if p in per_patient else float("nan") for p in test_patients]
        ax.bar(x + i * width, vals, width, label=f"{k} clicks")
    ax.set_xticks(x + width * (len(CLICK_COUNTS) - 1) / 2)
    ax.set_xticklabels([p.split("::", 1)[-1][:18] for p in test_patients], rotation=15, ha="right")
    ax.set_ylabel("Mean Dice")
    ax.set_title(f"Per-patient Dice on test set - {args.tag}")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/test_per_patient_{args.tag}.png", dpi=120)
    plt.close()

    print(f"saved -> {latex_out}")


if __name__ == "__main__":
    main()