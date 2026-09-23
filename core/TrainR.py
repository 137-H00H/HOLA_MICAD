import os
import gc
import random
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss
from monai.transforms import Compose, SpatialPad, SpatialPadd, RandCropByPosNegLabeld, ResizeWithPadOrCrop
from monai.utils import set_determinism
from scipy.ndimage import gaussian_filter, label as connected_components

import aorta_data
import Clicksim as click_sim
from paths import LOCAL_RESULTS_DIR

PATCH_SIZE            = (64, 128, 128)
PATCHES_PER_PATIENT   = 4
N_EPOCHS              = 60
BATCH_SIZE            = 24
DEFAULT_LR            = 1e-3
CLICK_FREE_PROB       = 0.4
MAX_TRAIN_CORRECTIONS = 3
SIGMA                 = 3
SEED                  = 0
EARLY_STOP_PATIENCE    = 15
OVERLAP_RATIO          = 0.5
LOAD_WORKERS           = 4
WINDOW_BATCH_SIZE      = 16
VAL_CORRECTIONS        = 3

RESULTS_DIR = str(LOCAL_RESULTS_DIR)
SPLIT_OUT   = f"{RESULTS_DIR}/patient_split_3d.txt"

USE_WANDB     = True
WANDB_PROJECT = "RDM"

DATASET_LOADERS = {
    "sega":       aorta_data.list_sega_patients,
    "dissection": aorta_data.list_dissection_patients,
    "cisunet":    aorta_data.list_cisunet_patients,
    "aortaseg60": aorta_data.list_aortaseg60_patients,
    "tbad":       aorta_data.list_tbad_patients,
}


def elapsed(start, device=None):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def load_patient_volume(patient):
    ct, gt = aorta_data.load_patient_any(patient)
    ct_norm = scale_intensity(ct)
    gt_bin  = (gt > 0.5).astype(np.float32)
    return ct_norm, gt_bin


def load_fixed_validation_patch(patient, patch_transform, seed):
    patches = load_patient_patches(
        patient, patch_transform, np.random.default_rng(seed), augment=False)
    return patches[0]


def preload_validation_patches(patients, patch_transform):
    patches = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(LOAD_WORKERS, len(patients))) as executor:
        futures = {
            executor.submit(load_fixed_validation_patch, patient, patch_transform,
                            SEED + patient_index): patient
            for patient_index, patient in enumerate(patients)
        }
        loaded = {}
        for completed, future in enumerate(as_completed(futures), 1):
            patient = futures[future]
            try:
                loaded[patient] = future.result()
            except Exception as error:
                print(f"skip val patch {patient}: {error}")
    for patient in patients:
        if patient in loaded:
            patches.append((patient, loaded[patient]))
    gc.collect()
    return patches


def make_patch_transform(patch_size, num_samples=PATCHES_PER_PATIENT):
    return Compose([
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=7, neg=3,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
    ])


def scale_intensity(vol, a_min=-175, a_max=250):
    return ((np.clip(vol, a_min, a_max) - a_min) / (a_max - a_min)).astype(np.float32)


def augment_3d(ct, gt, rng):
    for axis in range(3):
        if rng.random() < 0.5:
            ct = np.flip(ct, axis=axis).copy()
            gt = np.flip(gt, axis=axis).copy()
    k = rng.integers(0, 4)
    if k > 0:
        ct = np.rot90(ct, k=k, axes=(1, 2)).copy()
        gt = np.rot90(gt, k=k, axes=(1, 2)).copy()
    if rng.random() < 0.5:
        ct = np.clip(ct * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1), 0, 1)
    if rng.random() < 0.3:
        ct = np.clip(ct + rng.normal(0, 0.02, ct.shape).astype(np.float32), 0, 1)
    return ct.astype(np.float32), gt.astype(np.float32)


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


def dice_score(pred, gt):
    pred  = (pred > 0.5).astype(np.float32)
    gt    = (gt   > 0.5).astype(np.float32)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (pred * gt).sum() / denom)


def iou_score(pred, gt):
    pred  = (pred > 0.5).astype(np.float32)
    gt    = (gt   > 0.5).astype(np.float32)
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union == 0:
        return float("nan")
    return float(intersection / union)


def load_patient_patches(patient, patch_transform, rng, augment=False, volume=None):
    ct_norm, gt_bin = volume if volume is not None else load_patient_volume(patient)
    data = {"image": ct_norm[None], "label": gt_bin[None]}
    samples = patch_transform(data)
    patches = []
    exact_fitter = ResizeWithPadOrCrop(spatial_size=PATCH_SIZE, mode="constant")
    for s in samples:
        ct_p = s["image"].numpy()[0].copy()
        gt_p = s["label"].numpy()[0].copy()
        if ct_p.shape != PATCH_SIZE:
            ct_p = np.asarray(exact_fitter(ct_p[None]))[0]
        if gt_p.shape != PATCH_SIZE:
            gt_p = np.asarray(exact_fitter(gt_p[None]))[0]
        if augment:
            ct_p, gt_p = augment_3d(ct_p, gt_p, rng)
        patches.append((ct_p, gt_p))
    return patches


def simulate_clicks_batch(net, volumes, labels, device, rng, corrections):
    shapes = [volume.shape for volume in volumes]
    positive_clicks, negative_clicks = [], []
    for label, shape in zip(labels, shapes):
        if label.shape[0] == 0 or label.sum() == 0:
            positive_clicks.append([])
            negative_clicks.append([])
            continue
        mid_z = shape[0] // 2
        if mid_z >= label.shape[0]:
            mid_z = label.shape[0] // 2
        seeds = click_sim.initial_seed_click(label[mid_z], rng=rng)
        positive_clicks.append([(mid_z, row, column) for row, column in seeds])
        negative_clicks.append([])

    volume_tensor = torch.from_numpy(np.stack(volumes)).float().to(device)
    predictions = [np.zeros(shape, dtype=np.float32) for shape in shapes]
    for _ in range(corrections):
        click_tensor = make_click_channels_batch(
            shapes, positive_clicks, negative_clicks, device)
        tensor = torch.cat((volume_tensor[:, None], click_tensor), dim=1)
        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = net(tensor)
        probabilities = torch.softmax(output, dim=1)[:, 1].float().cpu().numpy()
        for index, (prediction, label) in enumerate(zip(probabilities, labels)):
            if label.shape[0] == 0 or label.sum() == 0:
                continue
            predictions[index] = (prediction > 0.5).astype(np.float32)
            if predictions[index].shape != label.shape:
                continue
            errors = np.abs(predictions[index] - label)
            best_z = int(errors.sum(axis=(1, 2)).argmax())
            click_2d, click_type = click_sim.simulate_next_click(
                predictions[index][best_z], label[best_z], rng=rng)
            if click_2d is not None:
                row, column = click_2d
                target = positive_clicks if click_type == "positive" else negative_clicks
                target[index].append((best_z, row, column))
    return positive_clicks, negative_clicks, predictions


def make_click_channels_batch(shapes, positive_clicks, negative_clicks, device):
    if len(set(shapes)) != 1:
        raise ValueError("batched click channels require equal patch shapes")
    channels = torch.zeros((len(shapes), 2, *shapes[0]), device=device)
    for batch_index, click_sets in enumerate(zip(positive_clicks, negative_clicks)):
        for channel_index, clicks in enumerate(click_sets):
            for z, y, x in clicks:
                channels[batch_index, channel_index, int(z), int(y), int(x)] = 1.0

    radius = int(3 * SIGMA)
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(coordinates ** 2) / (2 * SIGMA ** 2))
    kernel /= kernel.sum()
    flattened = channels.reshape(-1, 1, *shapes[0])
    flattened = F.conv3d(flattened, kernel.view(1, 1, -1, 1, 1), padding=(radius, 0, 0))
    flattened = F.conv3d(flattened, kernel.view(1, 1, 1, -1, 1), padding=(0, radius, 0))
    flattened = F.conv3d(flattened, kernel.view(1, 1, 1, 1, -1), padding=(0, 0, radius))
    channels = flattened.reshape(len(shapes), 2, *shapes[0])
    maxima = channels.flatten(2).amax(dim=2).view(len(shapes), 2, 1, 1, 1)
    return channels / maxima.clamp_min(1e-8)


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


def get_random_dataset_order(order_seed):
    names = list(DATASET_LOADERS.keys())
    order_rng = np.random.default_rng(order_seed)
    order_rng.shuffle(names)
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",      type=float, default=DEFAULT_LR)
    parser.add_argument("--tag",     type=str,   default="3d")
    parser.add_argument("--epochs",  type=int,   default=N_EPOCHS)
    parser.add_argument("--patch_d", type=int,   default=PATCH_SIZE[0])
    parser.add_argument("--patch_h", type=int,   default=PATCH_SIZE[1])
    parser.add_argument("--patch_w", type=int,   default=PATCH_SIZE[2])
    parser.add_argument("--order_seed", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    args = parser.parse_args()

    patch_dhw = (args.patch_d, args.patch_h, args.patch_w)
    lr_tag    = f"lr{args.lr:g}".replace(".", "p")
    model_out = f"{RESULTS_DIR}/deepedit_3d_{args.tag}_{lr_tag}.pt"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_determinism(seed=SEED)
    rng = np.random.default_rng(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"device={device} ({device_name})", flush=True)

    dataset_order = get_random_dataset_order(args.order_seed)
    included = dataset_order[:args.stage]
    print(f"order_seed={args.order_seed}  full_random_order={dataset_order}  "
          f"stage={args.stage}  included_this_stage={included}", flush=True)

    descriptive_tag = f"{args.tag}_{'-'.join(included)}" if included else args.tag

    if USE_WANDB:
        config = vars(args)
        config["dataset_order"] = dataset_order
        config["included_this_stage"] = included
        wandb.init(project=WANDB_PROJECT, name=descriptive_tag, config=config,
                   settings=wandb.Settings(init_timeout=120))

    fixed_split = read_fixed_split(SPLIT_OUT)

    all_patients = aorta_data.list_all_usable_patients()
    for name in included:
        all_patients += DATASET_LOADERS[name]()

    stage_patients = set(all_patients)
    splits = {
        "train": [p for p in fixed_split["train"] if p in stage_patients],
        "val":   fixed_split["val"],
        "test":  fixed_split["test"],
    }

    print(f"patch={patch_dhw}  lr={args.lr}  epochs={args.epochs}  "
          f"stage_total={len(all_patients)}  "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    patch_transform = make_patch_transform(patch_dhw)
    validation_patch_transform = make_patch_transform(patch_dhw, num_samples=1)

    val_data = preload_validation_patches(splits["val"], validation_patch_transform)

    net     = build_dynunet_3d(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optim   = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_dice     = -1.0
    best_val_iou      = -1.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_data = []
        for p in splits["train"]:
            try:
                train_data.extend(load_patient_patches(p, patch_transform, rng, augment=True))
            except Exception as e:
                print(f"skip train {p}: {e}")
        rng.shuffle(train_data)

        net.train()
        epoch_loss, n_batches = 0.0, 0

        for b_start in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[b_start:b_start + BATCH_SIZE]
            inputs = [None] * len(batch)
            interactive_indices = [index for index in range(len(batch))
                                   if rng.random() >= CLICK_FREE_PROB]
            for index, (ct_norm, _) in enumerate(batch):
                if index not in interactive_indices:
                    inputs[index] = build_input_3d(ct_norm, [], [], ct_norm.shape)
            if interactive_indices:
                net.eval()
                interactive_volumes = [batch[index][0] for index in interactive_indices]
                interactive_labels = [batch[index][1] for index in interactive_indices]
                positive, negative, _ = simulate_clicks_batch(
                    net, interactive_volumes, interactive_labels, device, rng,
                    MAX_TRAIN_CORRECTIONS)
                net.train()
                for local_index, batch_index in enumerate(interactive_indices):
                    volume = batch[batch_index][0]
                    inputs[batch_index] = build_input_3d(
                        volume, positive[local_index], negative[local_index], volume.shape)
            targets = [gt_bin[None] for _, gt_bin in batch]

            x = torch.from_numpy(np.stack(inputs)).float().to(device)
            y = torch.from_numpy(np.stack(targets)).long().to(device)
            optim.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss = loss_fn(net(x), y)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        net.eval()
        val_dices, val_ious = [], []
        for val_start_index in range(0, len(val_data), BATCH_SIZE):
            panel_batch = val_data[val_start_index:val_start_index + BATCH_SIZE]
            volumes = [item[1][0] for item in panel_batch]
            labels = [item[1][1] for item in panel_batch]
            _, _, predictions = simulate_clicks_batch(
                net, volumes, labels, device, rng, corrections=VAL_CORRECTIONS)
            val_dices.extend(dice_score(prediction, label)
                              for prediction, label in zip(predictions, labels))
            val_ious.extend(iou_score(prediction, label)
                            for prediction, label in zip(predictions, labels))

        mean_val_dice = float(np.nanmean(val_dices)) if val_dices else float("nan")
        mean_val_iou  = float(np.nanmean(val_ious))  if val_ious  else float("nan")
        improved      = mean_val_dice > best_val_dice

        if USE_WANDB:
            wandb.log({"train_loss": avg_loss,
                       f"PATCH_val_dice_{VAL_CORRECTIONS}clicks": mean_val_dice,
                       f"PATCH_val_iou_{VAL_CORRECTIONS}clicks": mean_val_iou}, step=epoch)

        if improved:
            best_val_dice     = mean_val_dice
            best_val_iou      = mean_val_iou
            epochs_no_improve = 0
            torch.save(net.state_dict(), model_out)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={avg_loss:.4f}  "
              f"val_dice@{VAL_CORRECTIONS}clicks={mean_val_dice:.4f}  "
              f"LR={sched.get_last_lr()[0]:.4g}  "
              f"best val Dice @{VAL_CORRECTIONS} clicks: {best_val_dice:.4f}  "
              f"model -> {model_out}", flush=True)

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"early stopping at epoch {epoch+1}")
            break

        sched.step()

        del train_data
        gc.collect()

    print(f"best subset val dice={best_val_dice:.4f} iou={best_val_iou:.4f}; saved -> {model_out}")

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()