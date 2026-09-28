import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import sys

# Locate shared utilities when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

from paths import LOCAL_RESULTS_DIR

RESULTS_DIR = str(LOCAL_RESULTS_DIR)
SEED = 0
CNN_CROP_SIZE = (96, 96, 96)
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-3
ACCEPTABLE_DICE = 0.85
TAU_CANDIDATES = np.arange(0.50, 0.96, 0.02)


class CompletenessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self._block(2, 16)
        self.conv2 = self._block(16, 32)
        self.conv3 = self._block(32, 64)
        self.conv4 = self._block(64, 128)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x).flatten(1)
        return torch.sigmoid(self.fc(x)).squeeze(1)


class MaskDataset(Dataset):
    def __init__(self, manifest_rows):
        self.rows = manifest_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        stacked = np.load(row["sample_path"])  # shape (2, D, H, W): CT, mask
        return torch.from_numpy(stacked).float(), float(row["actual_dice"])


def load_manifest():
    with open(f"{RESULTS_DIR}/completeness_training_manifest.json") as f:
        return json.load(f)


def select_tau(model, val_loader, device, acceptable_dice=ACCEPTABLE_DICE):
    model.eval()
    all_q, all_y = [], []
    with torch.no_grad():
        for masks, dice in val_loader:
            q = model(masks.to(device)).cpu().numpy()
            all_q.extend(q.tolist())
            all_y.extend(dice.numpy().tolist())
    all_q, all_y = np.array(all_q), np.array(all_y)

    best_tau, best_precision, best_coverage = None, 0.0, 0.0
    print("\ntau  |  precision (stopped & actually good)  |  coverage (%% stopped)")
    for tau in TAU_CANDIDATES:
        stop_mask = all_q >= tau
        if stop_mask.sum() == 0:
            continue
        precision = float((all_y[stop_mask] >= acceptable_dice).mean())
        coverage = float(stop_mask.mean())
        print(f"{tau:.2f} |  {precision:.3f}  |  {coverage:.3f}")
        if precision >= 0.90 and coverage > best_coverage:
            best_tau, best_precision, best_coverage = float(tau), precision, coverage

    if best_tau is None:
        best_tau = float(TAU_CANDIDATES[-1])
        print(f"\nWARNING: no tau reached 90% precision -- defaulting to strict tau={best_tau}")
    else:
        print(f"\nSelected tau={best_tau:.2f} (precision={best_precision:.3f}, "
              f"stops {best_coverage*100:.1f}% of cases correctly)")

    corr = float(np.corrcoef(all_q, all_y)[0, 1])
    mae = float(np.mean(np.abs(all_q - all_y)))
    return best_tau, corr, mae


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = load_manifest()
    print(f"loaded {len(manifest)} training samples")

    patients = sorted(set(r["patient"] for r in manifest))
    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * 0.2))
    val_patients = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    train_rows = [r for r in manifest if r["patient"] in train_patients]
    val_rows = [r for r in manifest if r["patient"] in val_patients]
    print(f"train: {len(train_rows)} samples ({len(train_patients)} patients)")
    print(f"val:   {len(val_rows)} samples ({len(val_patients)} patients)")

    train_loader = DataLoader(MaskDataset(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MaskDataset(val_rows), batch_size=BATCH_SIZE, shuffle=False)

    model = CompletenessCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for masks, dice in train_loader:
            masks, dice = masks.to(device), dice.to(device).float()
            optimizer.zero_grad()
            q_pred = model(masks)
            loss = loss_fn(q_pred, dice)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for masks, dice in val_loader:
                masks, dice = masks.to(device), dice.to(device).float()
                q_pred = model(masks)
                val_losses.append(loss_fn(q_pred, dice).item())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        print(f"epoch {epoch}/{EPOCHS}  train_loss={mean_train_loss:.4f}  val_loss={mean_val_loss:.4f}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model.state_dict(), f"{RESULTS_DIR}/completeness_cnn.pt")

    print(f"\nbest val_loss={best_val_loss:.4f}, checkpoint saved -> {RESULTS_DIR}/completeness_cnn.pt")

    model.load_state_dict(torch.load(f"{RESULTS_DIR}/completeness_cnn.pt", map_location=device))
    tau, corr, mae = select_tau(model, val_loader, device)

    metadata = {
        "tau": tau, "val_correlation": corr, "val_mae": mae,
        "acceptable_dice": ACCEPTABLE_DICE, "cnn_input_size": list(CNN_CROP_SIZE),
    }
    with open(f"{RESULTS_DIR}/completeness_cnn_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n=== Final validation performance ===")
    print(f"correlation (predicted vs actual dice): r={corr:.3f}")
    print(f"mean absolute error: {mae:.3f}")
    print(f"metadata saved -> {RESULTS_DIR}/completeness_cnn_metadata.json")


if __name__ == "__main__":
    main()