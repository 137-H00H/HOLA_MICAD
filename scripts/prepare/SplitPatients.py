import numpy as np
from pathlib import Path
import sys

# Locate shared utilities when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

import aorta_data
from paths import LOCAL_RESULTS_DIR

SPLIT_OUT = f"{LOCAL_RESULTS_DIR}/patient_split_3d.txt"
SEED = 0
TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20


def main():
    all_patients = []
    for name in ["base", "sega", "dissection", "cisunet", "aortaseg60", "tbad"]:
        pts = aorta_data.list_patients(name)
        print(f"{name}: {len(pts)} patients")
        all_patients.extend(pts)

    print(f"total: {len(all_patients)} patients")

    rng = np.random.default_rng(SEED)
    shuffled = list(all_patients)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * TRAIN_FRACTION)
    n_val = int(n_total * VAL_FRACTION)

    train_patients = shuffled[:n_train]
    val_patients = shuffled[n_train:n_train + n_val]
    test_patients = shuffled[n_train + n_val:]

    print(f"train={len(train_patients)}  val={len(val_patients)}  test={len(test_patients)}")

    with open(SPLIT_OUT, "w") as f:
        f.write("# train\n")
        for p in train_patients:
            f.write(f"{p}\n")
        f.write("# val\n")
        for p in val_patients:
            f.write(f"{p}\n")
        f.write("# test\n")
        for p in test_patients:
            f.write(f"{p}\n")

    print(f"saved -> {SPLIT_OUT}")


if __name__ == "__main__":
    main()