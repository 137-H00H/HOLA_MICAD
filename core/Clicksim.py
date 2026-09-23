import numpy as np
from scipy.ndimage import distance_transform_cdt
from skimage import measure


def initial_seed_click(label_2d, connected_regions=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    label_2d = (label_2d > 0.5).astype(np.float32)
    if label_2d.sum() == 0:
        return []

    blobs = measure.label(label_2d.astype(int), background=0)
    max_blob = int(blobs.max())

    clicks = []
    for ridx in range(1, min(connected_regions, max_blob) + 1):
        region = (blobs == ridx).astype(np.float32)
        if region.sum() == 0:
            continue

        distance = distance_transform_cdt(region).flatten()
        probability = np.exp(distance) - 1.0

        idx = np.where(region.flatten() > 0)[0]
        if len(idx) == 0:
            continue
        probs = probability[idx]
        probs = probs / probs.sum()

        seed = rng.choice(idx, size=1, p=probs)[0]
        r, c = np.unravel_index(seed, region.shape)
        clicks.append((int(r), int(c)))

    return clicks


def find_discrepancy(pred_2d, label_2d):
    pred = (pred_2d > 0.5).astype(np.float32)
    label = (label_2d > 0.5).astype(np.float32)

    disparity = label - pred
    under_seg = (disparity > 0).astype(np.float32)
    over_seg = (disparity < 0).astype(np.float32)
    return under_seg, over_seg


def click_from_discrepancy(discrepancy_2d, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    discrepancy_2d = (discrepancy_2d > 0.5).astype(np.float32)
    if discrepancy_2d.sum() == 0:
        return None

    distance = distance_transform_cdt(discrepancy_2d).flatten()
    probability = np.exp(distance) - 1.0
    idx = np.where(discrepancy_2d.flatten() > 0)[0]

    probs = probability[idx]
    total = probs.sum()
    if total <= 0:
        seed = rng.choice(idx, size=1)[0]
    else:
        seed = rng.choice(idx, size=1, p=probs / total)[0]

    r, c = np.unravel_index(seed, discrepancy_2d.shape)
    return (int(r), int(c))


def simulate_next_click(pred_2d, label_2d, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    under_seg, over_seg = find_discrepancy(pred_2d, label_2d)
    under_area = int(under_seg.sum())
    over_area = int(over_seg.sum())

    if under_area == 0 and over_area == 0:
        return None, None

    if under_area >= over_area:
        click = click_from_discrepancy(under_seg, rng)
        return click, "positive"
    else:
        click = click_from_discrepancy(over_seg, rng)
        return click, "negative"