"""
One-time preprocessing: trains an AffineRegistrationNet (see registration_network.py) on
FIRE/LongDR/Tianjin baseline<->follow-up pairs via self-supervised photometric loss, then runs
it once more over every pair to produce a cache of pre-warped follow-up images, keyed by
(source, baseline_path) -- content-addressed, so it stays valid regardless of dataset
construction order later.

This is a preprocessing step, not part of GAN training itself: Module 2's data loaders
consume the resulting cache to substitute a warped, baseline-aligned follow-up image in place
of the raw one wherever a cache entry exists, instead of registering on-the-fly inside the
training loop every epoch.

Registration-quality prefiltering: Tianjin already ships its own registration-quality score
(corrected_manifest.csv's `quality` column, from the dataset providers' own SIFT/ECC pairing
recovery). This script's --tianjin-min-pair-quality reuses it via TianjinLongitudinalDataset's
own min_pair_quality argument, so this network isn't trained on obviously-mismatched pairs.

Validation against FIRE's ground truth (CODE_FIXES item 5.2): FIRE ships manually placed
control-point correspondences per pair (FIRE/Ground Truth/control_points_<pair>_1_2.txt, one
"x1 y1 x2 y2" row per point: baseline image 1, follow-up image 2, original pixels). With
--validate-fire (default on when FIRE is loaded), each method's predicted mapping from baseline to
follow-up coordinates is applied to the baseline points and compared with the follow-up points:
mean / median error per pair (in original FIRE pixels) and the FIRE-style AUC of the success-rate
curve over thresholds 0-25 px, for 'identity' (no registration), 'net' (this network) and 'sift'
(classical SIFT + ratio test + RANSAC affine, cv2.estimateAffine2D). Written to
--validation-out. --method sift builds the cache with the classical registration instead of the
network (no training) -- use it if the network's gain over identity is negligible.

Usage:
    python train_registration.py \
        --fire-dir /content/data/FIRE_dataset \
        --longdr-dir /content/data/LongDRScreening_20150209 \
        --tianjin-dir /content/data/retinal-dr-longitudinal \
        --tianjin-min-pair-quality 0.5 \
        --num-epochs 10 \
        --checkpoint-out ./registration_net.ckpt \
        --cache-out ./registered_followups.pt
"""
import argparse
import os
import sys
import time

import json

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registration_network import AffineRegistrationNet, registration_photometric_loss  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from fire_dataset import FIREDataset  # noqa: E402
from longdr_dataset import LongDRScreeningDataset  # noqa: E402
from tianjin_dataset import TianjinLongitudinalDataset  # noqa: E402


def denorm(x):
    return ((x + 1) / 2).clamp_(0, 1)


def _pair_collate(batch):
    """Only extracts 'baseline'/'follow_up' -- sidesteps the mixed-extra-keys collate issue
    documented in combined_dataset.py's AugmentedPairDataset, which doesn't apply here since
    this script never touches Tianjin's extra keys."""
    baseline = torch.stack([b["baseline"] for b in batch])
    follow_up = torch.stack([b["follow_up"] for b in batch])
    return baseline, follow_up


def _get_baseline_path(dataset, index):
    """Extracts the raw baseline image path from whichever of the three dataset classes this
    is -- each stores its pairs slightly differently (see fire_dataset.py/longdr_dataset.py/
    tianjin_dataset.py), so this is a small dispatch rather than a shared interface change."""
    if hasattr(dataset, "pairs") and dataset.pairs and isinstance(dataset.pairs[0], (list, tuple)):
        return dataset.pairs[index][0]  # FIREDataset: [baseline_path, followup_path]
    if hasattr(dataset, "pairs") and dataset.pairs and isinstance(dataset.pairs[0], dict):
        return dataset.pairs[index]["baseline"]  # LongDRScreeningDataset
    if hasattr(dataset, "manifest"):
        return dataset.manifest.iloc[index]["baseline_path"]  # TianjinLongitudinalDataset
    raise TypeError(f"Don't know how to extract a baseline path from {type(dataset)}")


def _get_followup_path(dataset, index):
    """Same dispatch as _get_baseline_path, for the follow-up image (an absolute/openable path)."""
    if hasattr(dataset, "pairs") and dataset.pairs and isinstance(dataset.pairs[0], (list, tuple)):
        return dataset.pairs[index][1]
    if hasattr(dataset, "pairs") and dataset.pairs and isinstance(dataset.pairs[0], dict):
        return dataset.pairs[index]["follow_up"]
    if hasattr(dataset, "manifest"):
        return os.path.join(dataset.dataset_dir, dataset.manifest.iloc[index]["followup_path"])
    raise TypeError(f"Don't know how to extract a follow-up path from {type(dataset)}")


def _openable(dataset, path):
    return path if os.path.isabs(path) or os.path.isfile(path) else os.path.join(dataset.dataset_dir, path)


def sift_affine(baseline_path, followup_path, size=512, min_matches=6, ratio=0.75):
    """Classical registration: SIFT keypoints on the green channel of both images (resized to
    size x size), Lowe ratio-test matches, RANSAC affine follow-up -> baseline
    (cv2.estimateAffine2D). Returns the 2x3 matrix in [0, 1]-normalised image coordinates (so it
    applies at any resolution), or None when too few inliers."""
    import cv2

    def gray(p):
        return np.array(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))[:, :, 1]

    sift = cv2.SIFT_create()
    kb, db = sift.detectAndCompute(gray(baseline_path), None)
    kf, df = sift.detectAndCompute(gray(followup_path), None)
    if db is None or df is None or len(kb) < min_matches or len(kf) < min_matches:
        return None
    matches = cv2.BFMatcher().knnMatch(df, db, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < min_matches:
        return None
    src = np.float32([kf[m.queryIdx].pt for m in good]) / size
    dst = np.float32([kb[m.trainIdx].pt for m in good]) / size
    M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0 / size)
    if M is None or inliers is None or int(inliers.sum()) < min_matches:
        return None
    return M


def warp_with_affine01(img_chw01, M):
    """Warps a CHW [0,1] tensor with a follow-up -> baseline affine in [0,1] coordinates (output in
    the baseline frame)."""
    import cv2

    s = img_chw01.shape[-1]
    scale = np.diag([s, s, 1.0])
    Mpx = (scale @ np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(scale))[:2]
    hwc = (img_chw01.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out = cv2.warpAffine(hwc, Mpx, (s, s), flags=cv2.INTER_LINEAR, borderValue=0)
    return np.transpose(out, (2, 0, 1))


def theta_to_baseline_to_followup(theta):
    """affine_grid's theta (align_corners=False) maps output = baseline-frame coordinates in
    [-1, 1] to sampling locations in the follow-up; as a function of [0, 1] coordinates."""
    t = np.asarray(theta, dtype=np.float64)

    def f(pts01):
        n = pts01 * 2 - 1
        return ((np.c_[n, np.ones(len(n))] @ t.T) + 1) / 2
    return f


def fire_control_points(fire_dir, baseline_path):
    """(baseline_pts, followup_pts) in [0, 1] coordinates from FIRE's ground truth, or None."""
    pair = os.path.basename(baseline_path).rsplit("_", 1)[0]  # A01_1.jpg -> A01
    name = f"control_points_{pair}_1_2.txt"
    for root, _dirs, files in os.walk(fire_dir):
        if name in files:
            pts = np.loadtxt(os.path.join(root, name), ndmin=2)
            w, h = Image.open(baseline_path).size
            return pts[:, :2] / [w, h], pts[:, 2:4] / [w, h], max(w, h)
    return None


def validate_on_fire(fire_ds, fire_dir, net, device, image_size, sift_size):
    """Control-point error per method (see module docstring), in original FIRE pixels."""
    methods = {"identity": [], "net": [], "sift": []}
    sift_failed = 0
    for i in range(len(fire_ds)):
        b_path, f_path = fire_ds.pairs[i]
        gt = fire_control_points(fire_dir, b_path)
        if gt is None:
            continue
        b01, f01, orig = gt
        mappings = {"identity": lambda p: p}
        if net is not None:
            sample = fire_ds[i]
            with torch.no_grad():
                _w, theta = net(sample["baseline"].unsqueeze(0).to(device), sample["follow_up"].unsqueeze(0).to(device))
            mappings["net"] = theta_to_baseline_to_followup(theta[0].cpu().numpy())
        M = sift_affine(b_path, f_path, size=sift_size)
        if M is None:
            sift_failed += 1
            mappings["sift"] = mappings["identity"]
        else:
            Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]  # baseline -> follow-up
            mappings["sift"] = lambda p, Minv=Minv: np.c_[p, np.ones(len(p))] @ Minv.T
        for name, fn in mappings.items():
            methods[name].append(float(np.linalg.norm((fn(b01) - f01) * orig, axis=1).mean()))
    thresholds = np.arange(0, 25.01, 0.5)
    out = {"n_pairs": len(methods["identity"]), "sift_failed_pairs": sift_failed, "units": "original FIRE px"}
    for name, errs in methods.items():
        if not errs:
            continue
        errs = np.array(errs)
        success = [(errs <= t).mean() for t in thresholds]
        out[name] = {"mean_error": float(errs.mean()), "median_error": float(np.median(errs)),
                     "auc_0_25px": float(np.trapezoid(success, thresholds) / 25.0)}
    return out


def build_datasets(config):
    datasets = []
    sources = []  # parallel list of (source_name, dataset) for the content-addressed cache pass
    if config.fire_dir:
        try:
            ds = FIREDataset(config.fire_dir, image_size=config.image_size, category="A")
            if len(ds) > 0:
                datasets.append(ds)
                sources.append(("FIRE", ds))
        except Exception as e:
            print(f"⚠ Could not load FIRE: {e}")
    if config.longdr_dir:
        try:
            ds = LongDRScreeningDataset(config.longdr_dir, image_size=config.image_size, use_normalized=True)
            if len(ds) > 0:
                datasets.append(ds)
                sources.append(("LongDRScreening", ds))
        except Exception as e:
            print(f"⚠ Could not load LongDRScreening: {e}")
    if config.tianjin_dir:
        try:
            ds = TianjinLongitudinalDataset(
                config.tianjin_dir, image_size=config.image_size, min_pair_quality=config.tianjin_min_pair_quality,
                condition_on="baseline",  # register every pair, whatever the follow-up grade
            )
            if len(ds) > 0:
                datasets.append(ds)
                sources.append(("Tianjin", ds))
        except Exception as e:
            print(f"⚠ Could not load Tianjin: {e}")
    if not datasets:
        raise RuntimeError("No datasets loaded!")
    return datasets, sources


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Training-only affine registration network")
    print("=" * 70)
    print(f"Device: {device}")

    print("\n[1/3] Loading FIRE + LongDRScreening + Tianjin pairs...")
    datasets, sources = build_datasets(config)
    combined = ConcatDataset(datasets)
    print(f"✓ Combined: {len(combined)} pairs across {len(datasets)} source(s)")

    loader = DataLoader(
        combined, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=_pair_collate
    )

    net = AffineRegistrationNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    if config.method == "sift":
        config.num_epochs = 0  # classical registration -- nothing to train

    print(f"\n[2/3] Training for {config.num_epochs} epochs (self-supervised photometric loss)...")
    start_time = time.time()
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        for baseline, follow_up in loader:
            baseline, follow_up = baseline.to(device), follow_up.to(device)
            warped, _theta = net(baseline, follow_up)
            loss = registration_photometric_loss(warped, baseline)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader) if len(loader) else float("nan")
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] | L1: {avg_loss:.4f} | Elapsed: {elapsed / 60:.1f}min")

    if config.method == "net":
        os.makedirs(os.path.dirname(config.checkpoint_out) or ".", exist_ok=True)
        torch.save(net.state_dict(), config.checkpoint_out)
        print(f"✓ Saved registration network to {config.checkpoint_out}")
    net.eval()

    fire_ds = next((ds for name, ds in sources if name == "FIRE"), None)
    if config.validate_fire and fire_ds is not None:
        print("\n[validation] FIRE ground-truth control points (identity / net / sift)...")
        val = validate_on_fire(fire_ds, config.fire_dir, net if config.method == "net" else None, device,
                               config.image_size, config.sift_size)
        val["cache_method"] = config.method
        print(json.dumps(val, indent=2))
        if val["n_pairs"] == 0:
            print("  ⚠ no control_points_*_1_2.txt files found under --fire-dir (FIRE/Ground Truth/)")
        if config.validation_out:
            os.makedirs(os.path.dirname(config.validation_out) or ".", exist_ok=True)
            with open(config.validation_out, "w") as f:
                json.dump(val, f, indent=2)
            print(f"✓ Wrote {config.validation_out}")

    print(f"\n[3/3] Pre-warping every pair once ({config.method}) and writing the cache (keyed by (source, baseline_path))...")
    cache = {}
    n_fallback = 0
    with torch.no_grad():
        for source_name, ds in sources:
            for i in range(len(ds)):
                sample = ds[i]
                if config.method == "sift":
                    M = sift_affine(_openable(ds, _get_baseline_path(ds, i)), _get_followup_path(ds, i),
                                    size=config.sift_size)
                    if M is None:  # too few matches: keep the raw follow-up (identity)
                        n_fallback += 1
                        M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
                    warped_uint8 = warp_with_affine01(denorm(sample["follow_up"].clone()), M)
                else:
                    baseline = sample["baseline"].unsqueeze(0).to(device)
                    follow_up = sample["follow_up"].unsqueeze(0).to(device)
                    warped, _theta = net(baseline, follow_up)
                    warped_uint8 = (denorm(warped[0]).cpu().numpy() * 255).astype(np.uint8)  # CHW, 0-255

                baseline_path = _get_baseline_path(ds, i)
                cache[(source_name, baseline_path)] = warped_uint8

    os.makedirs(os.path.dirname(config.cache_out) or ".", exist_ok=True)
    torch.save(cache, config.cache_out)
    print(f"✓ Wrote {len(cache)} pre-warped follow-up images to {config.cache_out}"
          + (f" ({n_fallback} SIFT failures kept unregistered)" if n_fallback else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire-dir", default=None)
    parser.add_argument("--longdr-dir", default=None)
    parser.add_argument("--tianjin-dir", default=None)
    parser.add_argument("--tianjin-min-pair-quality", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-out", default="./registration_net.ckpt")
    parser.add_argument("--cache-out", default="./registered_followups.pt")
    parser.add_argument("--method", choices=["net", "sift"], default="net",
                        help="net: the self-supervised affine network (default); sift: classical SIFT + "
                             "RANSAC affine per pair (no training)")
    parser.add_argument("--sift-size", type=int, default=512, help="Resolution SIFT keypoints are found at")
    parser.add_argument("--no-validate-fire", dest="validate_fire", action="store_false",
                        help="Skip the FIRE ground-truth control-point validation")
    parser.add_argument("--validation-out", default=None, help="JSON path for the FIRE validation numbers")
    args = parser.parse_args()
    train(args)
