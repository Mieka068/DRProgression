"""
One-time preprocessing: trains an AffineRegistrationNet (see registration_network.py) on
FIRE/LongDR/Tianjin baseline<->follow-up pairs via self-supervised photometric loss, then runs
it once more over every pair to produce a cache of pre-warped follow-up images, keyed by
(source, baseline_path) -- content-addressed, not index-addressed, so it stays valid
regardless of dataset construction order later.

This is a preprocessing step, not part of GAN training itself: Module 2's data loaders
consume the resulting cache (see docs/IMPLEMENTATION_PLAN.md Task F) to substitute a warped,
baseline-aligned follow-up image in place of the raw one wherever a cache entry exists,
instead of registering on-the-fly inside the training loop every epoch.

Registration-quality prefiltering: Tianjin already ships its own registration-quality score
(corrected_manifest.csv's `quality` column, from the dataset providers' own SIFT/ECC pairing
recovery). This script's --tianjin-min-pair-quality reuses it via TianjinLongitudinalDataset's
own min_pair_quality argument, so this network isn't trained on obviously-mismatched pairs.

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

import numpy as np
import torch
import torch.optim as optim
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
                config.tianjin_dir, image_size=config.image_size, min_pair_quality=config.tianjin_min_pair_quality
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
    print("Training-only affine registration network (Task E)")
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

    os.makedirs(os.path.dirname(config.checkpoint_out) or ".", exist_ok=True)
    torch.save(net.state_dict(), config.checkpoint_out)
    print(f"✓ Saved registration network to {config.checkpoint_out}")

    print(
        "\n[3/3] Pre-warping every pair once and writing the cache "
        "(keyed by (source, baseline_path), content-addressed -- see module docstring)..."
    )
    net.eval()
    cache = {}
    with torch.no_grad():
        for source_name, ds in sources:
            for i in range(len(ds)):
                sample = ds[i]
                baseline = sample["baseline"].unsqueeze(0).to(device)
                follow_up = sample["follow_up"].unsqueeze(0).to(device)
                warped, _theta = net(baseline, follow_up)
                warped_uint8 = (denorm(warped[0]).cpu().numpy() * 255).astype(np.uint8)  # CHW, 0-255

                baseline_path = _get_baseline_path(ds, i)
                cache[(source_name, baseline_path)] = warped_uint8

    os.makedirs(os.path.dirname(config.cache_out) or ".", exist_ok=True)
    torch.save(cache, config.cache_out)
    print(f"✓ Wrote {len(cache)} pre-warped follow-up images to {config.cache_out}")


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
    args = parser.parse_args()
    train(args)
