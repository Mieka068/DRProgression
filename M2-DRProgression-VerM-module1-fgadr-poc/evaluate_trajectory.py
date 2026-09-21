"""
Per-cascade-step evaluation for Module 2's autoregressive multi-stage trajectory synthesis
(see DRForestGAN-v2/trajectory_inference.py).

Evaluates each cascade step against real follow-up images, bucketed by how many stages ahead
of the baseline that step is (step_count = followup_icdr - baseline_icdr). Also reports
Module 1's own re-grading consistency per step (does the synthesized image at step N read as
stage N to Module 1's own classifier?) -- a distinct signal from generator quality
(PSNR/SSIM/FID): a low consistency rate at a given step could mean the generator produced
something implausible, or that Module 1's own grader is unreliable on synthesized
(out-of-distribution) images. This script measures both without attempting to disentangle
which.

Only Tianjin has a grade on both the baseline and the follow-up image (see
module3/dataset.py) -- FIRE and LongDR have no follow-up grade, so they cannot participate in
this evaluation. For a given pair, only the cascade step whose target stage equals the real
followup_icdr has a real image to compare against -- earlier/later steps in the same
trajectory have no ground truth for PSNR/SSIM/FID (Module 1's re-grading consistency check is
available at every step, since it only needs the synthesized image itself). This script
therefore reports one PSNR/SSIM comparison per pair, at that pair's own step_count bucket,
plus a consistency flag at every intermediate step along the way.

Usage (after a trained Module 2 checkpoint, a Tianjin module1 cache, and Module 1
classifier/segmentation checkpoints all exist):
    python evaluate_trajectory.py \
        --tianjin-dir /content/data/retinal-dr-longitudinal \
        --tianjin-module1-cache /content/drive/.../module1_outputs_tianjin.pt \
        --generator-checkpoint ./DRForestGAN-v2/stargan/models_poc/final-G.ckpt \
        --classifier-checkpoint /content/drive/.../final_weights.pt \
        --seg-checkpoint EX=/path/model_EX.pth.tar --seg-checkpoint MA=/path/model_MA.pth.tar \
        --out ./trajectory_eval_results.json

Dry run (random-init Module 1 weights, for smoke-testing the plumbing):
    python evaluate_trajectory.py --tianjin-dir ... --tianjin-module1-cache ... \
        --generator-checkpoint ... --dry-run --out /tmp/dry_run.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "DRForestGAN-v2"))
from base_model import Generator  # noqa: E402
from trajectory_inference import synthesize_trajectory  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "module3"))
from dataset import TianjinSurvivalDataset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "module1"))
from apply_to_progression_data import ALL_LESIONS, load_classifier, load_segmentation_model  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tianjin_dataset import _module1_image_id_candidates  # noqa: E402


def denorm(x):
    return ((x + 1) / 2).clamp_(0, 1)


def load_module2_image(path, image_size):
    """Loads an image in the SAME [-1,1] normalization convention Module 2's training/
    inference code uses (not Module 3's ImageNet stats) -- this evaluates the Module 2
    generator, not a Module 3 model."""
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)


def load_mask(cache, baseline_path, image_size):
    for candidate_id in _module1_image_id_candidates(baseline_path):
        if candidate_id in cache:
            mask_np = cache[candidate_id]["mask"]
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (image_size, image_size), Image.NEAREST
            )
            return torch.from_numpy(np.array(mask_img) > 0).float().unsqueeze(0).unsqueeze(0)
    return torch.zeros(1, 1, image_size, image_size)


def load_module1_models(config, device):
    classifier = load_classifier(config.classifier_checkpoint, device=device, dry_run=config.dry_run)
    seg_models = {lesion: None for lesion in ALL_LESIONS}
    for entry in config.seg_checkpoint:
        lesion, path = entry.split("=", 1)
        seg_models[lesion] = load_segmentation_model(path, device=device, dry_run=config.dry_run)
    if config.dry_run and not config.seg_checkpoint:
        seg_models["EX"] = load_segmentation_model(None, device=device, dry_run=True)
    return classifier, seg_models


def evaluate(config):
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/4] Loading Tianjin survival dataset (source of real baseline+follow-up grade pairs)...")
    survival_dataset = TianjinSurvivalDataset(
        config.tianjin_dir, module1_cache_path=config.tianjin_module1_cache, image_size=config.image_size
    )
    manifest = survival_dataset.manifest
    cache = torch.load(config.tianjin_module1_cache, weights_only=False)

    print("\n[2/4] Loading Generator checkpoint and Module 1 classifier/segmentation models...")
    style_dim = config.style_dim or config.c_dim
    G = Generator(
        conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num, style_dim=style_dim
    ).to(device)
    G.load_state_dict(torch.load(config.generator_checkpoint, map_location=device))
    G.eval()
    classifier, seg_models = load_module1_models(config, device)

    print(f"\n[3/4] Synthesizing trajectories for {len(manifest)} real pairs...")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    per_step_psnr = defaultdict(list)
    per_step_ssim = defaultdict(list)
    per_step_images = defaultdict(lambda: {"real": [], "fake": []})
    consistency_by_step = defaultdict(list)  # step index within a trajectory (1, 2, ...) -> [bool]
    n_skipped_no_progression = 0

    for _, row in manifest.iterrows():
        step_count = int(row["followup_icdr"]) - int(row["baseline_icdr"])
        if step_count <= 0:
            n_skipped_no_progression += 1
            continue  # the cascade evaluates progression, not regression/no-change

        baseline_path = os.path.join(config.tianjin_dir, row["baseline_path"])
        followup_path = os.path.join(config.tianjin_dir, row["followup_path"])
        baseline_img = load_module2_image(baseline_path, config.image_size).to(device)
        real_followup_img = load_module2_image(followup_path, config.image_size).to(device)
        mask = load_mask(cache, row["baseline_path"], config.image_size)

        trajectory = synthesize_trajectory(
            G,
            classifier,
            seg_models,
            baseline_img,
            mask,
            start_stage=int(row["baseline_icdr"]),
            end_stage=int(row["followup_icdr"]),
            c_dim=config.c_dim,
            device=device,
        )
        for step_idx, entry in enumerate(trajectory[1:], start=1):
            if entry["consistency"] is not None:
                consistency_by_step[step_idx].append(bool(entry["consistency"]))

        final_stage, synthesized = trajectory[-1]["stage"], trajectory[-1]["image"]
        assert final_stage == int(row["followup_icdr"])

        real_01 = denorm(real_followup_img)
        fake_01 = denorm(synthesized)
        psnr = float(psnr_metric(fake_01, real_01))
        ssim = float(ssim_metric(fake_01, real_01))

        per_step_psnr[step_count].append(psnr)
        per_step_ssim[step_count].append(ssim)
        per_step_images[step_count]["real"].append(real_01.cpu())
        per_step_images[step_count]["fake"].append(fake_01.cpu())

    print(
        f"  Skipped {n_skipped_no_progression} pairs with no real progression (step_count <= 0) "
        "-- the cascade evaluates progression, not regression/no-change."
    )

    print("\n[4/4] Aggregating per-step metrics...")
    results = {"quality": {}, "module1_consistency": {}}
    for step_count in sorted(per_step_psnr.keys()):
        n = len(per_step_psnr[step_count])
        entry = {
            "n_pairs": n,
            "mean_psnr": float(np.mean(per_step_psnr[step_count])),
            "mean_ssim": float(np.mean(per_step_ssim[step_count])),
        }
        if n >= config.min_n_for_fid:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance

                fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
                for img in per_step_images[step_count]["real"]:
                    fid_metric.update(img.to(device), real=True)
                for img in per_step_images[step_count]["fake"]:
                    fid_metric.update(img.to(device), real=False)
                entry["fid"] = float(fid_metric.compute())
            except ImportError:
                entry["fid"] = None
        else:
            entry["fid"] = None
            entry["fid_note"] = (
                f"n_pairs ({n}) < min_n_for_fid ({config.min_n_for_fid}) -- FID is a "
                "distributional metric and unreliable on very few samples, so it's omitted "
                "here rather than reported misleadingly."
            )
        results["quality"][f"step_{step_count}"] = entry
        print(
            f"  step_count={step_count}: n={n}, PSNR={entry['mean_psnr']:.3f}, "
            f"SSIM={entry['mean_ssim']:.3f}, FID={entry['fid']}"
        )

    for step_idx in sorted(consistency_by_step.keys()):
        flags = consistency_by_step[step_idx]
        rate = float(np.mean(flags))
        results["module1_consistency"][f"cascade_step_{step_idx}"] = {
            "n": len(flags),
            "module1_agrees_with_target_stage_rate": rate,
        }
        print(f"  cascade step {step_idx}: Module 1 re-grading agreed with the target stage "
              f"{rate:.1%} of the time (n={len(flags)})")

    with open(config.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Wrote {config.out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tianjin-dir", required=True)
    parser.add_argument("--tianjin-module1-cache", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--classifier-checkpoint", default=None, help="Required unless --dry-run")
    parser.add_argument(
        "--seg-checkpoint",
        action="append",
        default=[],
        metavar="LESION=PATH",
        help="Repeatable, e.g. --seg-checkpoint EX=/path/model_EX.pth.tar --seg-checkpoint MA=/path/...",
    )
    parser.add_argument("--dry-run", action="store_true", help="Random-init Module 1 weights, no checkpoints needed")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=5)
    parser.add_argument("--g-conv-dim", type=int, default=64)
    parser.add_argument("--g-repeat-num", type=int, default=6)
    parser.add_argument(
        "--style-dim", type=int, default=None, help="Defaults to c_dim, matching Generator's own default"
    )
    parser.add_argument("--min-n-for-fid", type=int, default=5)
    parser.add_argument("--out", default="./trajectory_eval_results.json")
    args = parser.parse_args()

    if not args.dry_run and not args.classifier_checkpoint:
        parser.error("--classifier-checkpoint is required unless --dry-run is set")

    evaluate(args)
