"""
Per-cascade-step evaluation for Module 2's autoregressive multi-stage trajectory synthesis
(see DRForestGAN-v2/trajectory_inference.py and docs/IMPLEMENTATION_PLAN.md Task D).

RQ1 asks how clinical credibility varies ACROSS severity stage transitions -- a cascade that
is only ever evaluated at step 1 doesn't answer that. This script evaluates each cascade step
against REAL follow-up images, bucketed by how many stages ahead of the baseline that step is
(step_count = followup_icdr - baseline_icdr).

Only Tianjin has a real grade on BOTH the baseline and the follow-up image (see
module3/dataset.py) -- FIRE and LongDR have no follow-up grade at all, so they cannot
participate in this evaluation regardless of architecture; this is a real data limitation,
not an oversight (see docs/ROADMAP.md's Objective 2 "training supervision" design question).
For a given real pair, only the cascade step whose target stage equals the real followup_icdr
has a real image to compare against -- earlier/later steps in the same trajectory have no
ground truth (see trajectory_inference.py's own docstring on why intermediate steps are
unsupervised). This script therefore reports one (real, synthesized) comparison per pair, at
that pair's own step_count bucket -- not a comparison at every step of every trajectory.

Usage (after a trained Module 2 checkpoint and a Tianjin module1 cache both exist):
    python evaluate_trajectory.py \
        --tianjin-dir /content/data/retinal-dr-longitudinal \
        --tianjin-module1-cache /content/drive/.../module1_outputs_tianjin.pt \
        --generator-checkpoint ./DRForestGAN-v2/stargan/models_poc/final-G.ckpt \
        --out ./trajectory_eval_results.json
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


def evaluate(config):
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/3] Loading Tianjin survival dataset (source of real baseline+follow-up grade pairs)...")
    survival_dataset = TianjinSurvivalDataset(
        config.tianjin_dir, module1_cache_path=config.tianjin_module1_cache, image_size=config.image_size
    )
    manifest = survival_dataset.manifest
    cache = torch.load(config.tianjin_module1_cache, weights_only=False)

    print("\n[2/3] Loading Generator checkpoint...")
    style_dim = config.style_dim or config.c_dim
    G = Generator(
        conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num, style_dim=style_dim
    ).to(device)
    G.load_state_dict(torch.load(config.generator_checkpoint, map_location=device))
    G.eval()

    print(f"\n[3/3] Synthesizing trajectories for {len(manifest)} real pairs...")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    per_step_psnr = defaultdict(list)
    per_step_ssim = defaultdict(list)
    per_step_images = defaultdict(lambda: {"real": [], "fake": []})
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
            baseline_img,
            mask,
            start_stage=int(row["baseline_icdr"]),
            end_stage=int(row["followup_icdr"]),
            c_dim=config.c_dim,
            device=device,
        )
        final_stage, synthesized = trajectory[-1]
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

    results = {}
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
        results[f"step_{step_count}"] = entry
        print(
            f"  step_count={step_count}: n={n}, PSNR={entry['mean_psnr']:.3f}, "
            f"SSIM={entry['mean_ssim']:.3f}, FID={entry['fid']}"
        )

    with open(config.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Wrote {config.out}")
    print(
        "\nHonest framing for RQ1: report per-step numbers separately, not a single pooled "
        "number -- a bigger step_count means a longer autoregressive chain (more compounding "
        "generator error), and small-N buckets should be captioned as such, not treated as "
        "equally reliable as larger ones."
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tianjin-dir", required=True)
    parser.add_argument("--tianjin-module1-cache", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
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
    evaluate(args)
