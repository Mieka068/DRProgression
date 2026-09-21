"""
Saves labeled panel figures of Module 2's synthesized multi-stage trajectory, with Module 3's
per-step progression-probability estimate captioned beneath each panel (manuscript Sec 3.4.4's
"pipeline output" -- the two modules paired together, not evaluated in isolation).

Checked directly before writing this: nothing else in the repo writes Module 2's synthesized
trajectory images to disk. train_module2_poc.py only saves a single-step training-sanity strip
(training_samples_poc/epoch_N.jpg). evaluate_trajectory.py calls synthesize_trajectory() and
reduces its output straight to PSNR/SSIM/FID/consistency numbers -- the images themselves are
computed in memory and never saved. This script is the first thing that keeps them.

FRAMING, read before presenting a figure this script produces: Module 3 does not predict an
exact date. It predicts a probability of progressing within a fixed horizon, and the only
horizon it has real training signal for is 2 years (Tianjin's one checkpoint) -- see
module3/train_module3_poc.py's own module docstring. Every caption below therefore reads
"Est. P(further progression within 2yr): NN%", never a specific month or date.

OPEN ARCHITECTURAL CAVEAT, not a bug: running Module 3 on a SYNTHESIZED intermediate-cascade
image (treating step 2's generated image as a fresh "baseline" to ask "how long until step
3?") is mechanically possible -- the model just takes an image + grade + LBS stratum -- but
statistically untested, since Module 3 was only ever trained on real photographs. Every
synthesized-image estimate below is captioned "(synthesized image -- untested)" distinctly
from the real-baseline estimate, and this module prints the caveat once per run rather than
silently treating both as equally validated. Whether to include synthesized-image estimates in
the actual defense/presentation is a judgment call for the adviser, not something this script
decides (see docs/IMPLEMENTATION_PLAN.md's Open decisions #6).

Usage (after notebooks 06 AND 07 have both produced trained checkpoints):
    python visualize_trajectory.py \
        --tianjin-dir <path> --tianjin-module1-cache <path> \
        --generator-checkpoint ./DRForestGAN-v2/stargan/models_poc/final-G.ckpt \
        --classifier-checkpoint <path> \
        --seg-checkpoint EX=<path> --seg-checkpoint MA=<path> \
        --module3-checkpoint ./module3_runs_poc/final-model.ckpt \
        --module3-thresholds-json ./module3_runs_poc/poc_results.json \
        --num-patients 4 --out-dir ./trajectory_figures/

Omitting --module3-checkpoint still works -- figures render with the Module 1/2 content only
(image + stage label + Module 1 consistency check), no progression-probability caption.

--dry-run renders the figure layout with random Module 1/2 weights, useful for checking the
panel layout looks right before real checkpoints exist. It does not affect Module 3 -- Module
3 is used with real weights whenever --module3-checkpoint is given, regardless of --dry-run,
since loading it isn't the expensive/blocking part this flag is meant to skip.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless -- this script only ever saves figures, never shows them
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "DRForestGAN-v2"))
from base_model import Generator  # noqa: E402
from trajectory_inference import synthesize_trajectory  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "module3"))
from dataset import IMAGENET_MEAN, IMAGENET_STD, TianjinSurvivalDataset  # noqa: E402
from model import EfficientNetWeibullSurvival  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "module1"))
from apply_to_progression_data import ALL_LESIONS, load_classifier, load_segmentation_model  # noqa: E402
from compute_lbs import classify_lbs_stratum, compute_lbs, estimate_retinal_fov_mask  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tianjin_dataset import _module1_image_id_candidates  # noqa: E402

STRATUM_TO_IDX = {"low": 0, "medium": 1, "high": 2}


def denorm(x):
    return ((x + 1) / 2).clamp_(0, 1)


def load_module2_image(path, image_size):
    """Loads an image in Module 2's [-1,1] normalization convention (not Module 3's ImageNet
    stats) -- this feeds the Generator, matching train_module2_poc.py/evaluate_trajectory.py."""
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


def image_tensor_to_rgb_uint8(image_tensor):
    """[-1,1] CHW tensor -> HxWx3 uint8 numpy array, for compute_lbs's FOV estimation and for
    matplotlib's imshow."""
    denorm_01 = denorm(image_tensor[0] if image_tensor.dim() == 4 else image_tensor)
    return (denorm_01.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def prepare_for_module3(image_tensor, target_size=380):
    """Converts a Module 2 [-1,1]-range image tensor to Module 3's expected ImageNet-normalized
    input -- the two modules use different preprocessing (see module3/dataset.py's own note on
    why), so this is a real conversion, not just a resize."""
    denorm_01 = denorm(image_tensor)
    resized = F.interpolate(denorm_01, size=(target_size, target_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(1, 3, 1, 1)
    return (resized - mean) / std


def estimate_module3_progression(module3_model, image_tensor_pm1, mask_tensor, stage, thresholds_by_grade, device):
    """
    Runs Module 3 on one step's image (real or synthesized) and returns
    (probability_of_progression_within_2yr, lbs_value, stratum). Computes LBS fresh from the
    step's own image + mask (compute_lbs.py), classifies it against the SAVED training
    thresholds (classify_lbs_stratum -- not a refit), and reads the stage itself as the
    baseline-grade input (Task H), consistent with how Module 3 was trained: the grade of the
    image being scored, not the original real baseline's grade, once we're several cascade
    steps in.
    """
    rgb_uint8 = image_tensor_to_rgb_uint8(image_tensor_pm1)
    fov_mask = estimate_retinal_fov_mask(rgb_uint8)
    lesion_mask_np = mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8)
    if lesion_mask_np.shape != fov_mask.shape:
        lesion_mask_img = Image.fromarray((lesion_mask_np * 255).astype(np.uint8)).resize(
            (fov_mask.shape[1], fov_mask.shape[0]), Image.NEAREST
        )
        lesion_mask_np = (np.array(lesion_mask_img) > 0).astype(np.uint8)
    lbs_value = compute_lbs(lesion_mask_np, fov_mask)

    stratum = classify_lbs_stratum(lbs_value, stage, thresholds_by_grade)
    stratum_idx = torch.tensor([STRATUM_TO_IDX[stratum]], dtype=torch.long, device=device)
    grade_tensor = torch.tensor([stage], dtype=torch.long, device=device)

    module3_input = prepare_for_module3(image_tensor_pm1, target_size=380).to(device)
    with torch.no_grad():
        shape, scale = module3_model(module3_input, stratum_idx, grade_tensor)
        s_2yr = module3_model.survival_function(shape, scale, torch.tensor([2.0], device=device))
    prob_progression = float((1 - s_2yr).item())
    return prob_progression, lbs_value, stratum


def load_module1_models(config, device):
    classifier = load_classifier(config.classifier_checkpoint, device=device, dry_run=config.dry_run)
    seg_models = {lesion: None for lesion in ALL_LESIONS}
    for entry in config.seg_checkpoint:
        lesion, path = entry.split("=", 1)
        seg_models[lesion] = load_segmentation_model(path, device=device, dry_run=config.dry_run)
    if config.dry_run and not config.seg_checkpoint:
        seg_models["EX"] = load_segmentation_model(None, device=device, dry_run=True)
    return classifier, seg_models


def render_patient_figure(patient_id, trajectory, module3_estimates, out_path):
    """One PNG per patient: one column per trajectory stage (baseline + every synthesized
    step), each showing the image, stage label, Module 1 consistency check, and Module 3's
    progression-probability caption where available."""
    n = len(trajectory)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, entry in zip(axes, trajectory):
        rgb = image_tensor_to_rgb_uint8(entry["image"])
        ax.imshow(rgb)
        ax.axis("off")

        is_baseline = entry["consistency"] is None
        title = f"Stage {entry['stage']}" + (" (real baseline)" if is_baseline else " (synthesized)")
        lines = [title]
        if not is_baseline:
            lines.append(f"Module 1 consistency: {entry['consistency']}")

        est = module3_estimates.get(entry["stage"])
        if est is not None:
            prob, _lbs, stratum = est
            caption = f"Est. P(further progression within 2yr): {prob:.0%}  [{stratum} LBS]"
            if not is_baseline:
                caption += "\n(synthesized image -- untested)"
            lines.append(caption)

        ax.set_title("\n".join(lines), fontsize=9)

    fig.suptitle(f"Patient {patient_id}", fontsize=12, y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(config.out_dir, exist_ok=True)

    print("\n[1/4] Loading Tianjin survival dataset (source of real progression pairs)...")
    survival_dataset = TianjinSurvivalDataset(
        config.tianjin_dir, module1_cache_path=config.tianjin_module1_cache, image_size=config.image_size
    )
    manifest = survival_dataset.manifest
    cache = torch.load(config.tianjin_module1_cache, weights_only=False)

    progressing = manifest[manifest["followup_icdr"] > manifest["baseline_icdr"]]
    if len(progressing) == 0:
        raise RuntimeError("No real progression pairs (followup_icdr > baseline_icdr) found -- nothing to visualize.")
    selected = progressing.head(config.num_patients)
    if len(selected) < config.num_patients:
        print(f"  Only {len(selected)} progression pairs available (requested {config.num_patients}).")

    print("\n[2/4] Loading Generator checkpoint and Module 1 classifier/segmentation models...")
    style_dim = config.style_dim or config.c_dim
    G = Generator(
        conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num, style_dim=style_dim
    ).to(device)
    G.load_state_dict(torch.load(config.generator_checkpoint, map_location=device))
    G.eval()
    classifier, seg_models = load_module1_models(config, device)

    module3_model = None
    thresholds_by_grade = None
    if config.module3_checkpoint:
        print("\n[3/4] Loading Module 3 checkpoint + saved LBS thresholds...")
        if not config.module3_thresholds_json:
            raise ValueError("--module3-thresholds-json is required alongside --module3-checkpoint.")
        with open(config.module3_thresholds_json) as f:
            module3_results = json.load(f)
        if "lbs_thresholds_by_grade" not in module3_results:
            raise KeyError(
                f"{config.module3_thresholds_json} has no 'lbs_thresholds_by_grade' key -- "
                "retrain with the current train_module3_poc.py (see docs/IMPLEMENTATION_PLAN.md "
                "Task I), which saves this automatically."
            )
        thresholds_by_grade = {int(k): tuple(v) for k, v in module3_results["lbs_thresholds_by_grade"].items()}

        module3_model = EfficientNetWeibullSurvival(pretrained=False).to(device)
        module3_model.load_state_dict(torch.load(config.module3_checkpoint, map_location=device))
        module3_model.eval()
        print(
            "\nNOTE: Module 3 estimates on SYNTHESIZED (non-real) images are an untested "
            "extrapolation -- Module 3 was only ever trained on real photographs. "
            "Synthesized-image estimates are captioned '(synthesized image -- untested)' "
            "below; treat them as illustrative, not validated (see this script's module "
            "docstring and docs/IMPLEMENTATION_PLAN.md's Open decisions #6)."
        )
    else:
        print("\n[3/4] --module3-checkpoint not given -- figures will render Module 1/2 content only.")

    print(f"\n[4/4] Synthesizing and rendering {len(selected)} patient trajector{'y' if len(selected)==1 else 'ies'}...")
    for _, row in selected.iterrows():
        baseline_path = os.path.join(config.tianjin_dir, row["baseline_path"])
        baseline_img = load_module2_image(baseline_path, config.image_size).to(device)
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

        module3_estimates = {}
        if module3_model is not None:
            for entry in trajectory:
                module3_estimates[entry["stage"]] = estimate_module3_progression(
                    module3_model, entry["image"], entry["mask"], entry["stage"], thresholds_by_grade, device
                )

        out_path = os.path.join(config.out_dir, f"patient_{row['patient_id']}.png")
        render_patient_figure(row["patient_id"], trajectory, module3_estimates, out_path)
        print(f"  ✓ Saved {out_path}")

    print(f"\n✓ Wrote {len(selected)} figure(s) to {config.out_dir}")


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
    parser.add_argument("--dry-run", action="store_true", help="Random-init Module 1/2 weights, no checkpoints needed")
    parser.add_argument("--module3-checkpoint", default=None, help="Optional -- omit to render Module 1/2 content only")
    parser.add_argument("--module3-thresholds-json", default=None, help="Required alongside --module3-checkpoint")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=5)
    parser.add_argument("--g-conv-dim", type=int, default=64)
    parser.add_argument("--g-repeat-num", type=int, default=6)
    parser.add_argument("--style-dim", type=int, default=None, help="Defaults to c_dim, matching Generator's own default")
    parser.add_argument("--num-patients", type=int, default=4)
    parser.add_argument("--out-dir", default="./trajectory_figures/")
    args = parser.parse_args()

    if not args.dry_run and not args.classifier_checkpoint:
        parser.error("--classifier-checkpoint is required unless --dry-run is set")
    if args.module3_checkpoint and not args.module3_thresholds_json:
        parser.error("--module3-thresholds-json is required alongside --module3-checkpoint")

    run(args)
