"""
Inference on ANY fundus image: grade it with Module 1, then cascade Module 2 stage by stage up
to stage 4 (PDR) -- or to --end-stage.

Unlike visualize_trajectory.py / evaluate_trajectory.py (Tianjin pairs only, stopping at each
pair's real follow-up grade), this needs no dataset and no follow-up image:

  1. Module 1 on the input image at full resolution -> starting stage (argmax grade, or
     --start-stage to override with a known clinical grade) + lesion mask.
  2. DRForestGAN-v2/trajectory_inference.py::synthesize_trajectory from that stage to
     --end-stage. Each step generates the next stage, re-segments the synthesized image with
     Module 1 for the next step's mask, and re-grades it (consistency check, logged only).
  3. Saves, per input image: one PNG per stage, a labelled strip figure (same layout as
     notebook 08's figures), and a JSON with Module 1's grade, LBS and consistency per stage.
  4. Optionally Module 3's "P(further progression within 2yr)" per panel
     (--module3-checkpoint + --module3-thresholds-json), captioned "(synthesized image --
     untested)" on synthesized panels, exactly as in visualize_trajectory.py.

Usage:
    python generate_progression.py --input ./my_fundus.jpg \
        --generator-checkpoint ./DRForestGAN-v2/stargan/models_poc_from_nonlong/final-G.ckpt \
        --classifier-checkpoint .../final_weights.pt \
        --seg-checkpoint EX=... --seg-checkpoint HE=... --seg-checkpoint MA=... --seg-checkpoint SE=... \
        --out-dir ./progression_outputs/

--input can also be a folder (every .jpg/.jpeg/.png/.tif in it). --dry-run uses random Module
1/2 weights to check the plumbing without checkpoints.

Caveats (also printed in the JSON): Module 1 doesn't segment neovascularisation, the defining
PDR sign, so stage-4 output reflects heavier EX/HE/MA/SE rather than learned new vessels unless
the generator picked that up from EyePACS stage-4 images; and errors compound along the
cascade, so later stages are the least reliable.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torchvision.utils import save_image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "DRForestGAN-v2"))
sys.path.insert(0, os.path.join(ROOT, "module1"))
sys.path.insert(0, ROOT)
from apply_to_progression_data import (  # noqa: E402
    add_classifier_norm_args,
    apply_classifier_norm_args,
    process_image,
)
from base_model import Generator  # noqa: E402
from compute_lbs import compute_lbs, estimate_retinal_fov_mask  # noqa: E402
from trajectory_inference import synthesize_trajectory  # noqa: E402
from visualize_trajectory import (  # noqa: E402
    denorm,
    estimate_module3_progression,
    image_tensor_to_rgb_uint8,
    load_module1_models,
    load_module2_image,
    render_patient_figure,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
STAGE_NAMES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "PDR"}


def list_inputs(path):
    if os.path.isfile(path):
        return [path]
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise FileNotFoundError(f"No images ({IMAGE_EXTS}) in {path}")
    return files


def mask_to_tensor(mask_np, size):
    from PIL import Image

    img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return torch.from_numpy(np.array(img) > 0).float().unsqueeze(0).unsqueeze(0)


def step_lbs(entry):
    rgb = image_tensor_to_rgb_uint8(entry["image"])
    return compute_lbs(entry["mask"][0, 0].cpu().numpy().astype(np.uint8), estimate_retinal_fov_mask(rgb))


def run_one(path, G, classifier, seg_models, module3, thresholds, config, device):
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(config.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Module 1 on the real input, at full resolution (better than re-grading the 128px tensor).
    m1 = process_image(path, classifier, seg_models, device=device)
    start = config.start_stage if config.start_stage is not None else int(m1["grade"])
    image = load_module2_image(path, config.image_size).to(device)
    mask = mask_to_tensor(m1["mask"], config.image_size).to(device)
    summary = {
        "input": path,
        "module1_input_grade": int(m1["grade"]),
        "module1_input_grade_probs": [float(p) for p in m1["grade_probs"]],
        "start_stage": start,
        "start_stage_source": "--start-stage" if config.start_stage is not None else "Module 1 grade",
        "end_stage": config.end_stage,
        "input_lbs": float(m1["lbs"]),
        "trained_lesions": m1["trained_lesions"],
    }

    # 2) Cascade.
    if start >= config.end_stage:
        print(f"  {name}: already at stage {start} (>= end stage {config.end_stage}) -- nothing to synthesize")
        trajectory = [{"stage": start, "image": image, "mask": mask, "consistency": None, "grade_probs": None}]
    else:
        trajectory = synthesize_trajectory(G, classifier, seg_models, image, mask, start_stage=start,
                                           end_stage=config.end_stage, c_dim=config.c_dim, device=device)

    # 3) Save per-stage images, strip figure, JSON.
    steps = []
    module3_estimates = {}
    for entry in trajectory:
        stage = entry["stage"]
        kind = "input" if entry["consistency"] is None else "synth"
        save_image(denorm(entry["image"].clone()), os.path.join(out_dir, f"stage{stage}_{kind}.png"))
        step = {"stage": stage, "stage_name": STAGE_NAMES.get(stage), "synthesized": kind == "synth",
                "lbs": float(step_lbs(entry))}
        if entry["grade_probs"] is not None:
            probs = np.asarray(entry["grade_probs"])
            step.update(module1_grade=int(probs.argmax()), module1_consistent=bool(entry["consistency"]),
                        module1_prob_of_target_stage=float(probs[stage]))
        if module3 is not None:
            prob, lbs_val, stratum = estimate_module3_progression(
                module3, entry["image"], entry["mask"], stage, thresholds, device)
            module3_estimates[stage] = (prob, lbs_val, stratum)
            step.update(module3_p_progress_2yr=prob, lbs_stratum=stratum)
        steps.append(step)
    summary["steps"] = steps
    summary["caveats"] = [
        "Module 1 does not segment neovascularisation (NV), the defining sign of PDR.",
        "Each step builds on the previous synthesized image, so errors compound; later stages are least reliable.",
        "Module 3 estimates on synthesized images are outside its training distribution (untested).",
    ]
    render_patient_figure(name, trajectory, module3_estimates, os.path.join(out_dir, "trajectory.png"))
    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump(summary, f, indent=2)

    flags = " ".join(f"{s['stage']}:{'ok' if s.get('module1_consistent') else 'x'}" for s in steps if s["synthesized"])
    print(f"  ✓ {name}: Module 1 grade {m1['grade']} -> start {start}, synthesized up to "
          f"{trajectory[-1]['stage']} (consistency {flags or '-'}) -> {out_dir}")
    return summary


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if config.end_stage >= config.c_dim:
        raise ValueError(f"--end-stage must be < --c-dim ({config.c_dim})")

    G = Generator(conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num,
                  style_dim=config.style_dim or config.c_dim).to(device)
    if not config.dry_run:
        G.load_state_dict(torch.load(config.generator_checkpoint, map_location=device))
    G.eval()
    classifier, seg_models = load_module1_models(config, device)

    module3 = thresholds = None
    if config.module3_checkpoint:
        sys.path.insert(0, os.path.join(ROOT, "module3"))
        from model import EfficientNetWeibullSurvival

        with open(config.module3_thresholds_json) as f:
            thresholds = {int(k): tuple(v) for k, v in json.load(f)["lbs_thresholds_by_grade"].items()}
        module3 = EfficientNetWeibullSurvival(pretrained=False).to(device)
        module3.load_state_dict(torch.load(config.module3_checkpoint, map_location=device))
        module3.eval()

    inputs = list_inputs(config.input)
    print(f"Generating stage trajectories up to stage {config.end_stage} for {len(inputs)} image(s)...")
    os.makedirs(config.out_dir, exist_ok=True)
    summaries = [run_one(p, G, classifier, seg_models, module3, thresholds, config, device) for p in inputs]
    with open(os.path.join(config.out_dir, "all_trajectories.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"✓ Done -- outputs in {config.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="A fundus image, or a folder of them")
    p.add_argument("--generator-checkpoint", default=None, help="Required unless --dry-run")
    p.add_argument("--classifier-checkpoint", default=None, help="Required unless --dry-run")
    p.add_argument("--seg-checkpoint", action="append", default=[], metavar="LESION=PATH",
                   help="Repeatable: EX=..., HE=..., MA=..., SE=...")
    p.add_argument("--start-stage", type=int, default=None,
                   help="Override Module 1's grade with a known clinical stage (0-4)")
    p.add_argument("--end-stage", type=int, default=4, help="Last stage to synthesize (default 4 = PDR)")
    p.add_argument("--module3-checkpoint", default=None)
    p.add_argument("--module3-thresholds-json", default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--c-dim", type=int, default=5)
    p.add_argument("--g-conv-dim", type=int, default=64)
    p.add_argument("--g-repeat-num", type=int, default=6)
    p.add_argument("--style-dim", type=int, default=None)
    p.add_argument("--out-dir", default="./progression_outputs/")
    p.add_argument("--dry-run", action="store_true", help="Random Module 1/2 weights, no checkpoints")
    add_classifier_norm_args(p)
    args = p.parse_args()
    apply_classifier_norm_args(args)

    if not args.dry_run and not (args.generator_checkpoint and args.classifier_checkpoint):
        p.error("--generator-checkpoint and --classifier-checkpoint are required unless --dry-run")
    if args.module3_checkpoint and not args.module3_thresholds_json:
        p.error("--module3-thresholds-json is required alongside --module3-checkpoint")
    main(args)
