"""
Round 4 (item R4-3.4): does the Generator draw the stage it is asked for, by a grader that is NOT
the Discriminator's own stage head?

Round 3's pretraining reported 74-100 % "agreement" for the fakes of most stages -- but that was the
Discriminator's stage head, which graded REAL held-out images at 17.5 % accuracy (balanced 27 %), so
the Generator had learned to satisfy that head, not to draw lesions. This script takes held-out
EyePACS images (grade-balanced), asks the Generator for every stage 0-4, and grades each output with
every --grader (Module 1's ResNet-50 graders, e.g. EyePACS and FGADR). For each grader it reports:

    agreement[k]     fraction of stage-k requests the grader reads as k (chance ~20 %)
    confusion        requested stage (rows) x graded stage (columns)
    real_accuracy    the grader on the same REAL images at this image size -- the ceiling

Report the grader that was NOT in the Generator's loss (gan_guidance.FrozenGrader) as the main
number; the loss grader's agreement is optimistic by construction.

Usage:
    python tools/stage_control_check.py --generator .../final-G.ckpt --image-size 256 \
        --dataset-dir /content/data/eyepacs --preset eyepacs_resized \
        --grader eyepacs=.../best_validation_weights.pt:eyepacs --grader fgadr=.../final_weights.pt:fgadr \
        --per-grade 40 --out .../stage_control_check.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "module1"), os.path.join(ROOT, "DRForestGAN-v2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from apply_to_progression_data import CLASSIFIER_NORMS, load_classifier  # noqa: E402
from base_model import Generator, add_arch_arg, arch_kwargs, check_run_config, load_checkpoint  # noqa: E402
from crosssectional_dataset import CrossSectionalDRDataset  # noqa: E402


@torch.no_grad()
def grade_batch(clf, x_pm1, mean, std, size=512):
    """Module 1 grader on a [-1, 1] batch (resized to the grader's 512 px, its training normalisation)."""
    x = ((x_pm1 + 1) / 2).clamp(0, 1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return clf((x - m) / s).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--preset", default="eyepacs_resized")
    ap.add_argument("--module1-cache", default=None, help="Supplies the lesion-mask input channel (else zeros)")
    ap.add_argument("--grader", action="append", required=True, metavar="NAME=PATH:NORM")
    ap.add_argument("--per-grade", type=int, default=40, help="Held-out real images per grade")
    ap.add_argument("--split-seed", type=int, default=0, help="The pretraining's --seed (its val split)")
    ap.add_argument("--val-fraction", type=float, default=0.1, help="The pretraining's val_fraction")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--c-dim", type=int, default=5)
    ap.add_argument("--g-conv-dim", type=int, default=64)
    ap.add_argument("--g-repeat-num", type=int, default=6)
    ap.add_argument("--style-dim", type=int, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", required=True)
    add_arch_arg(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    check_run_config(args.generator, args.image_size, args.model_arch)
    G = Generator(conv_dim=args.g_conv_dim, c_dim=args.c_dim, repeat_num=args.g_repeat_num,
                  style_dim=args.style_dim, **arch_kwargs(args.model_arch)[0]).to(dev)
    load_checkpoint(G, args.generator, dev, args.model_arch)
    G.eval()

    ds = CrossSectionalDRDataset(args.dataset_dir, preset=args.preset, image_size=args.image_size, augment=False,
                                 module1_cache_path=args.module1_cache)
    # The pretraining's own held-out images: crosssectional_dataset.get_crosssectional_loaders' split
    # (per grade: seeded shuffle, the first val_fraction are val), replicated with the same seed.
    rng = np.random.RandomState(args.split_seed)
    pick = np.random.RandomState(args.seed)
    idx = []
    for g in range(args.c_dim):
        cand = np.where(ds.grades == g)[0]
        rng.shuffle(cand)
        val = cand[:int(round(len(cand) * args.val_fraction))].copy()
        pick.shuffle(val)
        idx += val[:args.per_grade].tolist()
    print(f"{len(idx)} held-out real images (up to {args.per_grade} per grade) x {args.c_dim} requested stages")

    graders = {}
    for spec in args.grader:
        name, rest = spec.split("=", 1)
        path, norm = rest.rsplit(":", 1)
        graders[name] = (load_classifier(path, device=dev), *CLASSIFIER_NORMS[norm], path, norm)

    conf = {n: np.zeros((args.c_dim, args.c_dim), dtype=int) for n in graders}
    real_hits = {n: 0 for n in graders}
    for b in range(0, len(idx), args.batch_size):
        batch = [ds[i] for i in idx[b:b + args.batch_size]]
        x = torch.stack([s["image"] for s in batch]).to(dev)
        mask = torch.stack([s["mask"] for s in batch]).to(dev)
        src = torch.tensor([s["grade"] for s in batch], device=dev)
        for name, (clf, mean, std, _p, _n) in graders.items():
            real_hits[name] += int((grade_batch(clf, x, mean, std) == src).sum())
        for k in range(args.c_dim):
            c = F.one_hot(torch.full((len(batch),), k, device=dev), args.c_dim).float()
            with torch.no_grad():
                fake = G(torch.cat([x, mask], 1), c)
            for name, (clf, mean, std, _p, _n) in graders.items():
                for pg in grade_batch(clf, fake, mean, std).tolist():
                    conf[name][k, pg] += 1

    out = {"generator": args.generator, "image_size": args.image_size, "n_real": len(idx), "graders": {}}
    for name, (_c, _m, _s, path, norm) in graders.items():
        cm = conf[name]
        agree = {str(k): float(cm[k, k] / max(cm[k].sum(), 1)) for k in range(args.c_dim)}
        out["graders"][name] = {"checkpoint": path, "norm": norm, "agreement": agree,
                                "mean_agreement": float(np.mean(list(agree.values()))),
                                "real_accuracy_at_this_size": real_hits[name] / len(idx),
                                "confusion_requested_x_graded": cm.tolist()}
        print(f"\n{name} ({norm}): real images at {args.image_size} px graded correctly "
              f"{real_hits[name] / len(idx):.1%} (the ceiling); requested stage read as that stage:")
        print("  " + "  ".join(f"stage {k}: {v:.1%}" for k, v in agree.items())
              + f"  | mean {out['graders'][name]['mean_agreement']:.1%} (chance 20 %)")
        print("  confusion (rows = requested, columns = graded):")
        for k in range(args.c_dim):
            print("   ", k, cm[k].tolist())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ Wrote {args.out}")


if __name__ == "__main__":
    main()
