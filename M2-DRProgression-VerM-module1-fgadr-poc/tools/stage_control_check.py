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

Round 9 -- two image-level checks (no grader needed), on the same images:
    no_change        each image requested at its OWN stage: mean |G(x) - x| inside the retina (0-1 scale)
                     and, per grader, how often the grade of that output differs from the grade of the
                     input itself (both should be near 0 -- round 8 read stage-0 requests as grade 4 in 40 %)
    redness[k]       mean red added inside the retina for stage-k requests: relu(dR - dG), d = output - input
                     on a 0-1 scale (tint + streaks together); change[k] = mean |G(x) - x| for stage-k requests
    tint[k]          the WHOLE-IMAGE colour shift towards red: relu(mean dR - mean dG) over the retina, per image
    streak_redness[k]  LOCAL red added once each image's own average shift is removed: relu((dR - mean dR) -
                     (dG - mean dG)) inside the retina -- red streaks / blotches, not a uniform tint
    strong_streak_area[k]  fraction of the retina where that local red / magenta jump exceeds STRONG_STREAK (0.08 on a
                     0-1 scale): the vivid bars, not mild mottling (which streak_redness also counts)
    disc_darkening[k]  how much the BRIGHTEST 1 % of the input retina (the optic disc) gets darker in the output: mean
                     relu(luminance in - luminance out) over those pixels (the dark disc spot)
    highlight_darkening[k]  the same over ALL highlights (brightest 3 % + near-saturated red: disc, exudates, glare) --
                     round 9's dark spots also hit exudates and glare; teal_shift[k] = how much red drops relative to
                     green / blue there (the spots are teal)
Compare these between runs on the same images (same --seed / --split-seed), not as absolute values.
--g-upsample / --g-padding render the Generator differently without retraining (base_model.apply_render_options);
--grid-out saves a sample grid (rows: one held-out image per grade; columns: input, then stages 0-4).

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
from base_model import (Generator, add_arch_arg, add_render_args, apply_saved_render_options, arch_kwargs,  # noqa: E402
                        check_run_config, load_checkpoint)
from crosssectional_dataset import CrossSectionalDRDataset  # noqa: E402


@torch.no_grad()
def grade_batch(clf, x_pm1, mean, std, size=512):
    """Module 1 grader on a [-1, 1] batch (resized to the grader's 512 px, its training normalisation)."""
    x = ((x_pm1 + 1) / 2).clamp(0, 1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return clf((x - m) / s).argmax(1)


STRONG_STREAK = 0.08


def brightest_mask(x01, m, frac=0.01):
    """[B,1,H,W] mask of the brightest `frac` of each image's retina pixels (luminance) -- the optic disc."""
    lum = (0.299 * x01[:, :1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3])
    out = torch.zeros_like(lum)
    for i in range(lum.size(0)):
        inside = torch.nonzero(m[i].flatten() > 0).flatten()
        if inside.numel():
            k = max(1, int(round(frac * inside.numel())))
            top = inside[torch.topk(lum[i].flatten()[inside], k).indices]  # exactly the k brightest retina pixels
            out[i].view(-1)[top] = 1.0
    return out, lum


def highlight_mask(x01, m, frac=0.03, red_sat=0.95):
    """[B,1,H,W] highlights: exactly the brightest `frac` of each image's retina pixels plus near-saturated red (as in
    train_module2_nonlong_pretrain.highlight_mask)."""
    out, _ = brightest_mask(x01, m, frac)
    return torch.maximum(out, ((x01[:, :1] > red_sat) & (m > 0)).float())


def retina(x_pm1, threshold=0.04):
    """[B,1,H,W] retina mask of a [-1, 1] batch: brightest channel above threshold on a 0-1 scale."""
    return (((x_pm1 + 1) / 2).amax(1, keepdim=True) > threshold).float()


def masked_mean(v, m):
    """Per-sample mean of a [B,1,H,W] map v over mask m."""
    return (v * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp(min=1.0)


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
    ap.add_argument("--grid-out", default=None, help="Optional .jpg: one held-out image per grade x (input + stages)")
    add_arch_arg(ap)
    add_render_args(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    check_run_config(args.generator, args.image_size, args.model_arch)
    G = Generator(conv_dim=args.g_conv_dim, c_dim=args.c_dim, repeat_num=args.g_repeat_num,
                  style_dim=args.style_dim, **arch_kwargs(args.model_arch)[0]).to(dev)
    load_checkpoint(G, args.generator, dev, args.model_arch)
    apply_saved_render_options(G, args.generator, args.g_upsample, args.g_padding)
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
    red_sum, chg_sum, n_k = np.zeros(args.c_dim), np.zeros(args.c_dim), np.zeros(args.c_dim)
    tint_sum, streak_sum = np.zeros(args.c_dim), np.zeros(args.c_dim)
    strong_sum, dark_sum = np.zeros(args.c_dim), np.zeros(args.c_dim)
    hl_sum, teal_sum = np.zeros(args.c_dim), np.zeros(args.c_dim)
    own_chg, own_n = 0.0, 0
    own_flip = {n: 0 for n in graders}
    for b in range(0, len(idx), args.batch_size):
        batch = [ds[i] for i in idx[b:b + args.batch_size]]
        x = torch.stack([s["image"] for s in batch]).to(dev)
        mask = torch.stack([s["mask"] for s in batch]).to(dev)
        src = torch.tensor([s["grade"] for s in batch], device=dev)
        real_grade = {}
        for name, (clf, mean, std, _p, _n) in graders.items():
            real_grade[name] = grade_batch(clf, x, mean, std)
            real_hits[name] += int((real_grade[name] == src).sum())
        m = retina(x)
        x01 = (x + 1) / 2
        disc, lum_in = brightest_mask(x01, m)
        hl = highlight_mask(x01, m)
        own_fake = torch.zeros_like(x)
        for k in range(args.c_dim):
            c = F.one_hot(torch.full((len(batch),), k, device=dev), args.c_dim).float()
            with torch.no_grad():
                fake = G(torch.cat([x, mask], 1), c)
            d = (fake + 1) / 2 - x01
            red_sum[k] += float(masked_mean(F.relu(d[:, :1] - d[:, 1:2]), m).sum())
            mean_r = masked_mean(d[:, :1], m).view(-1, 1, 1, 1)  # each image's own average shift
            mean_g = masked_mean(d[:, 1:2], m).view(-1, 1, 1, 1)
            tint_sum[k] += float(F.relu(mean_r - mean_g).sum())
            local_red = (d[:, :1] - mean_r) - (d[:, 1:2] - mean_g)
            streak_sum[k] += float(masked_mean(F.relu(local_red), m).sum())
            strong_sum[k] += float(masked_mean((local_red > STRONG_STREAK).float(), m).sum())
            f01 = (fake + 1) / 2
            lum_out = 0.299 * f01[:, :1] + 0.587 * f01[:, 1:2] + 0.114 * f01[:, 2:3]
            dark_sum[k] += float(masked_mean(F.relu(lum_in - lum_out), disc).sum())
            hl_sum[k] += float(masked_mean(F.relu(lum_in - lum_out), hl).sum())
            teal_sum[k] += float(masked_mean(F.relu((d[:, 1:2] + d[:, 2:3]) / 2 - d[:, :1]), hl).sum())
            chg_sum[k] += float(masked_mean(d.abs().mean(1, keepdim=True), m).sum())
            n_k[k] += len(batch)
            own = src == k
            own_fake[own] = fake[own]
            for name, (clf, mean, std, _p, _n) in graders.items():
                for pg in grade_batch(clf, fake, mean, std).tolist():
                    conf[name][k, pg] += 1
        own_chg += float(masked_mean(((own_fake + 1) / 2 - x01).abs().mean(1, keepdim=True), m).sum())
        own_n += len(batch)
        for name, (clf, mean, std, _p, _n) in graders.items():
            own_flip[name] += int((grade_batch(clf, own_fake, mean, std) != real_grade[name]).sum())

    if args.grid_out:  # first held-out image of each grade: input, then stages 0..c_dim-1
        from torchvision.utils import save_image
        rows = []
        for g in range(args.c_dim):
            hit = next((i for i in idx if int(ds.grades[i]) == g), None)
            if hit is None:
                continue
            s = ds[hit]
            x = s["image"].unsqueeze(0).to(dev)
            mk = s["mask"].unsqueeze(0).to(dev)
            row = [x]
            for k in range(args.c_dim):
                c = F.one_hot(torch.tensor([k], device=dev), args.c_dim).float()
                with torch.no_grad():
                    row.append(G(torch.cat([x, mk], 1), c))
            rows.append(torch.cat(row))
        if rows:
            os.makedirs(os.path.dirname(os.path.abspath(args.grid_out)), exist_ok=True)
            save_image((torch.cat(rows) + 1) / 2, args.grid_out, nrow=args.c_dim + 1, padding=4)
            print(f"✓ Sample grid: {args.grid_out}")
    out = {"generator": args.generator, "image_size": args.image_size, "n_real": len(idx), "graders": {},
           "g_upsample": args.g_upsample, "g_padding": args.g_padding,
           "redness": {str(k): float(red_sum[k] / max(n_k[k], 1)) for k in range(args.c_dim)},
           "change": {str(k): float(chg_sum[k] / max(n_k[k], 1)) for k in range(args.c_dim)},
           "no_change": {"mean_abs_change": own_chg / max(own_n, 1),
                         "grade_changed_rate": {n: own_flip[n] / max(own_n, 1) for n in graders}}}
    out["tint"] = {str(k): float(tint_sum[k] / max(n_k[k], 1)) for k in range(args.c_dim)}
    out["streak_redness"] = {str(k): float(streak_sum[k] / max(n_k[k], 1)) for k in range(args.c_dim)}
    out["strong_streak_area"] = {str(k): float(strong_sum[k] / max(n_k[k], 1)) for k in range(args.c_dim)}
    out["disc_darkening"] = {str(k): float(dark_sum[k] / max(n_k[k], 1)) for k in range(args.c_dim)}
    out["mean_strong_streak_area"] = float(np.mean(list(out["strong_streak_area"].values())))
    out["mean_disc_darkening"] = float(np.mean(list(out["disc_darkening"].values())))
    out["highlight_darkening"] = {str(k): float(hl_sum[k] / max(n_k[k], 1)) for k in range(args.c_dim)}
    out["teal_shift"] = {str(k): float(teal_sum[k] / max(n_k[k], 1)) for k in range(args.c_dim)}
    out["mean_highlight_darkening"] = float(np.mean(list(out["highlight_darkening"].values())))
    out["mean_teal_shift"] = float(np.mean(list(out["teal_shift"].values())))
    out["mean_redness"] = float(np.mean(list(out["redness"].values())))
    out["mean_tint"] = float(np.mean(list(out["tint"].values())))
    out["mean_streak_redness"] = float(np.mean(list(out["streak_redness"].values())))
    print("Image checks (inside the retina, 0-1 scale; compare between runs):")
    print("  redness added per requested stage: " + "  ".join(f"{k}: {v:.4f}" for k, v in out["redness"].items())
          + f"  | mean {out['mean_redness']:.4f}")
    print("  -> local streaks:                  " + "  ".join(f"{k}: {v:.4f}" for k, v in out["streak_redness"].items())
          + f"  | mean {out['mean_streak_redness']:.4f}")
    print("  -> whole-image red tint:           " + "  ".join(f"{k}: {v:.4f}" for k, v in out["tint"].items())
          + f"  | mean {out['mean_tint']:.4f}")
    print("  strong-streak area (share of retina): " + "  ".join(f"{k}: {v:.4f}" for k, v in out["strong_streak_area"].items())
          + f"  | mean {out['mean_strong_streak_area']:.4f}")
    print("  disc darkening (brightest 1 %):    " + "  ".join(f"{k}: {v:.4f}" for k, v in out["disc_darkening"].items())
          + f"  | mean {out['mean_disc_darkening']:.4f}")
    print("  highlight darkening (all bright):  " + "  ".join(f"{k}: {v:.4f}" for k, v in out["highlight_darkening"].items())
          + f"  | mean {out['mean_highlight_darkening']:.4f}  (teal shift {out['mean_teal_shift']:.4f})")
    print("  mean change per requested stage:   " + "  ".join(f"{k}: {v:.4f}" for k, v in out["change"].items()))
    print(f"  no-change test (own stage requested): mean change {out['no_change']['mean_abs_change']:.4f} | grade "
          "changed: " + ", ".join(f"{n} {r:.1%}" for n, r in out["no_change"]["grade_changed_rate"].items()))
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
