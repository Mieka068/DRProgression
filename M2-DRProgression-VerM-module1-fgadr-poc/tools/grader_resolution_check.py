"""
Round 4, Phase 0 diagnostic (item R4-0.2): how well does Module 1's grader read REAL Tianjin images
once they are shrunk to Module 2's resolution?

The cascade metric (evaluate_trajectory.py) re-grades synthesized images that are image_size px
(128 or 256), upsampled to the grader's 512 px. Round 3 got 1.5 % agreement with the target stage.
Part of that may be resolution, not the Generator: microaneurysms and small haemorrhages are
sub-pixel at 128 px. This script grades the real baseline and follow-up images of Tianjin eyes
(clinical ICDR grades) at full resolution and after a round trip through each --sizes value
(retina crop -> resize to s x s -> the grader's own 512 px resize), with each --grader:

    accuracy / quadratic kappa / within-one-grade vs. the clinical grade, and agreement with the
    same grader's full-resolution prediction.

The 128 / 256 rows are the ceiling a perfect Generator could reach on the cascade metric at that
size -- report the cascade agreement next to them. If 256 px is clearly better than 128 px, train
Module 2 at 256 px (MODULE2_IMAGE_SIZE in notebooks 06-10).

Usage:
    python tools/grader_resolution_check.py --tianjin-dir /content/data/retinal-dr-longitudinal \
        --tianjin-module1-cache .../module1_outputs_tianjin.pt \
        --grader eyepacs=.../eyepacs/saves/best_validation_weights.pt:eyepacs \
        --grader fgadr=.../fgadr/saves/final_weights.pt:fgadr \
        --sizes 128 256 --max-pairs 300 --out .../grader_resolution_check.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "module1"), os.path.join(ROOT, "module3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from apply_to_progression_data import CLASSIFIER_NORMS, load_classifier, predict_grade  # noqa: E402
from dataset import TianjinSurvivalDataset  # noqa: E402  (module3)
from fundus_preprocess import load_fundus  # noqa: E402


def scores(pred, true):
    from sklearn.metrics import cohen_kappa_score
    pred, true = np.asarray(pred), np.asarray(true)
    return {"n": int(len(true)), "accuracy": float((pred == true).mean()),
            "within_one_grade": float((np.abs(pred - true) <= 1).mean()),
            "quadratic_kappa": float(cohen_kappa_score(true, pred, weights="quadratic"))
            if len(set(true)) > 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tianjin-dir", required=True)
    ap.add_argument("--tianjin-module1-cache", required=True)
    ap.add_argument("--grader", action="append", required=True, metavar="NAME=PATH:NORM",
                    help="Repeatable; NORM is fgadr or eyepacs")
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--max-pairs", type=int, default=300, help="Eye-pairs to grade (random, seeded); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = TianjinSurvivalDataset(args.tianjin_dir, module1_cache_path=args.tianjin_module1_cache,
                                      image_size=128).manifest
    if args.max_pairs and len(manifest) > args.max_pairs:
        # keep every progressing eye (the cascade's cases), fill the rest at random
        prog = manifest[manifest["followup_icdr"] > manifest["baseline_icdr"]]
        rest = manifest.drop(prog.index).sample(n=max(0, args.max_pairs - len(prog)), random_state=args.seed)
        manifest = prog.head(args.max_pairs) if len(prog) >= args.max_pairs else pd.concat([prog, rest])
    images = []  # (path, clinical icdr, visit)
    for _, r in manifest.iterrows():
        images.append((os.path.join(args.tianjin_dir, r["baseline_path"]), int(r["baseline_icdr"]), "baseline"))
        images.append((os.path.join(args.tianjin_dir, r["followup_path"]), int(r["followup_icdr"]), "followup"))
    print(f"Grading {len(images)} real Tianjin images ({len(manifest)} eye-pairs) at full resolution and at "
          f"{args.sizes} px")

    out = {"n_pairs": int(len(manifest)), "n_images": len(images), "sizes": args.sizes, "graders": {}}
    for spec in args.grader:
        name, rest = spec.split("=", 1)
        path, norm = rest.rsplit(":", 1)
        mean, std = CLASSIFIER_NORMS[norm]
        clf = load_classifier(path, device=device)
        preds = {"full": [], **{s: [] for s in args.sizes}}
        true, visits = [], []
        for i, (p, g, visit) in enumerate(images):
            try:
                img, _box, _size = load_fundus(p, True)
            except Exception as e:
                print(f"  ⚠ {p}: {e}")
                continue
            true.append(g); visits.append(visit)
            preds["full"].append(predict_grade(clf, img, mean=mean, std=std, device=device, retina_crop=False))
            for s in args.sizes:
                small = img.resize((s, s), Image.BILINEAR)
                preds[s].append(predict_grade(clf, small, mean=mean, std=std, device=device, retina_crop=False))
            if (i + 1) % 100 == 0:
                print(f"  {name}: {i + 1}/{len(images)}")
        res = {}
        for key, pr in preds.items():
            res[str(key)] = {"vs_clinical": scores(pr, true),
                             "vs_clinical_followups": scores([q for q, v in zip(pr, visits) if v == "followup"],
                                                             [t for t, v in zip(true, visits) if v == "followup"]),
                             "agrees_with_full_res": float(np.mean(np.asarray(pr) == np.asarray(preds["full"])))}
        out["graders"][name] = {"checkpoint": path, "norm": norm, "results": res}
        print(f"\n{name} ({norm}) -- vs Tianjin's clinical grades:")
        print(f"  {'resolution':<11}{'accuracy':>9}{'kappa':>8}{'within-1':>10}{'= full-res':>12}")
        for key in ["full"] + args.sizes:
            r = res[str(key)]
            k = r["vs_clinical"]["quadratic_kappa"]
            print(f"  {str(key) + (' px' if key != 'full' else ''):<11}{r['vs_clinical']['accuracy']:>9.3f}"
                  f"{(k if k is not None else float('nan')):>8.3f}{r['vs_clinical']['within_one_grade']:>10.3f}"
                  f"{r['agrees_with_full_res']:>12.3f}")
    if len(args.sizes) >= 2:
        lo, hi = min(args.sizes), max(args.sizes)
        gain = {n: (g["results"][str(hi)]["vs_clinical"]["quadratic_kappa"] or 0)
                - (g["results"][str(lo)]["vs_clinical"]["quadratic_kappa"] or 0) for n, g in out["graders"].items()}
        out["kappa_gain_high_vs_low"] = gain
        best = max(gain.values())
        out["recommendation"] = (f"train Module 2 at {hi} px (kappa +{best:.2f} over {lo} px)" if best >= 0.05
                                 else f"{hi} px barely helps the grader (kappa +{best:.2f}) -- {lo} px is enough")
        print(f"\nRecommendation: {out['recommendation']}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ Wrote {args.out}")


if __name__ == "__main__":
    main()
