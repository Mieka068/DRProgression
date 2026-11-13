"""
Why does the FGADR grader score far below chance on Tianjin (accuracy ~0.02, kappa ~0) when it gets
kappa 0.81 on FGADR's own test set? This script tests the explanation in
module1/FGADR_GRADER_DIAGNOSIS.md: FGADR's grade labels are confounded with the image source.

FGADR file names end in _1 / _2 / _3 (the image source). In DR_Seg_Grading_Label.csv every _2 and _3
image is grade 3 or 4, and all grade 0-2 images are _1 (see `--labels-only`, which needs no images).
A grader can therefore score well on FGADR by recognising the source rather than the lesions; on
Tianjin, which is none of those sources, such a grader has nothing to go on.

Parts (each prints a confusion matrix: rows = true ICDR grade, columns = predicted):
  A. labels only: grade x source table, and the FGADR test score of a rule that sees ONLY the source.
  B. --fgadr-root: the grader on FGADR's test split (DRG-Net's released 1500/342 split), split by
     source. A source shortcut shows up as: grade-3/4 images from source _1 often predicted 0-2,
     grade-3/4 images from _2/_3 almost always predicted 3/4.
  C. --tianjin-dir: the grader on Tianjin vs. its clinical grades (per eye, as in
     compare_graders_tianjin.py), with the predicted-grade histogram and mean softmax, with and
     without the retina crop (rules out the crop as the cause).

Usage (Colab, after notebook 04_05's setup cells):
    python module1/diagnose_fgadr_grader.py --labels-only
    python module1/diagnose_fgadr_grader.py \
        --checkpoint ".../drgnet_replication_prefinals/classification/fgadr/saves/final_weights.pt" \
        --fgadr-root /content/FGADR-Seg-set_Release/Seg-set \
        --tianjin-dir /content/data/retinal-dr-longitudinal --out fgadr_grader_diagnosis.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_CSV = os.path.join(_HERE, "data", "drgnet_fgadr_classification_split.csv")
NUM_STAGES = 5


def confusion(true, pred):
    cm = np.zeros((NUM_STAGES, NUM_STAGES), dtype=int)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def summary(true, pred):
    from sklearn.metrics import cohen_kappa_score

    true, pred = np.asarray(true, int), np.asarray(pred, int)
    kappa = cohen_kappa_score(true, pred, weights="quadratic") if len(set(true) | set(pred)) > 1 else float("nan")
    return {"n": int(len(true)), "accuracy": float((true == pred).mean()), "quadratic_kappa": float(kappa),
            "within_one_grade": float((np.abs(true - pred) <= 1).mean()),
            "predicted_counts": np.bincount(pred, minlength=NUM_STAGES).tolist(),
            "confusion": confusion(true, pred).tolist()}


def show(title, s):
    print(f"\n{title}: n={s['n']}  accuracy {s['accuracy']:.3f}  qwk {s['quadratic_kappa']:.3f}  "
          f"within-one {s['within_one_grade']:.3f}")
    print("  predicted-grade counts:", s["predicted_counts"])
    print("  confusion (rows = true 0-4, cols = predicted 0-4):")
    for k, row in enumerate(s["confusion"]):
        print(f"    {k}: " + " ".join(f"{v:5d}" for v in row))


def fgadr_labels():
    df = pd.read_csv(SPLIT_CSV)
    df["source"] = df["image"].str.extract(r"_(\d+)\.")[0].astype(int)
    return df


def part_a(df):
    print("=" * 80 + "\nA. FGADR grade x image source (file-name suffix), all 1842 images")
    print(pd.crosstab(df["source"], df["grade"], margins=True).to_string())
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    rule = train.groupby("source")["grade"].agg(lambda s: int(s.mode()[0])).to_dict()
    s = summary(test["grade"], test["source"].map(rule))
    show(f"Source-only rule {rule} (sees no pixels) on FGADR test", s)
    return {"grade_by_source": pd.crosstab(df["source"], df["grade"]).to_dict(), "source_only_rule": s}


def part_b(df, model, fgadr_root, device, predict_grade):
    print("=" * 80 + "\nB. Grader on FGADR's test split, by image source (no retina crop, as in training)")
    test = df[df["split"] == "test"].copy()
    img_dir = os.path.join(fgadr_root, "Original_Images")
    test["pred"] = [predict_grade(model, os.path.join(img_dir, n), device=device, retina_crop=False)
                    for n in test["image"]]
    out = {"all": summary(test["grade"], test["pred"])}
    show("all sources", out["all"])
    for name, sel in (("source _1", test["source"] == 1), ("sources _2/_3", test["source"] > 1)):
        out[name] = summary(test.loc[sel, "grade"], test.loc[sel, "pred"])
        show(name, out[name])
    severe = test[test["grade"] >= 3]
    for name, sel in (("_1", severe["source"] == 1), ("_2/_3", severe["source"] > 1)):
        p = severe.loc[sel, "pred"]
        print(f"  grade-3/4 images from source {name}: {len(p)}, predicted >= 3: {(p >= 3).mean():.2f}")
    return out


def part_c(model, tianjin_dir, device, predict_grade, limit):
    from compare_graders_tianjin import load_rows

    print("=" * 80 + "\nC. Grader on Tianjin vs. clinical grades (per eye, laterality-resolved rows)")
    rows = [r for r in load_rows(tianjin_dir, limit) if r["eye_grade"] is not None]
    out = {}
    for crop in (True, False):
        true, pred, probs = [], [], []
        for r in rows:
            try:
                g, p = predict_grade(model, r["path"], device=device, return_probs=True, retina_crop=crop)
            except Exception as e:  # unreadable image
                print(f"  ⚠ {r['path']}: {e}")
                continue
            true.append(r["eye_grade"])
            pred.append(g)
            probs.append(p)
        key = "retina_crop" if crop else "no_crop"
        out[key] = summary(true, pred)
        out[key]["mean_softmax"] = np.mean(probs, axis=0).round(3).tolist()
        out[key]["clinical_counts"] = np.bincount(true, minlength=NUM_STAGES).tolist()
        show(f"Tianjin, {key}", out[key])
        print("  clinical-grade counts:", out[key]["clinical_counts"], " mean softmax:", out[key]["mean_softmax"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-only", action="store_true", help="Part A only (no checkpoint or images)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--fgadr-root", help="FGADR Seg-set folder with Original_Images/ (part B)")
    ap.add_argument("--tianjin-dir", help="Tianjin folder as in compare_graders_tianjin.py (part C)")
    ap.add_argument("--limit-patients", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = fgadr_labels()
    result = {"A_labels": part_a(df)}
    if not args.labels_only:
        if not args.checkpoint:
            ap.error("--checkpoint is required unless --labels-only")
        import torch

        sys.path.insert(0, _HERE)
        sys.path.insert(0, os.path.join(_HERE, ".."))
        sys.path.insert(0, os.path.join(_HERE, "..", "module3"))
        from apply_to_progression_data import load_classifier, predict_grade, set_classifier_norm

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        set_classifier_norm("fgadr")
        model = load_classifier(args.checkpoint, device=device)
        if args.fgadr_root:
            result["B_fgadr_test"] = part_b(df, model, args.fgadr_root, device, predict_grade)
        if args.tianjin_dir:
            result["C_tianjin"] = part_c(model, args.tianjin_dir, device, predict_grade, args.limit_patients)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✓ Wrote {args.out}")


if __name__ == "__main__":
    main()
