"""
Which Module 1 grader fits the pipeline's data? Scores every given grader checkpoint against
Tianjin's CLINICAL grades (baseline and 2-year follow-up images, Organized_Data of Patients.xlsx) --
the images the pipeline actually grades -- so FGADR- and EyePACS-trained graders are compared on the
same images. Their own test-set numbers (notebook 11) can't be compared: different datasets, and
FGADR's test set also picks its best epoch.

Two views, both reported for baseline, follow-up and the two visits pooled:
  * patient (PRIMARY): the worse-eye ("At-risk Eye Grade") clinical grade vs. the highest grade the
    grader gives that patient's images at that visit. Needs no left/right resolution, so it doesn't
    depend on module1/resolve_eye_laterality.py being right.
  * eye: each image's eye-specific clinical grade, for rows whose laterality_resolved.csv entry is
    OD/OS (skipped if that file is missing). Only as trustworthy as the laterality.
Metrics: n, accuracy, quadratic-weighted kappa (the DR-grading standard), within-one-grade accuracy,
per-grade recall and the confusion matrix (rows = clinical ICDR grade, columns = predicted).
Excluded clinical grades (6 = after laser, 7 = ungradable) and missing ones are skipped.

Usage:
    python module1/compare_graders_tianjin.py --tianjin-dir /content/data/retinal-dr-longitudinal \
        --grader fgadr=.../classification/fgadr/saves/final_weights.pt \
        --grader eyepacs=.../classification/eyepacs/saves/best_validation_weights.pt \
        --out grader_comparison_tianjin.json
The name before '=' picks the input normalisation (a name starting with 'eyepacs' -> EyePACS stats,
anything else -> FGADR's); use name:norm=path to set it explicitly (norm: fgadr / eyepacs).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "module3"))
from apply_to_progression_data import (  # noqa: E402
    CLASSIFIER_NORMS,
    load_classifier,
    predict_grade,
    set_classifier_norm,
)
from tianjin_dataset import EXCLUDED_GRADES, tianjin_grade_to_icdr  # noqa: E402

NUM_STAGES = 5


def icdr(raw):
    if pd.isna(raw) or int(raw) in EXCLUDED_GRADES:
        return None
    return int(np.clip(tianjin_grade_to_icdr(int(raw)), 0, NUM_STAGES - 1))


def metrics(true, pred):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    true, pred = np.asarray(true, int), np.asarray(pred, int)
    if len(true) == 0:
        return {"n": 0}
    cm = confusion_matrix(true, pred, labels=list(range(NUM_STAGES)))
    per_n = cm.sum(1)
    return {
        "n": int(len(true)),
        "accuracy": float((true == pred).mean()),
        "quadratic_kappa": float(cohen_kappa_score(true, pred, weights="quadratic"))
        if len(set(true) | set(pred)) > 1 else None,
        "within_one_grade": float((np.abs(true - pred) <= 1).mean()),
        "per_grade_recall": {str(k): (float(cm[k, k] / per_n[k]) if per_n[k] else None) for k in range(NUM_STAGES)},
        "clinical_grade_counts": per_n.tolist(),
        "confusion": cm.tolist(),
    }


def load_rows(tianjin_dir, limit=None):
    """One row per image (baseline and follow-up) with its patient, visit, eye and clinical grades."""
    from dataset import TianjinSurvivalDataset  # module3/dataset.py -- reads both xlsx sheets

    labels = TianjinSurvivalDataset._load_progression_labels(tianjin_dir)
    manifest = pd.read_csv(os.path.join(tianjin_dir, "corrected_manifest.csv"))
    manifest["patient_id"] = manifest["patient_id"].astype(str).str.strip()
    lat_path = os.path.join(tianjin_dir, "laterality_resolved.csv")
    if os.path.isfile(lat_path):
        manifest = manifest.merge(pd.read_csv(lat_path)[["key", "eye"]], on="key", how="left")
    else:
        print(f"  (no {lat_path} -- per-eye view skipped)")
        manifest["eye"] = np.nan
    manifest = manifest.merge(labels, on="patient_id", how="inner")
    if limit:
        manifest = manifest[manifest["patient_id"].isin(sorted(manifest["patient_id"].unique())[:limit])]
    rows = []
    for _, r in manifest.iterrows():
        for visit, path_col, prefix in (("baseline", "baseline_path", "baseline_"), ("followup", "followup_path", "followup_")):
            eye = r["eye"] if r["eye"] in ("OD", "OS") else None
            rows.append({
                "patient_id": r["patient_id"], "visit": visit,
                "path": os.path.join(tianjin_dir, r[path_col]),
                "worse_eye_grade": icdr(r[prefix + "worse_eye_grade_raw"]),
                "eye_grade": icdr(r[f"{prefix}{eye.lower()}_grade_raw"]) if eye else None,
            })
    return rows


def evaluate(rows, preds):
    """{view: {visit: metrics}} for one grader's per-image predictions (None = unreadable image)."""
    out = {"patient": {}, "eye": {}}
    for visit in ("baseline", "followup", "both"):
        sel = [(r, p) for r, p in zip(rows, preds) if p is not None and (visit == "both" or r["visit"] == visit)]
        # patient view: max prediction over the patient's images at a visit vs. the worse-eye grade
        groups = {}
        for r, p in sel:
            if r["worse_eye_grade"] is None:
                continue
            g = groups.setdefault((r["patient_id"], r["visit"]), [r["worse_eye_grade"], []])
            g[1].append(p)
        out["patient"][visit] = metrics([t for t, _ in groups.values()], [max(ps) for _, ps in groups.values()])
        eye = [(r["eye_grade"], p) for r, p in sel if r["eye_grade"] is not None]
        out["eye"][visit] = metrics([t for t, _ in eye], [p for _, p in eye])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tianjin-dir", required=True)
    ap.add_argument("--grader", action="append", required=True, metavar="NAME[:NORM]=CHECKPOINT")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-patients", type=int, default=None, help="Debug: first N patients only")
    ap.add_argument("--out", default="grader_comparison_tianjin.json")
    args = ap.parse_args()

    rows = load_rows(args.tianjin_dir, args.limit_patients)
    n_pat = len({r["patient_id"] for r in rows})
    print(f"Tianjin: {len(rows)} images ({n_pat} patients, baseline + follow-up) with clinical grades")

    results = {}
    for spec in args.grader:
        name, path = spec.split("=", 1)
        name, _, norm = name.partition(":")
        norm = norm or ("eyepacs" if name.lower().startswith("eyepacs") else "fgadr")
        if norm not in CLASSIFIER_NORMS:
            ap.error(f"unknown normalisation {norm!r} for {name}")
        if not os.path.isfile(path):
            print(f"⚠ {name}: {path} not found -- skipped")
            continue
        set_classifier_norm(norm)
        model = load_classifier(path, device=args.device)
        preds = []
        for k, r in enumerate(rows):
            try:
                preds.append(predict_grade(model, r["path"], device=args.device))
            except Exception as e:  # unreadable image
                print(f"  ⚠ {r['path']}: {e}")
                preds.append(None)
            if (k + 1) % 500 == 0:
                print(f"  {name}: {k + 1}/{len(rows)} images")
        results[name] = {"checkpoint": path, "normalisation": norm, **evaluate(rows, preds)}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAgreement with Tianjin's clinical grades (quadratic kappa / accuracy / within one grade):")
    print(f"  {'grader':<28}{'view':<9}{'baseline':>22}{'follow-up':>22}{'both':>22}")
    for name, r in results.items():
        for view in ("patient", "eye"):
            cells = []
            for visit in ("baseline", "followup", "both"):
                m = r[view][visit]
                cells.append("--" if not m.get("n") else
                             f"{(m['quadratic_kappa'] or 0):.3f}/{m['accuracy']:.2f}/{m['within_one_grade']:.2f}")
            print(f"  {name:<28}{view:<9}" + "".join(f"{c:>22}" for c in cells))
    def _key(n):  # patient-level kappa over both visits, ties broken by accuracy
        m = results[n]["patient"]["both"]
        return (m.get("quadratic_kappa") if m.get("quadratic_kappa") is not None else -1, m.get("accuracy", 0))
    ranked = sorted(results, key=_key, reverse=True)
    summary = {"ranking_by_patient_kappa_both_visits": ranked,
               "recommended": ranked[0] if ranked else None,
               "n_images": len(rows), "n_patients": n_pat}
    if ranked:
        best = results[ranked[0]]["patient"]["both"]
        print(f"\nHighest patient-level kappa (both visits): {ranked[0]} "
              f"(kappa {best.get('quadratic_kappa') or 0:.3f}, accuracy {best['accuracy']:.2f}, n={best['n']}). "
              "The per-eye rows depend on the laterality file; read them with that in mind.")
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "graders": results}, f, indent=2)
    print(f"✓ Wrote {args.out}")


if __name__ == "__main__":
    main()
