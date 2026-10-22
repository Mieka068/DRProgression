"""
Resolve OD (right eye) vs. OS (left eye) for each Tianjin baseline image by locating the
optic disc and checking which half of the image it falls in (standard fundus photography
convention: optic disc left-of-center => OD/right eye; right-of-center => OS/left eye).

Organized_Data of Patients.xlsx stores DR grade per eye (separate OS Grade / OD Grade /
At-risk Eye Grade columns), not per patient. corrected_manifest.csv's rows (one per
baseline<->follow-up pair) carry no OD/OS marker, and baseline filenames give no laterality
hint either.

This is a one-time preprocessing pass, run before tianjin_dataset.py / module3/dataset.py can
resolve real per-eye grades -- both raise a FileNotFoundError pointing here if
laterality_resolved.csv doesn't exist yet in the dataset directory.

Re-tuned on branch AllModuleEdits-PreFinals (CODE_FIXES item 7.1; ~7 % of two-image patients
failed the same-eye check before, threshold 5 %):
  * the disc position is measured inside the retina's bounding box (field-of-view mask), not the
    whole frame, so black borders / off-centre fields of view don't shift it;
  * the disc is the largest bright blob (top --percentile of the blurred green channel) inside an
    eroded field of view -- rim glare is excluded, and scattered bright pixels no longer pull the
    estimate to the centre (the earlier mean of all top-1 % pixels);
  * pair constraint (--pair-constraint, on by default): a patient's two rows are two different
    eyes, so when the two disc positions disagree with that (same eye, or one uncertain), the
    row whose disc is further from the centre keeps its side and the other gets the opposite eye
    (eye_source = 'pair'); 'disc' = from its own image. --no-pair-constraint = per image only.
  * --disc-left-is {OD,OS}: which eye a left-of-centre disc means. 'OD' is the convention this
    script has always used. NOTE: in a standard (non-mirrored) fundus photograph the right eye's
    disc usually appears on the RIGHT, i.e. 'OS' here -- the two-row sanity check cannot detect a
    fully swapped convention (both rows still differ), so validate on EyePACS first:
  * --validate-eyepacs-dir: EyePACS file names carry the truth (<id>_left / <id>_right), so this
    runs the heuristic over --validate-n of them for a grid of --margin values, under both
    conventions, and prints / writes (--validate-out) accuracy, uncertain rate and the
    convention that fits. Run it before resolving Tianjin and set --margin / --disc-left-is
    from it (Tianjin uses the same standard fundus-camera views).

Usage:
    python resolve_eye_laterality.py --validate-eyepacs-dir /content/data/eyepacs \
        --validate-out ./laterality_eyepacs_validation.json
    python resolve_eye_laterality.py \
        --dataset-dir ./retinal-dr-longitudinal \
        --manifest corrected_manifest.csv \
        --out laterality_resolved.csv

After running, check the "same eye resolved twice" sanity check printed below (computed BEFORE
the pair constraint, so it still measures the heuristic). If more than ~5% of two-row patients
resolve to the same eye on both rows, the heuristic needs tuning (larger --margin, or a smaller
downscale for more precision).
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd


def find_optic_disc_x_fraction(image_path, downscale=4, percentile=99.0, legacy=False):
    """
    Returns the optic disc's horizontal position as a fraction of the retina's width (0 = left
    edge of the field of view, 1 = right edge). The optic disc is the brightest, largest bright
    region in the green channel (standard fundus-image OD-localization heuristic). Not a trained
    model; fast enough for ~1,100 images in a few minutes on CPU. legacy=True is the earlier
    estimate (mean of the top-1 % pixels of the whole frame).
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // downscale), max(1, h // downscale)))
    green = small[:, :, 1]  # green channel has the best disc/vessel contrast
    if legacy:
        blurred = cv2.GaussianBlur(green, (15, 15), 0)
        _, _max_val, _, max_loc = cv2.minMaxLoc(blurred)
        thresh_val = np.percentile(blurred, 99)
        ys, xs = np.where(blurred >= thresh_val)
        return (max_loc[0] if len(xs) == 0 else float(np.mean(xs))) / small.shape[1]

    fov = (green > 10).astype(np.uint8)
    cols = fov.sum(axis=0)
    xs_fov = np.nonzero(cols > 0.02 * max(cols.max(), 1))[0]
    if fov.sum() < 0.05 * fov.size or len(xs_fov) < 2:
        return None
    x0, x1 = xs_fov[0], xs_fov[-1] + 1
    # Erode the field of view so the bright rim / flash reflections at its edge are not candidates.
    k = max(3, int(0.06 * (x1 - x0)) | 1)
    inner = cv2.erode(fov, np.ones((k, k), np.uint8))
    sigma = max(1.0, 0.02 * (x1 - x0))
    blurred = cv2.GaussianBlur(green.astype(np.float32), (0, 0), sigma)
    vals = blurred[inner > 0]
    if vals.size == 0:
        return None
    bright = ((blurred >= np.percentile(vals, percentile)) & (inner > 0)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return None
    # Largest blob; ties broken by brightness.
    best = max(range(1, n), key=lambda i: (stats[i, cv2.CC_STAT_AREA], blurred[labels == i].mean()))
    return float((centroids[best][0] - x0) / (x1 - x0))


def resolve_laterality(x_frac, margin=0.1, disc_left_is="OD"):
    """
    x_frac < 0.5 - margin  => disc left of center  => disc_left_is (default OD)
    x_frac > 0.5 + margin  => disc right of center => the other eye
    otherwise              => "uncertain" (near-center detection, don't guess)
    """
    other = "OS" if disc_left_is == "OD" else "OD"
    if x_frac is None:
        return "uncertain"
    if x_frac < 0.5 - margin:
        return disc_left_is
    if x_frac > 0.5 + margin:
        return other
    return "uncertain"


def apply_pair_constraint(manifest):
    """Two rows of one patient are two different eyes: where the per-image calls don't say so,
    the row with the more off-centre disc keeps its side and the other gets the opposite eye."""
    manifest["eye_source"] = np.where(manifest["eye"] == "uncertain", "none", "disc")
    flip = {"OD": "OS", "OS": "OD"}
    n_fixed = 0
    for _, g in manifest.groupby("patient_id"):
        if len(g) != 2 or g["disc_x_fraction"].isna().any():
            continue
        eyes = g["eye"].tolist()
        if eyes[0] != eyes[1] and "uncertain" not in eyes:
            continue
        conf = (g["disc_x_fraction"] - 0.5).abs()
        anchor, other = conf.idxmax(), conf.idxmin()
        if manifest.at[anchor, "eye"] == "uncertain":
            continue  # neither row is confident -- leave both to the worse-eye fallback
        manifest.at[other, "eye"] = flip[manifest.at[anchor, "eye"]]
        manifest.at[other, "eye_source"] = "pair"
        n_fixed += 1
    return n_fixed


def validate_on_eyepacs(eyepacs_dir, n=500, margins=(0.0, 0.05, 0.1, 0.15, 0.2), downscale=4,
                        percentile=99.0, seed=0):
    """Accuracy of the disc heuristic against EyePACS' _left / _right file names."""
    files = []
    for root, _dirs, names in os.walk(eyepacs_dir):
        for f in names:
            stem = os.path.splitext(f)[0].lower()
            if f.lower().endswith((".jpeg", ".jpg", ".png")) and stem.endswith(("_left", "_right")):
                files.append((os.path.join(root, f), "OS" if stem.endswith("_left") else "OD"))
    if not files:
        raise SystemExit(f"No <id>_left / <id>_right images under {eyepacs_dir}")
    rng = np.random.RandomState(seed)
    files = [files[i] for i in rng.choice(len(files), size=min(n, len(files)), replace=False)]
    out = {"n_images": len(files), "downscale": downscale, "percentile": percentile, "results": []}
    for legacy in (False, True):
        xs = [find_optic_disc_x_fraction(p, downscale, percentile, legacy=legacy) for p, _ in files]
        truth = [t for _, t in files]
        for convention in ("OD", "OS"):
            for m in margins:
                pred = [resolve_laterality(x, m, convention) for x in xs]
                decided = [(p, t) for p, t in zip(pred, truth) if p != "uncertain"]
                acc = float(np.mean([p == t for p, t in decided])) if decided else None
                out["results"].append({"estimator": "legacy" if legacy else "fov_blob", "disc_left_is": convention,
                                       "margin": m, "accuracy_on_decided": acc,
                                       "uncertain_rate": 1 - len(decided) / len(files)})
    ok = [r for r in out["results"] if r["accuracy_on_decided"] is not None]
    best = max(ok, key=lambda r: (r["accuracy_on_decided"] * (1 - r["uncertain_rate"]), r["accuracy_on_decided"]))
    out["best"] = best
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=None, help="Tianjin dataset (omit with --validate-eyepacs-dir)")
    parser.add_argument("--manifest", default="corrected_manifest.csv")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="Brightness percentile (inside the field of view) for disc candidates")
    parser.add_argument("--disc-left-is", choices=["OD", "OS"], default="OD",
                        help="Eye a left-of-centre disc means (see the module docstring; validate first)")
    parser.add_argument("--legacy-estimator", action="store_true",
                        help="Earlier disc estimate (mean of the frame's top-1%% pixels)")
    parser.add_argument("--no-pair-constraint", dest="pair_constraint", action="store_false")
    parser.add_argument("--out", default="laterality_resolved.csv")
    parser.add_argument("--validate-eyepacs-dir", default=None,
                        help="Validate the heuristic on EyePACS (_left/_right names) instead of / before resolving")
    parser.add_argument("--validate-n", type=int, default=500)
    parser.add_argument("--validate-out", default=None)
    args = parser.parse_args()

    if args.validate_eyepacs_dir:
        val = validate_on_eyepacs(args.validate_eyepacs_dir, n=args.validate_n, downscale=args.downscale,
                                  percentile=args.percentile)
        print(pd.DataFrame(val["results"]).to_string(index=False))
        b = val["best"]
        print(f"\nBest on EyePACS (n={val['n_images']}): estimator={b['estimator']}, --disc-left-is "
              f"{b['disc_left_is']}, --margin {b['margin']} -> accuracy {b['accuracy_on_decided']:.1%} "
              f"on decided images, {b['uncertain_rate']:.1%} uncertain")
        if b["disc_left_is"] != args.disc_left_is:
            print(f"⚠ EyePACS fits --disc-left-is {b['disc_left_is']}, not {args.disc_left_is} -- the OD/OS "
                  "labels would be swapped with the current convention. Check a few images by eye, then pass "
                  f"--disc-left-is {b['disc_left_is']} when resolving Tianjin.")
        if args.validate_out:
            with open(args.validate_out, "w") as f:
                json.dump(val, f, indent=2)
            print(f"Wrote {args.validate_out}")
        if not args.dataset_dir:
            return
    if not args.dataset_dir:
        parser.error("--dataset-dir is required unless only validating")

    manifest_path = os.path.join(args.dataset_dir, args.manifest)
    manifest = pd.read_csv(manifest_path)
    for required_col in ("key", "patient_id", "baseline_path"):
        if required_col not in manifest.columns:
            raise KeyError(
                f"{args.manifest} is missing expected column '{required_col}'. Actual "
                f"columns: {list(manifest.columns)}."
            )

    lateralities, x_fracs = [], []
    for _, row in manifest.iterrows():
        path = os.path.join(args.dataset_dir, row["baseline_path"])
        x_frac = find_optic_disc_x_fraction(path, downscale=args.downscale, percentile=args.percentile,
                                            legacy=args.legacy_estimator)
        lateralities.append(resolve_laterality(x_frac, margin=args.margin, disc_left_is=args.disc_left_is))
        x_fracs.append(x_frac)

    manifest["disc_x_fraction"] = x_fracs
    manifest["eye"] = lateralities  # "OD", "OS", or "uncertain"

    n_uncertain = (manifest["eye"] == "uncertain").sum()
    print(f"Resolved {len(manifest) - n_uncertain}/{len(manifest)} eyes from their own disc position "
          f"({n_uncertain} uncertain -- see fallback policy in tianjin_dataset.py)")

    # Sanity check: for patients with 2 rows, the two should resolve to different eyes (measured on
    # the per-image calls, before the pair constraint below rewrites them).
    two_row_patients = manifest.groupby("patient_id").filter(lambda g: len(g) == 2)
    if len(two_row_patients) == 0:
        print("No patients with exactly 2 baseline rows found -- skipping same-eye sanity check.")
    else:
        same_eye = two_row_patients.groupby("patient_id")["eye"].apply(
            lambda s: s.iloc[0] == s.iloc[1] and "uncertain" not in s.values
        )
        n_same = int(same_eye.sum())
        n_total = len(same_eye)
        rate = n_same / n_total if n_total else 0.0
        print(f"Patients with 2 baseline images resolving to the SAME eye "
              f"(should be rare -- flags heuristic problems): {n_same}/{n_total} ({rate:.1%})")
        if rate > 0.05:
            print(
                "⚠ Same-eye disagreement rate is above 5% -- try a larger --margin or smaller "
                "--downscale (or validate on EyePACS: --validate-eyepacs-dir) and re-run before "
                "using this output for training."
            )

    if args.pair_constraint:
        n_fixed = apply_pair_constraint(manifest)
        n_uncertain = (manifest["eye"] == "uncertain").sum()
        print(f"Pair constraint: {n_fixed} rows set to the opposite of their patient's other eye "
              f"-> {n_uncertain} uncertain left")

    out_path = os.path.join(args.dataset_dir, args.out)
    manifest.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
