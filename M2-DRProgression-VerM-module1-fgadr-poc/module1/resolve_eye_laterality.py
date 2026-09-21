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

Usage:
    python resolve_eye_laterality.py \
        --dataset-dir ./retinal-dr-longitudinal \
        --manifest corrected_manifest.csv \
        --out laterality_resolved.csv

After running, check the "same eye resolved twice" sanity check printed below. If more than
~5% of two-row patients resolve to the same eye on both rows, the heuristic needs tuning
(larger --margin, or a smaller downscale for more precision).
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd


def find_optic_disc_x_fraction(image_path, downscale=4):
    """
    Returns the optic disc's horizontal position as a fraction of image width (0=left edge,
    1=right edge). Approach: the optic disc is the brightest, largest connected bright region
    in the green channel (standard fundus-image OD-localization heuristic -- the disc has the
    highest local contrast/brightness of any retinal structure). Not a trained model;
    intentionally simple and fast enough to run over ~1,100 images in a few minutes on CPU.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // downscale), max(1, h // downscale)))
    green = small[:, :, 1]  # green channel has the best disc/vessel contrast
    blurred = cv2.GaussianBlur(green, (15, 15), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    # Guard against a blown-out/glare frame: fall back to brightest-region centroid via
    # thresholding the top 1% of pixel intensities instead of a single max pixel.
    thresh_val = np.percentile(blurred, 99)
    ys, xs = np.where(blurred >= thresh_val)
    if len(xs) == 0:
        x_frac = max_loc[0] / small.shape[1]
    else:
        x_frac = float(np.mean(xs)) / small.shape[1]
    return x_frac


def resolve_laterality(x_frac, margin=0.1):
    """
    x_frac < 0.5 - margin  => OD (right eye), disc left of center
    x_frac > 0.5 + margin  => OS (left eye), disc right of center
    otherwise              => "uncertain" (near-center detection, don't guess)
    """
    if x_frac is None:
        return "uncertain"
    if x_frac < 0.5 - margin:
        return "OD"
    if x_frac > 0.5 + margin:
        return "OS"
    return "uncertain"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--manifest", default="corrected_manifest.csv")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--out", default="laterality_resolved.csv")
    args = parser.parse_args()

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
        x_frac = find_optic_disc_x_fraction(path, downscale=args.downscale)
        lateralities.append(resolve_laterality(x_frac, margin=args.margin))
        x_fracs.append(x_frac)

    manifest["disc_x_fraction"] = x_fracs
    manifest["eye"] = lateralities  # "OD", "OS", or "uncertain"

    n_uncertain = (manifest["eye"] == "uncertain").sum()
    print(f"Resolved {len(manifest) - n_uncertain}/{len(manifest)} eyes "
          f"({n_uncertain} uncertain -- see fallback policy in tianjin_dataset.py)")

    out_path = os.path.join(args.dataset_dir, args.out)
    manifest.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    # Sanity check: for patients with 2 rows, the two should resolve to different eyes.
    two_row_patients = manifest.groupby("patient_id").filter(lambda g: len(g) == 2)
    if len(two_row_patients) == 0:
        print("No patients with exactly 2 baseline rows found -- skipping same-eye sanity check.")
        return
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
            "--downscale and re-run before using this output for training."
        )


if __name__ == "__main__":
    main()
