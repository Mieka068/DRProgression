"""
Lesion Burden Score (LBS) computation -- this is NOT part of DRG-Net; it's the manuscript's
own addition (Sec 3.2.6): LBS = (total lesion pixels) / (total retinal area pixels), used to
stratify patients into low/medium/high burden strata (33rd/66th percentile thresholds) for
Module 3's survival analysis, and as part of Module 1's output package forwarded to Module 2.

Works directly on DRG-Net's/FGADR's/IDRiD's own mask files, or on predicted mask arrays coming
back from a trained segmentation model -- same function either way.

Standalone self-test (no torch needed, just PIL/numpy/opencv): running this file directly
loads one FGADR sample, combines its 4 lesion masks, estimates the retinal FOV, computes LBS,
and saves a visualization PNG -- this is the "show one sample data" artifact for the
presentation's data section.
"""
import os

import cv2
import numpy as np
from PIL import Image


def estimate_retinal_fov_mask(image, threshold: int = 10) -> np.ndarray:
    """
    Fundus photos have a black background outside the circular field of view (FOV); a simple
    intensity threshold on the green channel (highest-contrast channel for fundus images)
    separates retina from background. Good enough for a POC LBS denominator -- swap for a
    learned FOV segmenter later if the threshold proves too coarse on noisy images.

    Args:
        image: path to an RGB image, or an already-loaded HxWx3 (RGB) or HxW array.
    Returns:
        HxW uint8 binary mask (1 = retina, 0 = background).
    """
    if isinstance(image, str):
        img = cv2.imread(image)  # BGR
        green = img[:, :, 1]
    elif image.ndim == 3:
        green = image[:, :, 1]
    else:
        green = image
    return (green > threshold).astype(np.uint8)


def load_binary_mask(path: str, threshold: int = 127) -> np.ndarray:
    """Load a lesion mask file (PNG/TIF, any bit depth) as a 0/1 uint8 array."""
    m = np.array(Image.open(path).convert("L"))
    return (m > threshold).astype(np.uint8)


def combine_lesion_masks(mask_paths, image_shape=None) -> np.ndarray:
    """
    OR-combine any number of lesion mask files into one binary mask. Entries that are None
    (lesion type absent for this image, e.g. FGADR's sparse IRMA/NV coverage) are skipped.
    """
    combined = None
    for p in mask_paths:
        if p is None or not os.path.isfile(p):
            continue
        m = load_binary_mask(p)
        combined = m if combined is None else np.logical_or(combined, m).astype(np.uint8)
    if combined is None:
        if image_shape is None:
            raise ValueError("No valid mask paths given and no image_shape fallback provided.")
        combined = np.zeros(image_shape, dtype=np.uint8)
    return combined


def compute_lbs(lesion_mask: np.ndarray, retinal_fov_mask: np.ndarray) -> float:
    """LBS = lesion_pixels / retinal_area_pixels, per the manuscript's Sec 3.2.6 formula."""
    lesion_pixels = float(np.count_nonzero(lesion_mask))
    retinal_pixels = float(np.count_nonzero(retinal_fov_mask))
    if retinal_pixels == 0:
        return 0.0
    return lesion_pixels / retinal_pixels


def stratify_lbs(lbs_values):
    """
    low/medium/high via 33rd/66th percentile thresholds, per manuscript Sec 3.2.6. Intended
    to be computed per severity stage (percentiles within each grade), not globally -- caller
    should group by grade before calling this on more than a POC-sized sample.
    """
    lbs_values = np.asarray(lbs_values, dtype=float)
    p33, p66 = np.percentile(lbs_values, [33, 66])

    def label(v):
        if v <= p33:
            return "low"
        elif v <= p66:
            return "medium"
        return "high"

    return [label(v) for v in lbs_values], (float(p33), float(p66))


def stratify_lbs_by_grade(lbs_values, grades):
    """
    Real caller for stratify_lbs's own documented intent: computes 33rd/66th percentile LBS
    thresholds independently WITHIN each severity-grade group (burden is only meaningful
    relative to peers at the same stage, not compared globally across stages -- see
    stratify_lbs's docstring), then returns one low/medium/high label per input value in the
    original order. Used by Module 3's survival input pipeline (see module3/dataset.py) --
    previously stratify_lbs only had its own __main__ self-test as a caller (see
    docs/ROADMAP.md's Objective 3 section).

    Args:
        lbs_values: sequence of LBS floats.
        grades: sequence of severity grades (e.g. ICDR 0-4), same length/order as lbs_values.
    Returns:
        (labels, thresholds_by_grade) -- labels is a list aligned to the input order;
        thresholds_by_grade is {grade: (p33, p66)}.
    """
    lbs_values = np.asarray(lbs_values, dtype=float)
    grades = np.asarray(grades)
    labels = [None] * len(lbs_values)
    thresholds_by_grade = {}
    for grade in np.unique(grades):
        idx = np.where(grades == grade)[0]
        group_labels, thresholds = stratify_lbs(lbs_values[idx])
        thresholds_by_grade[grade] = thresholds
        for i, label in zip(idx, group_labels):
            labels[i] = label
    return labels, thresholds_by_grade


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LBS self-test on one FGADR sample")
    parser.add_argument("--fgadr-root", default="./FGADR-Seg-set_Release/Seg-set")
    parser.add_argument("--image-name", default="0003_3.png")
    parser.add_argument("--out", default="./module1_lbs_sample.png")
    args = parser.parse_args()

    image_path = os.path.join(args.fgadr_root, "Original_Images", args.image_name)
    mask_folders = ["HardExudate_Masks", "Hemohedge_Masks", "Microaneurysms_Masks", "SoftExudate_Masks"]
    mask_paths = [os.path.join(args.fgadr_root, folder, args.image_name) for folder in mask_folders]

    rgb = np.array(Image.open(image_path).convert("RGB"))
    fov_mask = estimate_retinal_fov_mask(rgb)
    lesion_mask = combine_lesion_masks(mask_paths, image_shape=rgb.shape[:2])
    lbs = compute_lbs(lesion_mask, fov_mask)

    import pandas as pd

    labels_df = pd.read_csv(
        os.path.join(args.fgadr_root, "DR_Seg_Grading_Label.csv"), header=None, names=["filename", "grade"]
    )
    grade_row = labels_df[labels_df["filename"] == args.image_name]
    grade = int(grade_row["grade"].iloc[0]) if len(grade_row) else None

    print(f"Image: {args.image_name}")
    print(f"Grade (ICDR, 0-4): {grade}")
    print(f"Retinal FOV pixels: {int(fov_mask.sum())}")
    print(f"Lesion pixels (EX+HE+MA+SE union): {int(lesion_mask.sum())}")
    print(f"LBS: {lbs:.6f}")

    # Save a quick visualization: original | lesion mask overlay in red
    overlay = rgb.copy()
    overlay[lesion_mask.astype(bool)] = [255, 0, 0]
    blended = (0.6 * rgb + 0.4 * overlay).astype(np.uint8)
    side_by_side = np.concatenate([rgb, blended], axis=1)
    Image.fromarray(side_by_side).save(args.out)
    print(f"✓ saved sample visualization to {args.out}")
