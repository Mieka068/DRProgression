"""
Build the {'train','val','test': [(image_path, grade)]} pickle DRG-Net's
dr_classification/main.py expects (base.data_index), for our resized EyePACS copy:
https://www.kaggle.com/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset
(all_labels.csv = the original EyePACS train + test labels merged; ICDR 0-4, same scale as
FGADR/IDRiD, so no remapping).

Same output contract as make_fgadr_classification_pkl.py / make_idrid_classification_pkl.py, but
unlike those two there is no published split to stay faithful to, so this script makes one:

  * PATIENT-LEVEL split. EyePACS images are named <patient>_left / <patient>_right; both eyes of
    one patient look alike and usually share a grade, so splitting per image would put a
    patient's other eye in the test set and inflate test accuracy. Every patient's eyes go to
    the same split.
  * Stratified by the patient's worse-eye grade, so the rare stages 3/4 appear in val/test.
  * Seeded (--seed), default 80/10/10.

Columns and image locations are auto-detected exactly like crosssectional_dataset.py (shared
helpers), so this works whether images sit in one folder or in train/ + test/ subfolders.

--combine-with merges an existing pickle (e.g. FGADR's) split-by-split, for training Module 1 on
FGADR + EyePACS together. Note this changes the test set, so test metrics are no longer
comparable to the DRG-Net paper's FGADR-only numbers -- keep the FGADR-only run for that.

Usage:
    python module1/dataprep/make_eyepacs_classification_pkl.py \
        --eyepacs-root /content/data/eyepacs \
        --out /content/drive/MyDrive/Thesis_Datasets/module1_pkl/eyepacs_classification_pkl.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from crosssectional_dataset import find_file, index_images, load_labels  # noqa: E402


def patient_id(image_id: str) -> str:
    """'10_left' -> '10'. Falls back to the full id for names without a _left/_right suffix."""
    for suffix in ("_left", "_right"):
        if image_id.lower().endswith(suffix):
            return image_id[: -len(suffix)]
    return image_id


def build(eyepacs_root, out_path, labels_csv="all_labels.csv", image_col=None, label_col=None,
          val_frac=0.1, test_frac=0.1, max_per_grade=None, seed=0, combine_with=None):
    df = load_labels(find_file(eyepacs_root, labels_csv), image_col, label_col)
    index = index_images(eyepacs_root)
    df["path"] = df["image_id"].map(index)
    n_missing = int(df["path"].isna().sum())
    df = df[df["path"].notna()].copy()
    print(f"✓ {len(df)} labelled images found ({n_missing} labelled but missing on disk)")

    rng = np.random.RandomState(seed)
    if max_per_grade:
        df = df.groupby("grade", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max_per_grade), random_state=seed))
        print(f"  Capped at {max_per_grade}/grade -> {len(df)} images")

    df["patient"] = df["image_id"].map(patient_id)
    patient_grade = df.groupby("patient")["grade"].max()
    split_of = {}
    for g in sorted(patient_grade.unique()):
        patients = patient_grade.index[patient_grade == g].to_numpy()
        rng.shuffle(patients)
        n_test = int(round(len(patients) * test_frac))
        n_val = int(round(len(patients) * val_frac))
        for i, p in enumerate(patients):
            split_of[p] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    df["split"] = df["patient"].map(split_of)

    data_dict = {s: list(zip(df.loc[df["split"] == s, "path"], df.loc[df["split"] == s, "grade"].astype(int)))
                 for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        counts = np.bincount([g for _, g in data_dict[s]], minlength=5)
        print(f"✓ {s:5s}: {len(data_dict[s])} images, per grade {counts.tolist()}")

    if combine_with:
        with open(combine_with, "rb") as f:
            other = pickle.load(f)
        for s in ("train", "val", "test"):
            data_dict[s] = list(other.get(s, [])) + data_dict[s]
        print(f"✓ Combined with {combine_with}: "
              + ", ".join(f"{s} {len(data_dict[s])}" for s in ("train", "val", "test")))

    with open(out_path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"✓ wrote {out_path}")
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eyepacs-root", required=True, help="Folder the Kaggle zip was extracted to")
    parser.add_argument("--labels-csv", default="all_labels.csv", help="Found anywhere under --eyepacs-root")
    parser.add_argument("--image-col", default=None, help="Auto-detected if omitted")
    parser.add_argument("--label-col", default=None, help="Auto-detected if omitted")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--max-per-grade", type=int, default=None,
                        help="Optional cap per grade (EyePACS is ~73%% grade 0) to shorten training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--combine-with", default=None,
                        help="Optional existing pkl (e.g. fgadr_classification_pkl.pkl) to merge in")
    parser.add_argument("--out", default="./eyepacs_classification_pkl.pkl")
    args = parser.parse_args()
    build(args.eyepacs_root, args.out, args.labels_csv, args.image_col, args.label_col,
          args.val_frac, args.test_frac, args.max_per_grade, args.seed, args.combine_with)
