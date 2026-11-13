"""
Build the {'train','val','test': [(image_path, grade)]} pickle DRG-Net's
dr_classification/main.py expects (base.data_index), for our resized EyePACS copy:
https://www.kaggle.com/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset
(all_labels.csv = the original EyePACS train + test labels merged; ICDR 0-4, same scale as
FGADR/IDRiD, so no remapping). Pair with module1/configs/eyepacs_poc.yaml (= DRG-Net's
eyepacs_ce.yaml).

DRG-Net trained an EyePACS classifier (configs/eyepacs_ce.yaml -> kaggle_data_dict.pkl) but did
not release that pickle or the script that built it, so the exact split can't be copied. Two
options (--split):

  official (default) -- the Kaggle competition's own partition, the usual EyePACS protocol:
      train = the 35,126 original training images; val = Public test (~10,906);
      test = Private test (~42,670). Read from a split/Usage-style column in all_labels.csv if
      there is one, else from the image's folder (train/ vs test/). In all_labels.csv the `Usage`
      column is public/private for the Kaggle test images and BLANK for the 35,126 original
      training images, so a blank entry in a column whose other values are only test-set tokens
      means train (it used to be treated as unknown and dropped -> "train: 0 images"). If test images can't be told
      apart into Public/Private, val is instead a patient-level 10% of train (not the test set,
      so model selection doesn't see test images) -- printed when it happens.
  random -- patient-level, grade-stratified random split (--val-frac/--test-frac).

Both keep a patient's two eyes (<id>_left / <id>_right) in the same split; in the official
split Kaggle already does that.

--combine-with merges an existing pickle (e.g. FGADR's) split-by-split, to train Module 1 on
FGADR + EyePACS together. That changes the test set, so it is NOT a DRG-Net replication run.

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
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from crosssectional_dataset import find_file, index_images, load_labels, sample_per_grade  # noqa: E402


def patient_id(image_id: str) -> str:
    """'10_left' -> '10'. Falls back to the full id for names without a _left/_right suffix."""
    for suffix in ("_left", "_right"):
        if image_id.lower().endswith(suffix):
            return image_id[: -len(suffix)]
    return image_id


SPLIT_TOKENS = {"train": "train", "training": "train", "public": "val", "val": "val",
                "validation": "val", "private": "test", "test": "test", "testing": "test"}


def official_split_column(csv_path, image_ids):
    """{image_id: 'train'|'val'|'test'} from a split/Usage-like column, or None."""
    from crosssectional_dataset import IMAGE_COL_CANDIDATES, detect_column, image_stem

    raw = pd.read_csv(csv_path)
    img_col = detect_column(raw.columns, IMAGE_COL_CANDIDATES, "image-id")
    for col in raw.columns:
        if col == img_col:
            continue
        cleaned = raw[col].astype(str).str.strip().str.lower().where(raw[col].notna(), "")
        vals = cleaned[cleaned != ""]
        if not (len(vals) and vals.isin(SPLIT_TOKENS).all()):
            continue
        tokens = {SPLIT_TOKENS[v] for v in vals.unique()}
        # Kaggle's Usage column: public/private for test images, blank for the original training
        # images -> a blank means train whenever no non-blank value already means train.
        blank_is_train = (cleaned == "").any() and "train" not in tokens
        if len(tokens) + blank_is_train < 2:
            continue
        lookup = dict(SPLIT_TOKENS, **({"": "train"} if blank_is_train else {}))
        mapping = dict(zip(raw[img_col].map(image_stem), cleaned))
        print(f"  Official split from column '{col}': {sorted(vals.unique())}"
              + (f" + {int((cleaned == '').sum())} blank (= original training images -> train)"
                 if blank_is_train else ""))
        return {i: lookup.get(mapping[i]) if i in mapping else None for i in image_ids}
    return None


def split_from_folders(paths):
    out = {}
    for image_id, path in paths.items():
        parts = [p.lower() for p in os.path.normpath(path).split(os.sep)[:-1]]
        out[image_id] = next((SPLIT_TOKENS[p] for p in reversed(parts) if p in SPLIT_TOKENS), None)
    return out


def patient_level_split(df, fracs, rng):
    """Assigns df rows to splits by patient, stratified by worse-eye grade. fracs: {name: frac}
    for every split but the last listed one, which takes the remainder."""
    names = list(fracs) + ["__rest__"]
    patient_grade = df.groupby("patient")["grade"].max()
    split_of = {}
    for g in sorted(patient_grade.unique()):
        patients = patient_grade.index[patient_grade == g].to_numpy()
        rng.shuffle(patients)
        start = 0
        for name in names[:-1]:
            n = int(round(len(patients) * fracs[name]))
            for p in patients[start:start + n]:
                split_of[p] = name
            start += n
        for p in patients[start:]:
            split_of[p] = "__rest__"
    return df["patient"].map(split_of)


def build(eyepacs_root, out_path, labels_csv="all_labels.csv", image_col=None, label_col=None,
          split="official", val_frac=0.1, test_frac=0.1, max_per_grade=None, seed=0, combine_with=None):
    csv_path = find_file(eyepacs_root, labels_csv)
    df = load_labels(csv_path, image_col, label_col)
    index = index_images(eyepacs_root)
    df["path"] = df["image_id"].map(index)
    n_missing = int(df["path"].isna().sum())
    df = df[df["path"].notna()].copy()
    print(f"✓ {len(df)} labelled images found ({n_missing} labelled but missing on disk)")
    df["patient"] = df["image_id"].map(patient_id)
    rng = np.random.RandomState(seed)

    if split == "official":
        assigned = official_split_column(csv_path, df["image_id"])
        source = "labels CSV column"
        if assigned is None or all(v is None for v in assigned.values()):
            assigned = split_from_folders(dict(zip(df["image_id"], df["path"])))
            source = "image folders (train/ vs test/)"
        df["split"] = df["image_id"].map(assigned)
        if df["split"].isna().all():
            raise SystemExit("Could not find the official Kaggle train/test partition (no split column "
                             "in the CSV, no train/test folders). Use --split random.")
        n_unknown = int(df["split"].isna().sum())
        if n_unknown:
            print(f"  ⚠ {n_unknown} images with no official split -- dropped")
            df = df[df["split"].notna()].copy()
        print(f"  Official split read from {source}")
        if not (df["split"] == "val").any():
            print("  ⚠ Test images aren't marked Public/Private, so val = a patient-level "
                  f"{val_frac:.0%} of train (keeps the test set out of model selection)")
            train = df["split"] == "train"
            sub = patient_level_split(df[train], {"val": val_frac}, rng)
            df.loc[train, "split"] = sub.map(lambda s: "val" if s == "val" else "train")
    else:
        df["split"] = patient_level_split(df, {"test": test_frac, "val": val_frac}, rng).replace(
            {"__rest__": "train"})

    if max_per_grade:
        tr = df[df["split"] == "train"]
        tr = sample_per_grade(tr, max_per_grade, seed)
        df = pd.concat([tr, df[df["split"] != "train"]])
        print(f"  Capped TRAIN at {max_per_grade}/grade (val/test untouched) -> {len(tr)} train images")

    # Both eyes of a patient must share a split. Kaggle's partition already does this; if a
    # patient is split anyway, move all their images to the split holding most of them.
    n_splits = df.groupby("patient")["split"].nunique()
    leaks = n_splits.index[n_splits > 1]
    if len(leaks):
        majority = df[df["patient"].isin(leaks)].groupby("patient")["split"].agg(lambda x: x.value_counts().index[0])
        df.loc[df["patient"].isin(leaks), "split"] = df.loc[df["patient"].isin(leaks), "patient"].map(majority)
        print(f"  ⚠ {len(leaks)} patients had images in more than one split -- moved to their majority split")

    data_dict = {s: list(zip(df.loc[df["split"] == s, "path"], df.loc[df["split"] == s, "grade"].astype(int)))
                 for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        counts = np.bincount([g for _, g in data_dict[s]], minlength=5)
        print(f"✓ {s:5s}: {len(data_dict[s])} images, per grade {counts.tolist()}")
    if not data_dict["train"]:
        raise SystemExit("No training images -- check the split column / folders printed above.")

    if combine_with:
        with open(combine_with, "rb") as f:
            other = pickle.load(f)
        for s in ("train", "val", "test"):
            data_dict[s] = list(other.get(s, [])) + data_dict[s]
        print(f"✓ Combined with {combine_with}: "
              + ", ".join(f"{s} {len(data_dict[s])}" for s in ("train", "val", "test")))

    with open(out_path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"✓ wrote {out_path} (split={split})")
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eyepacs-root", required=True, help="Folder the Kaggle zip was extracted to")
    parser.add_argument("--labels-csv", default="all_labels.csv", help="Found anywhere under --eyepacs-root")
    parser.add_argument("--image-col", default=None, help="Auto-detected if omitted")
    parser.add_argument("--label-col", default=None, help="Auto-detected if omitted")
    parser.add_argument("--split", choices=["official", "random"], default="official",
                        help="official = Kaggle train / public-test val / private-test test")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--max-per-grade", type=int, default=None,
                        help="Optional cap per grade on TRAIN only (EyePACS is ~73%% grade 0) -- "
                             "shortens training but is not DRG-Net's protocol")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--combine-with", default=None,
                        help="Optional existing pkl (e.g. fgadr_classification_pkl.pkl) to merge in")
    parser.add_argument("--out", default="./eyepacs_classification_pkl.pkl")
    args = parser.parse_args()
    build(args.eyepacs_root, args.out, args.labels_csv, args.image_col, args.label_col, args.split,
          args.val_frac, args.test_frac, args.max_per_grade, args.seed, args.combine_with)
