"""
Build the {'train','val','test': [(image_path, grade)]} pickle for IDRiD, same contract as
make_fgadr_classification_pkl.py -- see that file's docstring for the full provenance note.

Adapted from DRG-Net's dr_classification/external/idrid_generate_pkl.py. Two changes from the
original, same as the FGADR script:
  1. Paths point at our local IDRiD_dataset/B. Disease Grading/ copy.
  2. `.iloc` instead of positional `row[0]`/`row[1]` (pandas 2.x compatibility).
One more note kept faithfully from the original: val == test (the reference script builds the
val split from the *test* CSV, not a held-out slice of train -- this is a quirk of the
published implementation, not a bug we introduced; see README.md).

Usage:
    python make_idrid_classification_pkl.py \
        --idrid-root "./IDRiD_dataset/B. Disease Grading" \
        --out ./idrid_classification_pkl.pkl
"""
import argparse
import os
import pickle

import pandas as pd


def build(idrid_root: str, out_path: str):
    train_image_dir = os.path.join(idrid_root, "1. Original Images", "a. Training Set")
    test_image_dir = os.path.join(idrid_root, "1. Original Images", "b. Testing Set")
    train_csv = os.path.join(
        idrid_root, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv"
    )
    test_csv = os.path.join(
        idrid_root, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv"
    )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    def to_list(df, image_dir):
        items = []
        missing = 0
        for _, row in df.iterrows():
            image_path = os.path.join(image_dir, str(row.iloc[0]) + ".jpg")
            if os.path.isfile(image_path):
                items.append((image_path, int(row.iloc[1])))
            else:
                missing += 1
        return items, missing

    train_data_list, train_missing = to_list(train_df, train_image_dir)
    test_data_list, test_missing = to_list(test_df, test_image_dir)
    val_data_list = test_data_list  # faithful to the original script's own quirk

    data_dict = {"train": train_data_list, "val": val_data_list, "test": test_data_list}

    with open(out_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"✓ train: {len(train_data_list)} (missing {train_missing})")
    print(f"✓ val:   {len(val_data_list)} (== test split)")
    print(f"✓ test:  {len(test_data_list)} (missing {test_missing})")
    print(f"✓ wrote {out_path}")

    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid-root", default="./IDRiD_dataset/B. Disease Grading")
    parser.add_argument("--out", default="./idrid_classification_pkl.pkl")
    args = parser.parse_args()
    build(args.idrid_root, args.out)
