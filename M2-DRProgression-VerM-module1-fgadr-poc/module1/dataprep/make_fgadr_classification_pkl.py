"""
Build the {'train','val','test': [(image_path, grade)]} pickle that DRG-Net's
dr_classification/main.py expects (base.data_index in the yaml config).

Adapted from DRG-Net's own dr_classification/external/fgadr_generate_pkl.py
(https://github.com/DFKI-Interactive-Machine-Learning/dr-joint-learning, pinned commit
0da1bbe885f438390a0e94ec486283d9b21d5547). Only two things changed from the original:
  1. Paths point at our local FGADR copy instead of the original authors' filesystem.
  2. `row[0]`/`row[1]` positional Series indexing (which relied on old-pandas fallback
     behaviour, removed in pandas 2.x / Colab's shipped pandas) is replaced with `.iloc`.
The 821/100/921 train/val/test row-slice split is kept EXACTLY as published, for fidelity to
the paper (see README.md) -- this is the same ordering as our local
DR_Seg_Grading_Label.csv since both are the official FGADR release, just re-verify with
--sanity-check before trusting it on a new copy of the dataset.

Usage:
    python make_fgadr_classification_pkl.py \
        --fgadr-root ./FGADR-Seg-set_Release/Seg-set \
        --out ./fgadr_classification_pkl.pkl
"""
import argparse
import os
import pickle

import pandas as pd


def build(fgadr_root: str, out_path: str, sanity_check: bool = True):
    image_dir = os.path.join(fgadr_root, "Original_Images")
    groundtruth_file = os.path.join(fgadr_root, "DR_Seg_Grading_Label.csv")

    # Our local copy has no header row (1,842 lines == 1,842 images) -- unlike whatever
    # copy the original script was run against. header=None + explicit names sidesteps that.
    df = pd.read_csv(groundtruth_file, header=None, names=["filename", "grade"])

    train_df = df.iloc[0:821]
    val_df = df.iloc[821:921]
    test_df = df.iloc[921:]

    def to_list(split_df):
        items = []
        missing = 0
        for _, row in split_df.iterrows():
            image_path = os.path.join(image_dir, row.iloc[0])
            if os.path.isfile(image_path):
                items.append((image_path, int(row.iloc[1])))
            else:
                missing += 1
        return items, missing

    train_data_list, train_missing = to_list(train_df)
    val_data_list, val_missing = to_list(val_df)
    test_data_list, test_missing = to_list(test_df)

    data_dict = {"train": train_data_list, "val": val_data_list, "test": test_data_list}

    with open(out_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"✓ train: {len(train_data_list)} (missing {train_missing})")
    print(f"✓ val:   {len(val_data_list)} (missing {val_missing})")
    print(f"✓ test:  {len(test_data_list)} (missing {test_missing})")
    print(f"✓ wrote {out_path}")

    if sanity_check and (train_missing or val_missing or test_missing):
        print(
            "⚠ Some rows in DR_Seg_Grading_Label.csv did not resolve to a local file. "
            "Confirm --fgadr-root points at a complete Original_Images/ folder before training."
        )

    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fgadr-root", default="./FGADR-Seg-set_Release/Seg-set")
    parser.add_argument("--out", default="./fgadr_classification_pkl.pkl")
    args = parser.parse_args()
    build(args.fgadr_root, args.out)
