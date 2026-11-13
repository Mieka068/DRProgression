"""
Build the {'train','val','test': [(image_path, grade)]} pickle that DRG-Net's
dr_classification/main.py expects (base.data_index in the yaml config).

DRG-Net (https://github.com/DFKI-Interactive-Machine-Learning/dr-joint-learning, pinned commit
0da1bbe885f438390a0e94ec486283d9b21d5547) ships TWO different FGADR split scripts, so --split
picks which one to reproduce:

  release (default) -- dr_classification/fgadr_generate_pkl.py: rows [0:1500] train,
      [1500:] test, and val IS the test list (val == test). This is the split of the data dict
      DRG-Net actually released (data/fgadr_data_dict.pkl, which configs/fgadr.yaml's
      commented-out `data_index: 'fgadr_data_dict.pkl'` points at): 1500 / 342 / 342.
      Use this to replicate their FGADR classification numbers.
  external -- dr_classification/external/fgadr_generate_pkl.py: [0:821] / [821:921] / [921:]
      (saved there as "processed_fgadr_pkl_file_frist_half.pkl", i.e. a side experiment).
      This was this repo's previous default; kept for comparability with earlier runs.

Note val == test in the release split is DRG-Net's own protocol (best_validation_weights.pt is
selected on the test images); reproduce it as-is for replication, and say so in the write-up.

--verify (on by default with --split release) checks the produced split image-by-image against
module1/data/drgnet_fgadr_classification_split.csv -- the file list extracted from DRG-Net's
released data/fgadr_data_dict.pkl -- so a reordered or incomplete local FGADR copy is caught
before training rather than silently changing the test set.

Two code changes from the original scripts: paths point at our local FGADR copy, and `.iloc`
replaces positional `row[0]`/`row[1]` (pandas 2.x).

Usage:
    python make_fgadr_classification_pkl.py \
        --fgadr-root ./FGADR-Seg-set_Release/Seg-set \
        --out ./fgadr_classification_pkl.pkl            # --split external for the old split
"""
import argparse
import os
import pickle

import pandas as pd

SPLITS = {
    "release": {"train": (0, 1500), "test": (1500, None), "val": "test"},
    "external": {"train": (0, 821), "val": (821, 921), "test": (921, None)},
}
RELEASE_SPLIT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                                 "drgnet_fgadr_classification_split.csv")


def verify_against_release(data_dict):
    ref = pd.read_csv(RELEASE_SPLIT_CSV)
    ok = True
    for split in ("train", "test"):
        want = list(zip(ref.loc[ref["split"] == split, "image"], ref.loc[ref["split"] == split, "grade"]))
        got = [(os.path.basename(p), g) for p, g in data_dict[split]]
        if got == want:
            print(f"✓ {split}: identical to DRG-Net's released split ({len(got)} images, same order and grades)")
            continue
        ok = False
        missing = set(w[0] for w in want) - set(g[0] for g in got)
        extra = set(g[0] for g in got) - set(w[0] for w in want)
        grade_diff = [n for (n, g), (n2, g2) in zip(got, want) if n == n2 and g != g2]
        print(f"⚠ {split}: differs from DRG-Net's released split -- {len(got)} vs {len(want)} images, "
              f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), {len(extra)} extra, "
              f"{len(grade_diff)} grade mismatches")
    return ok


def build(fgadr_root: str, out_path: str, split: str = "release", verify: bool = True):
    image_dir = os.path.join(fgadr_root, "Original_Images")
    groundtruth_file = os.path.join(fgadr_root, "DR_Seg_Grading_Label.csv")

    # Our local copy has no header row (1,842 lines == 1,842 images) -- unlike whatever
    # copy the original script was run against. header=None + explicit names sidesteps that.
    df = pd.read_csv(groundtruth_file, header=None, names=["filename", "grade"])
    if not str(df.iloc[0, 1]).strip().isdigit():  # a copy that does have a header row
        df = df.iloc[1:].reset_index(drop=True)

    def to_list(split_df):
        items, missing = [], 0
        for _, row in split_df.iterrows():
            image_path = os.path.join(image_dir, row.iloc[0])
            if os.path.isfile(image_path):
                items.append((image_path, int(row.iloc[1])))
            else:
                missing += 1
        return items, missing

    spec = SPLITS[split]
    data_dict, n_missing = {}, 0
    for name, rng in spec.items():
        if isinstance(rng, tuple):
            data_dict[name], m = to_list(df.iloc[rng[0]:rng[1]])
            n_missing += m
    for name, rng in spec.items():
        if isinstance(rng, str):
            data_dict[name] = data_dict[rng]  # faithful to DRG-Net: val == test

    with open(out_path, "wb") as f:
        pickle.dump(data_dict, f)

    for name in ("train", "val", "test"):
        note = f" (== {spec[name]})" if isinstance(spec[name], str) else ""
        print(f"✓ {name:5s}: {len(data_dict[name])}{note}")
    print(f"✓ wrote {out_path} (split={split})")
    if n_missing:
        print(f"⚠ {n_missing} rows in DR_Seg_Grading_Label.csv did not resolve to a local file. "
              "Confirm --fgadr-root points at a complete Original_Images/ folder before training.")
    if verify and split == "release" and not verify_against_release(data_dict):
        print("⚠ Not identical to DRG-Net's released split -- numbers won't be directly comparable.")
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fgadr-root", default="./FGADR-Seg-set_Release/Seg-set")
    parser.add_argument("--out", default="./fgadr_classification_pkl.pkl")
    parser.add_argument("--split", choices=list(SPLITS), default="release",
                        help="release = DRG-Net's released 1500/342 split (replication); "
                             "external = the 821/100/921 split this repo used before")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    build(args.fgadr_root, args.out, args.split, verify=not args.no_verify)
