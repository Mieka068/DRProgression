"""
Pool FGADR and DDR lesion-segmentation training data into one set, in the IDRiD layout DRG-Net's
dr_segmentation/train.py reads (same as prepare_idrid_segmentation.py / prepare_ddr_segmentation.py),
to train ONE model per lesion on both datasets -- the "combined" Module 1 segmenters
(notebook 11; the pipeline uses them with MODULE1_SOURCE='drgnet_combined').

Splits, kept identical to the single-dataset runs so the models are comparable:
  * FGADR -- DRG-Net's train_fgadr.py split by row order of DR_Seg_Grading_Label_Filtered.csv
    (utils.get_images_fgadr_from_pd): first 60% train, next 20% eval (model selection), last 20%
    test. Test images are NOT added here; they are scored separately (evaluate_segmentation_dice_iou.py).
  * DDR -- official train / val from prepare_ddr_segmentation.py's output (run that first);
    the DDR test set is scored separately (evaluate_model.py on the DDR layout).

    <out>/OriginalImages/TrainingSet/a_fgadr_<name>.jpg, a_ddr_<name>.jpg   (train)
                                     b_fgadr_<name>.jpg, b_ddr_<name>.jpg   (eval)
    <out>/Groundtruths/TrainingSet/{HardExudates,Haemorrhages,Microaneurysms,SoftExudates,Mask}/...
    <out>/combined_split.json   {"train_ratio": ...}

a_ sorts before b_, so DRG_SEG_TRAIN_RATIO = train_ratio makes train.py train on every train
image and select the best epoch on every eval image. DRG-Net evaluates whole images in batches,
so all images must share one size: FGADR images (1280 x 1280) are resized to the DDR prep's size
(ddr_split.json, default 1280 -> unchanged). FGADR masks are binarised (> 127) to 0/255 and the
field-of-view masks generated as for IDRiD/DDR. DDR files are linked, not copied.

Usage:
    python module1/dataprep/prepare_combined_segmentation.py \
        --fgadr-root /content/data/FGADR-Seg-set_Release/Seg-set \
        --filtered-csv module1/data/DR_Seg_Grading_Label_Filtered.csv \
        --ddr-seg-dir /content/data/DDR_seg_drgnet --out /content/data/COMB_seg_drgnet
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

LESIONS = {"EX": "HardExudates", "HE": "Haemorrhages", "MA": "Microaneurysms", "SE": "SoftExudates"}
FGADR_MASK_DIRS = {"EX": "HardExudate_Masks", "HE": "Hemohedge_Masks", "MA": "Microaneurysms_Masks",
                   "SE": "SoftExudate_Masks"}
FOV_GREEN_THRESHOLD = 10  # same as prepare_idrid_segmentation.py / prepare_ddr_segmentation.py


def fgadr_split(filtered_csv):
    """(train, eval, test) image names exactly as DRG-Net's get_images_fgadr_from_pd slices them."""
    names = list(pd.read_csv(filtered_csv)["Image_Name"])
    n_train, n_test = int(len(names) * 0.6), int(len(names) * 0.2)
    return names[:n_train], names[n_train:len(names) - n_test], names[len(names) - n_test:]


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def add_fgadr(fgadr_root, names, prefix, out, size):
    n = 0
    for name in names:
        img_path = os.path.join(fgadr_root, "Original_Images", name)
        if not os.path.isfile(img_path):
            print(f"  ⚠ FGADR image missing: {img_path}")
            continue
        stem = os.path.splitext(name)[0].replace(".", "_")
        out_name = f"{prefix}fgadr_{stem}"
        rgb = Image.open(img_path).convert("RGB")
        if rgb.size != (size, size):
            rgb = rgb.resize((size, size), Image.BICUBIC)
        dst = os.path.join(out, "OriginalImages", "TrainingSet", out_name + ".jpg")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        rgb.save(dst, quality=95)
        for les, folder in FGADR_MASK_DIRS.items():
            mp = os.path.join(fgadr_root, folder, name)
            if not os.path.isfile(mp):
                continue
            m = (np.array(Image.open(mp).convert("L")) > 127).astype(np.uint8) * 255
            m = Image.fromarray(m)
            if m.size != (size, size):
                m = m.resize((size, size), Image.NEAREST)
            mdst = os.path.join(out, "Groundtruths", "TrainingSet", LESIONS[les], f"{out_name}_{les}.tif")
            os.makedirs(os.path.dirname(mdst), exist_ok=True)
            m.save(mdst)
        green = np.asarray(rgb)[:, :, 1]
        fov = os.path.join(out, "Groundtruths", "TrainingSet", "Mask", out_name + "_MASK.tif")
        os.makedirs(os.path.dirname(fov), exist_ok=True)
        Image.fromarray(np.where(green > FOV_GREEN_THRESHOLD, 0, 255).astype(np.uint8)).save(fov)
        n += 1
    return n


def add_ddr(ddr_dir, src_prefix, dst_prefix, out):
    n = 0
    for img in sorted(glob.glob(os.path.join(ddr_dir, "OriginalImages", "TrainingSet", src_prefix + "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        out_name = dst_prefix + "ddr_" + stem[len(src_prefix):]
        link(img, os.path.join(out, "OriginalImages", "TrainingSet", out_name + ".jpg"))
        for les, folder in {**LESIONS, "MASK": "Mask"}.items():
            mp = os.path.join(ddr_dir, "Groundtruths", "TrainingSet", folder, f"{stem}_{les}.tif")
            if os.path.isfile(mp):
                link(mp, os.path.join(out, "Groundtruths", "TrainingSet", folder, f"{out_name}_{les}.tif"))
        n += 1
    return n


def prepare(fgadr_root, filtered_csv, ddr_seg_dir, out):
    with open(os.path.join(ddr_seg_dir, "ddr_split.json")) as f:
        size = json.load(f)["size"]
    train, evl, test = fgadr_split(filtered_csv)
    print(f"FGADR (DRG-Net split): {len(train)} train / {len(evl)} eval / {len(test)} test (test not pooled)")
    n = {
        "fgadr_train": add_fgadr(fgadr_root, train, "a_", out, size),
        "fgadr_eval": add_fgadr(fgadr_root, evl, "b_", out, size),
        "ddr_train": add_ddr(ddr_seg_dir, "train_", "a_", out),
        "ddr_val": add_ddr(ddr_seg_dir, "val_", "b_", out),
    }
    n_train, n_eval = n["fgadr_train"] + n["ddr_train"], n["fgadr_eval"] + n["ddr_val"]
    ratio = (n_train + 0.5) / (n_train + n_eval)
    assert int((n_train + n_eval) * ratio) == n_train
    info = {"train_ratio": ratio, "n_train": n_train, "n_eval": n_eval, "size": size, **n}
    with open(os.path.join(out, "combined_split.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"✓ train {n_train} (FGADR {n['fgadr_train']} + DDR {n['ddr_train']}) | eval {n_eval} "
          f"(FGADR {n['fgadr_eval']} + DDR {n['ddr_val']}) | all {size} x {size}")
    print(f"✓ combined layout in {out}; train_ratio {ratio:.6f} -> {os.path.join(out, 'combined_split.json')}")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fgadr-root", required=True, help="FGADR Seg-set folder (Original_Images, *_Masks)")
    parser.add_argument("--filtered-csv", required=True, help="module1/data/DR_Seg_Grading_Label_Filtered.csv")
    parser.add_argument("--ddr-seg-dir", required=True, help="Output of prepare_ddr_segmentation.py")
    parser.add_argument("--out", default="/content/data/COMB_seg_drgnet")
    args = parser.parse_args()
    prepare(args.fgadr_root, args.filtered_csv, args.ddr_seg_dir, args.out)
