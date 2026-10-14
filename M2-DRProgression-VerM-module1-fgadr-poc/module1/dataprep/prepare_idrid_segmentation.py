"""
Lay out IDRiD's segmentation set ("A. Segmentation", 54 train / 27 test images) in the folder
structure DRG-Net's IDRiD segmentation code reads (dr_segmentation/utils.py::get_images, used by
train.py and evaluate_model.py with config.py -> our module1/configs/config_idrid_seg.py):

    <out>/OriginalImages/{TrainingSet,TestingSet}/IDRiD_01.jpg
    <out>/Groundtruths/{TrainingSet,TestingSet}/HardExudates/IDRiD_01_EX.tif
                                              /Haemorrhages/IDRiD_01_HE.tif
                                              /Microaneurysms/IDRiD_01_MA.tif
                                              /SoftExudates/IDRiD_01_SE.tif
                                              /Mask/IDRiD_01_MASK.tif

(the same layout as DRG-Net's bundled dr_segmentation/data/IDRID_Samples/).

Images and lesion masks are symlinked (or copied with --copy), never converted -- DRG-Net reads
the original palette .tif files itself. IDRiD doesn't ship the field-of-view `Mask/*_MASK.tif`
DRG-Net needs for its brightness-balance / CLAHE preprocessing, so it is generated here with
DRG-Net's own convention (255 = background outside the retina, 0 = retina; checked against their
bundled IDRiD samples, where a green-channel threshold of 10 reproduces their mask on 99.9% of
pixels). Missing lesion files are fine: not every IDRiD image has SE, and DRG-Net skips images
without a mask for the lesion being trained.

Usage:
    python module1/dataprep/prepare_idrid_segmentation.py \
        --idrid-root /content/data/IDRiD_dataset --out /content/data/IDRiD_seg_drgnet
"""
import argparse
import glob
import os
import re
import shutil

import numpy as np
from PIL import Image

LESIONS = {  # DRG-Net folder name, file suffix, keywords in IDRiD's own folder names
    "EX": ("HardExudates", ("hard exudate",)),
    "HE": ("Haemorrhages", ("haemorrhage", "hemorrhage")),
    "MA": ("Microaneurysms", ("microaneurysm",)),
    "SE": ("SoftExudates", ("soft exudate",)),
}
FOV_GREEN_THRESHOLD = 10


def _norm(s):
    return re.sub(r"\s+", " ", s.lower().replace("_", " "))


def find_segmentation_root(root):
    hits = [d for d in glob.glob(os.path.join(root, "**"), recursive=True)
            if os.path.isdir(d) and "segmentation" in _norm(os.path.basename(d))
            and "groundtruth" not in _norm(os.path.basename(d))]
    if not hits:
        raise SystemExit(f"No IDRiD 'A. Segmentation' folder found under {root}. The Kaggle/IEEE IDRiD "
                         "download has three parts (A. Segmentation, B. Disease Grading, C. Localization); "
                         "only B is needed for classification, A for segmentation.")
    return sorted(hits, key=len)[0]


def split_of(path):
    n = _norm(path)
    return "TrainingSet" if "train" in n else "TestingSet" if "test" in n else None


def fov_mask(image_path):
    green = np.array(Image.open(image_path).convert("RGB"))[:, :, 1]
    return np.where(green > FOV_GREEN_THRESHOLD, 0, 255).astype(np.uint8)


def place(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    shutil.copy(src, dst) if copy else os.symlink(os.path.abspath(src), dst)


def prepare(idrid_root, out, copy=False):
    seg_root = find_segmentation_root(idrid_root)
    print(f"IDRiD segmentation set: {seg_root}")
    images = [p for p in glob.glob(os.path.join(seg_root, "**", "*.jpg"), recursive=True)
              if "original" in _norm(p)]
    tifs = glob.glob(os.path.join(seg_root, "**", "*.tif"), recursive=True)
    if not images:
        raise SystemExit(f"No original .jpg images under {seg_root}")

    counts = {}
    for img in sorted(images):
        split = split_of(os.path.relpath(img, seg_root))
        if split is None:
            continue
        name = os.path.splitext(os.path.basename(img))[0]  # IDRiD_01
        place(img, os.path.join(out, "OriginalImages", split, name + ".jpg"), copy)
        mask_path = os.path.join(out, "Groundtruths", split, "Mask", name + "_MASK.tif")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        Image.fromarray(fov_mask(img)).save(mask_path)
        counts.setdefault(split, {"images": 0, **{k: 0 for k in LESIONS}})["images"] += 1

    for tif in tifs:
        split = split_of(os.path.relpath(tif, seg_root))
        base = os.path.basename(tif)
        for abbv, (folder, keywords) in LESIONS.items():
            if base.upper().endswith(f"_{abbv}.TIF") and any(k in _norm(tif) for k in keywords) and split:
                name = base[: -len(f"_{abbv}.tif")]
                place(tif, os.path.join(out, "Groundtruths", split, folder, f"{name}_{abbv}.tif"), copy)
                counts[split][abbv] += 1

    for split, c in sorted(counts.items(), reverse=True):
        print(f"✓ {split}: {c['images']} images, masks " + ", ".join(f"{k} {c[k]}" for k in LESIONS))
    expected = {"TrainingSet": 54, "TestingSet": 27}
    for split, n in expected.items():
        if counts.get(split, {}).get("images") != n:
            print(f"⚠ {split}: expected {n} images (official IDRiD split), found "
                  f"{counts.get(split, {}).get('images', 0)}")
    print(f"✓ DRG-Net IDRiD layout written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid-root", required=True, help="Folder containing IDRiD's 'A. Segmentation'")
    parser.add_argument("--out", default="/content/data/IDRiD_seg_drgnet")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = parser.parse_args()
    prepare(args.idrid_root, args.out, args.copy)
