"""
Lay out DDR's lesion-segmentation set (Li et al., 2019, Information Sciences; 757 images,
official split 383 train / 149 val / 225 test, per-lesion .tif masks for EX/HE/MA/SE) in the
folder structure DRG-Net's IDRiD segmentation code reads (dr_segmentation/utils.py::get_images,
used by train.py and evaluate_model.py with config.py -> module1/configs/config_idrid_seg.py with
DRG_SEG_IMAGE_DIR pointed here) -- the same layout prepare_idrid_segmentation.py writes:

    <out>/OriginalImages/TrainingSet/{train_,val_}<name>.jpg
    <out>/OriginalImages/TestingSet/test_<name>.jpg
    <out>/Groundtruths/<set>/HardExudates/<image>_EX.tif  (+ Haemorrhages/_HE, Microaneurysms/_MA,
                                                           SoftExudates/_SE, Mask/_MASK.tif)
    <out>/ddr_split.json   {"train_ratio": ...}

Expected input (the Hugging Face ctmedtech/DDR-dataset download; found anywhere under --ddr-root):

    lesion_segmentation/images/{train,val,test}/<name>.jpg
    lesion_segmentation/annotations/{train,val,test}/{EX,HE,MA,SE}/<name>.tif

Split folders are matched by name: train; val / valid / validation; test / testing / tet.

Official split, kept: DRG-Net's train.py trains on the first train_ratio of the sorted
TrainingSet and evaluates (picks model_2.pth.tar by AP) on the rest. Train images are prefixed
train_ and val images val_, so the sorted order is all train then all val, and ddr_split.json's
train_ratio (pass it as DRG_SEG_TRAIN_RATIO, see drgnet_resume_patch.py) cuts exactly between
them: train on the 383, select on the 149, report on the 225 test images (evaluate_model.py).

Converted, not linked (unlike IDRiD), because:
  * masks -- DRG-Net reads a mask as RGB channel 0 / 255, which suits IDRiD's palette .tifs; DDR
    masks may be stored 0/1. They are binarised (> 0) and saved as 0/255 grayscale.
  * size -- DRG-Net's train.py evaluates whole images in batches of 4, which only works when
    every image has the same size (true for IDRiD, not for DDR), and its IDRIDDataset keeps every
    train + eval image and mask in RAM (IDRiD: 54 images; DDR: 532 at up to ~5000 px). Each image
    is cropped to the retina (black border removed, same FOV threshold as IDRiD), scaled so its
    longer side is --size and padded with black to --size x --size (masks: nearest-neighbour,
    padded with 0). Default 1280 = FGADR's native resolution (~5 GB of RAM for train + val);
    1536 / 2048 keep more detail for MA on a High-RAM runtime. Training still uses DRG-Net's
    512 px random crops, so --size must be >= 512.
The field-of-view Mask/*_MASK.tif DRG-Net needs is generated as for IDRiD (255 = background).

Usage:
    python module1/dataprep/prepare_ddr_segmentation.py \
        --ddr-root /content/data/DDR_raw --out /content/data/DDR_seg_drgnet
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

LESIONS = {"EX": "HardExudates", "HE": "Haemorrhages", "MA": "Microaneurysms", "SE": "SoftExudates"}
SPLIT_NAMES = {"train": "train", "val": "val", "valid": "val", "validation": "val",
               "test": "test", "testing": "test", "tet": "test"}
FOV_GREEN_THRESHOLD = 10  # same as prepare_idrid_segmentation.py
FOV_MARGIN = 8  # px kept around the retina when cropping
CROP = 512  # DRG-Net's training crop size (config IMAGE_SIZE); its RandomCrop doesn't pad


def find_seg_root(root):
    hits = sorted((d for d in glob.glob(os.path.join(root, "**", "lesion_segmentation"), recursive=True)
                   if os.path.isdir(d)), key=len)
    if not hits:
        raise SystemExit(f"No 'lesion_segmentation' folder under {root} (simple_segmentation is not used: "
                         "it has one combined mask per image, DRG-Net trains one model per lesion).")
    return hits[0]


def split_dirs(parent):
    """{'train'|'val'|'test': path} for the split folders directly under `parent`."""
    out = {}
    for d in sorted(os.listdir(parent)) if os.path.isdir(parent) else []:
        key = SPLIT_NAMES.get(d.lower())
        if key and os.path.isdir(os.path.join(parent, d)):
            out[key] = os.path.join(parent, d)
    return out


def load_mask(path, size):
    m = Image.open(path)
    m = np.array(m.convert("L") if m.mode not in ("1", "L", "I;16", "I", "F") else m)
    m = (m > 0).astype(np.uint8) * 255
    img = Image.fromarray(m)
    return img if img.size == size else img.resize(size, Image.NEAREST)


def crop_box(rgb):
    green = np.asarray(rgb)[:, :, 1]
    ys, xs = np.where(green > FOV_GREEN_THRESHOLD)
    if not len(ys):
        return (0, 0) + rgb.size
    h, w = green.shape
    return (max(xs.min() - FOV_MARGIN, 0), max(ys.min() - FOV_MARGIN, 0),
            min(xs.max() + FOV_MARGIN + 1, w), min(ys.max() + FOV_MARGIN + 1, h))


def to_canvas(img, size, fill):
    canvas = Image.new(img.mode, (size, size), fill)
    canvas.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return canvas


def prepare(ddr_root, out, size=1280):
    if size < CROP:
        raise SystemExit(f"--size must be >= {CROP} (DRG-Net's training crop)")
    seg = find_seg_root(ddr_root)
    images = split_dirs(os.path.join(seg, "images"))
    annos = split_dirs(os.path.join(seg, "annotations"))
    print(f"DDR lesion segmentation: {seg}")
    print(f"  image splits: {sorted(images)} | mask splits: {sorted(annos)}")
    if "train" not in images or "train" not in annos:
        raise SystemExit("Need lesion_segmentation/images/train and annotations/train")

    counts, native = {}, []
    for split in ("train", "val", "test"):
        if split not in images:
            print(f"  ⚠ no {split} images -- skipped")
            continue
        if split not in annos:
            print(f"  ⚠ no {split} masks -- {split} images skipped (they can't be trained or scored)")
            continue
        setname = "TestingSet" if split == "test" else "TrainingSet"
        c = counts.setdefault(split, {"images": 0, **{k: 0 for k in LESIONS}, **{f"{k}_nonempty": 0 for k in LESIONS}})
        for img_path in sorted(glob.glob(os.path.join(images[split], "*"))):
            if not img_path.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                continue
            name = os.path.splitext(os.path.basename(img_path))[0]
            out_name = f"{split}_{name}".replace(".", "_")  # get_images() splits names at '.'
            rgb = Image.open(img_path).convert("RGB")
            masks = {}
            for les in LESIONS:
                cand = glob.glob(os.path.join(annos[split], les, name + ".*"))
                if cand:
                    masks[les] = load_mask(cand[0], rgb.size)
            if not masks:
                print(f"  ⚠ {split}/{name}: no lesion masks -- skipped")
                continue
            native.append(rgb.size)
            box = crop_box(rgb)
            rgb = rgb.crop(box)
            masks = {k: m.crop(box) for k, m in masks.items()}
            scale = size / max(rgb.size)
            new = (max(1, round(rgb.size[0] * scale)), max(1, round(rgb.size[1] * scale)))
            rgb = to_canvas(rgb.resize(new, Image.BICUBIC), size, (0, 0, 0))
            masks = {k: to_canvas(m.resize(new, Image.NEAREST), size, 0) for k, m in masks.items()}

            img_dst = os.path.join(out, "OriginalImages", setname, out_name + ".jpg")
            os.makedirs(os.path.dirname(img_dst), exist_ok=True)
            rgb.save(img_dst, quality=95)
            for les, m in masks.items():
                dst = os.path.join(out, "Groundtruths", setname, LESIONS[les], f"{out_name}_{les}.tif")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                m.save(dst)
                c[les] += 1
                c[f"{les}_nonempty"] += int(np.asarray(m).any())
            green = np.asarray(rgb)[:, :, 1]
            fov = os.path.join(out, "Groundtruths", setname, "Mask", out_name + "_MASK.tif")
            os.makedirs(os.path.dirname(fov), exist_ok=True)
            Image.fromarray(np.where(green > FOV_GREEN_THRESHOLD, 0, 255).astype(np.uint8)).save(fov)
            c["images"] += 1

    for split, c in counts.items():
        print(f"✓ {split}: {c['images']} images | masks (with any lesion pixels): "
              + ", ".join(f"{k} {c[k]} ({c[k + '_nonempty']})" for k in LESIONS))
    expected = {"train": 383, "val": 149, "test": 225}
    for split, n in expected.items():
        if counts.get(split, {}).get("images", 0) != n:
            print(f"  ⚠ {split}: expected {n} images (official DDR split), found {counts.get(split, {}).get('images', 0)}")
    if native:
        ws, hs = zip(*native)
        print(f"  original sizes {min(ws)}-{max(ws)} x {min(hs)}-{max(hs)} px -> all {size} x {size} "
              "(retina cropped, scaled, padded)")

    n_train = counts.get("train", {}).get("images", 0)
    n_val = counts.get("val", {}).get("images", 0)
    if not n_val:
        print("  ⚠ no val images: train_ratio 0.8 -> DRG-Net carves its eval slice out of train")
        ratio = 0.8
    else:
        ratio = (n_train + 0.5) / (n_train + n_val)  # int((n_train + n_val) * ratio) == n_train
        assert int((n_train + n_val) * ratio) == n_train
    split_info = {"train_ratio": ratio, "n_train": n_train, "n_val": n_val,
                  "n_test": counts.get("test", {}).get("images", 0), "size": size}
    with open(os.path.join(out, "ddr_split.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"✓ DRG-Net layout written to {out}; train_ratio {ratio:.6f} "
          f"(train {n_train} / eval {n_val}) -> {os.path.join(out, 'ddr_split.json')}")
    return split_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddr-root", required=True, help="Folder the DDR zip was extracted to")
    parser.add_argument("--out", default="/content/data/DDR_seg_drgnet")
    parser.add_argument("--size", type=int, default=1280,
                        help="Every image becomes size x size (retina cropped, scaled, padded). 1280 = "
                             "FGADR's resolution; 1536/2048 on a High-RAM runtime (DRG-Net holds all "
                             "train+val images in memory).")
    args = parser.parse_args()
    prepare(args.ddr_root, args.out, args.size)
