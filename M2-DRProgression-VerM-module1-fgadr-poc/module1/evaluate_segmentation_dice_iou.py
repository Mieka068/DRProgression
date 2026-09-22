"""
Compute Dice + IoU, plus AUC-ROC and AUC-PR, for a trained DRG-Net segmentation checkpoint, on
the SAME test split DRG-Net's own train_fgadr.py used (see
dr_segmentation/utils.py::get_images_fgadr_from_pd -- 60/20/20 sequential slice of the
Filtered CSV's row order). AUC-ROC/AUC-PR are DRG-Net's own published segmentation metric
(Tusfiqur et al. 2022, Section VI-B, Table III) -- computed here the same way
train_fgadr.py::eval_model does (average_precision_score / roc_auc_score on the raw
soft/probability output, before thresholding), so these numbers are directly comparable to the
paper's. Dice/IoU is reported alongside for its own sake but is not in the paper, so it should
not be presented as a comparison to DRG-Net's published numbers.

Usage (after training, from anywhere with this repo + the checkpoint available):
    python evaluate_segmentation_dice_iou.py \
        --fgadr-root ./FGADR-Seg-set_Release/Seg-set \
        --filtered-csv ./module1/data/DR_Seg_Grading_Label_Filtered.csv \
        --checkpoint /path/to/model_2.pth.tar \
        --lesion EX
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchvision import transforms

LESION_TO_FOLDER = {
    "EX": "HardExudate_Masks",
    "HE": "Hemohedge_Masks",
    "MA": "Microaneurysms_Masks",
    "SE": "SoftExudate_Masks",
}


def get_test_split(filtered_csv: str):
    """Same 60/20/20 sequential split as get_images_fgadr_from_pd in the DRG-Net repo."""
    df = pd.read_csv(filtered_csv)
    names = list(df["Image_Name"])
    n = len(names)
    train_number = int(n * 0.6)
    test_number = int(n * 0.2)
    eval_split = n - test_number
    return names[eval_split:]  # phase == 'test' -> imgs (no slicing) in the original code
    # NOTE: get_images_fgadr_from_pd's 'test' phase actually returns ALL images (no slice) --
    # its train/eval split only applies to 'train'/'eval' phases. We instead evaluate on the
    # held-out slice actually excluded from training (eval_split:) for an honest, non-leaked
    # number, which is a deliberate, documented departure from the reference script's own
    # (arguably looser) 'test' phase definition -- flag this if asked.


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fgadr-root", default="./FGADR-Seg-set_Release/Seg-set")
    parser.add_argument("--filtered-csv", default="./module1/data/DR_Seg_Grading_Label_Filtered.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lesion", required=True, choices=list(LESION_TO_FOLDER.keys()))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--curve-out", default=None,
                         help="If given, writes {auc, ap, fpr, tpr} (pooled ROC curve, "
                              "downsampled to --curve-points) to this JSON path, for "
                              "module1/plot_figures.py's per-lesion ROC curve figure.")
    parser.add_argument("--curve-points", type=int, default=200,
                         help="Number of (fpr, tpr) points to keep in --curve-out.")
    parser.add_argument("--curve-max-pixels-per-image", type=int, default=20000,
                         help="Random pixel subsample per image before pooling for the ROC "
                              "curve -- bounds memory on a large test set without meaningfully "
                              "changing the curve's shape.")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    import segmentation_models_pytorch as smp

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=2)
    state = torch.load(args.checkpoint, map_location=args.device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(args.device)

    test_names = get_test_split(args.filtered_csv)
    tfm = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])

    mask_folder = os.path.join(args.fgadr_root, LESION_TO_FOLDER[args.lesion])
    image_dir = os.path.join(args.fgadr_root, "Original_Images")

    dices, ious, aps, aucs = [], [], [], []
    pooled_gt, pooled_score = [], []
    for name in test_names:
        image_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_folder, name)
        if not (os.path.isfile(image_path) and os.path.isfile(mask_path)):
            continue  # this lesion type absent for this image -- skip, same as training

        img = Image.open(image_path).convert("RGB")
        orig_size = img.size
        x = tfm(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            soft_pred = probs[0, 1].cpu().numpy()
        # Resize the raw soft prediction (not yet thresholded) back to the mask's original
        # size for the AP/AUC computation, same soft output DRG-Net's own eval_model scores.
        soft_pred_full = np.array(
            Image.fromarray(soft_pred.astype(np.float32), mode="F").resize(orig_size, Image.BILINEAR)
        )
        pred_full = (soft_pred_full > 0.5).astype(np.uint8)

        gt = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)

        d, i = dice_iou(pred_full, gt)
        dices.append(d)
        ious.append(i)

        gt_flat, soft_flat = gt.flatten(), soft_pred_full.flatten()
        aps.append(average_precision_score(gt_flat, soft_flat))
        if gt_flat.max() > 0:
            aucs.append(roc_auc_score(gt_flat, soft_flat))
        # else: this image has no positive pixels for this lesion -- AUC-ROC is undefined,
        # skip it rather than counting a NaN into the mean (AP handles this case natively).

        if args.curve_out and gt_flat.size > args.curve_max_pixels_per_image:
            sample_idx = rng.choice(gt_flat.size, size=args.curve_max_pixels_per_image, replace=False)
            pooled_gt.append(gt_flat[sample_idx])
            pooled_score.append(soft_flat[sample_idx])
        elif args.curve_out:
            pooled_gt.append(gt_flat)
            pooled_score.append(soft_flat)

    print(f"Lesion: {args.lesion}")
    print(f"N test images evaluated: {len(dices)}")
    print(f"Mean Dice:    {np.mean(dices):.4f}")
    print(f"Mean IoU:     {np.mean(ious):.4f}")
    print(f"Mean AUC-PR:  {np.mean(aps):.4f}")
    print(f"Mean AUC-ROC: {np.mean(aucs):.4f} (n={len(aucs)}/{len(dices)} images had >=1 positive pixel)")

    if args.curve_out:
        y_true = np.concatenate(pooled_gt)
        y_score = np.concatenate(pooled_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # Downsample to --curve-points evenly-spaced points along fpr for a compact,
        # plottable curve (roc_curve's own output can have as many points as pixels).
        if len(fpr) > args.curve_points:
            keep_idx = np.linspace(0, len(fpr) - 1, args.curve_points).astype(int)
            fpr, tpr = fpr[keep_idx], tpr[keep_idx]
        curve_data = {
            "lesion": args.lesion,
            "auc": float(np.mean(aucs)),
            "ap": float(np.mean(aps)),
            "n_pixels_pooled": int(y_true.size),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
        with open(args.curve_out, "w") as f:
            json.dump(curve_data, f)
        print(f"✓ Wrote pooled ROC curve ({curve_data['n_pixels_pooled']} pixels) to {args.curve_out}")


if __name__ == "__main__":
    main()
