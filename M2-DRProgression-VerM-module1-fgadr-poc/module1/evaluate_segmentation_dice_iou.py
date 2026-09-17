"""
Compute Dice + IoU for a trained DRG-Net segmentation checkpoint, on the SAME test split
DRG-Net's own train_fgadr.py used (see dr_segmentation/utils.py::get_images_fgadr_from_pd --
60/20/20 sequential slice of the Filtered CSV's row order). This is IN ADDITION to DRG-Net's
own AP/ROC-AUC eval (train_fgadr.py::eval_model) -- the manuscript separately commits to
Dice/IoU as Module 1's segmentation metric, so we report both (see module1/README.md).

Usage (after training, from anywhere with this repo + the checkpoint available):
    python evaluate_segmentation_dice_iou.py \
        --fgadr-root ./FGADR-Seg-set_Release/Seg-set \
        --filtered-csv ./module1/data/DR_Seg_Grading_Label_Filtered.csv \
        --checkpoint /path/to/model_2.pth.tar \
        --lesion EX
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
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
    args = parser.parse_args()

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

    dices, ious = [], []
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
            pred = (probs[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
        pred_img = Image.fromarray(pred * 255).resize(orig_size, Image.NEAREST)
        pred_full = (np.array(pred_img) > 0).astype(np.uint8)

        gt = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)

        d, i = dice_iou(pred_full, gt)
        dices.append(d)
        ious.append(i)

    print(f"Lesion: {args.lesion}")
    print(f"N test images evaluated: {len(dices)}")
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean IoU:  {np.mean(ious):.4f}")


if __name__ == "__main__":
    main()
