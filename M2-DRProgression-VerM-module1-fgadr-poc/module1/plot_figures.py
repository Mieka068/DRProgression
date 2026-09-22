"""
Module 1 figures (Task O): per-lesion ROC curves, classification confusion matrix, and
kappa/accuracy vs. epoch -- reads the JSON/CSV/log files the other module1 scripts already
produce, no new training or inference here.

Usage:
    # Per-lesion ROC curves (from evaluate_segmentation_dice_iou.py --curve-out):
    python plot_figures.py roc \
        --curve EX=./curve_ex.json --curve HE=./curve_he.json \
        --curve MA=./curve_ma.json --curve SE=./curve_se.json \
        --out ./figures/module1_roc_curves.png

    # Confusion matrix (from dump_classification_predictions.py):
    python plot_figures.py confusion \
        --predictions-csv ./predictions_fgadr.csv --dataset-name FGADR \
        --out ./figures/module1_confusion_fgadr.png

    # Kappa/accuracy vs epoch (from a CSV with columns epoch,kappa,accuracy -- the reliable
    # path; see --log-txt below for parsing DRG-Net's own training log instead):
    python plot_figures.py training-curve \
        --history-csv ./fgadr_classification_history.csv \
        --out ./figures/module1_kappa_vs_epoch_fgadr.png

    # Qualitative segmentation grid (image | ground truth | predicted mask):
    python plot_figures.py seg-grid \
        --fgadr-root ./FGADR-Seg-set_Release/Seg-set \
        --filtered-csv ./data/DR_Seg_Grading_Label_Filtered.csv \
        --checkpoint /path/to/model_EX.pth.tar --lesion EX --n-examples 3 \
        --out ./figures/module1_seg_grid_ex.png
"""
import argparse
import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from evaluate_segmentation_dice_iou import LESION_TO_FOLDER, get_test_split  # noqa: E402

LESION_ORDER = ["EX", "HE", "MA", "SE"]


def plot_roc_curves(curve_paths: dict, out_path: str, title="Module 1 Segmentation: Per-Lesion ROC"):
    fig, ax = plt.subplots(figsize=(6, 6))
    for lesion in LESION_ORDER:
        if lesion not in curve_paths:
            continue
        with open(curve_paths[lesion]) as f:
            curve = json.load(f)
        ax.plot(curve["fpr"], curve["tpr"], label=f"{lesion} (AUC={curve['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def plot_confusion_matrix(predictions_csv: str, out_path: str, dataset_name: str, num_classes=5):
    """5x5 (ICDR/DR-grade 0-4) confusion matrix from dump_classification_predictions.py's CSV."""
    true_grades, pred_grades = [], []
    with open(predictions_csv) as f:
        for row in csv.DictReader(f):
            if dataset_name and row["dataset"] != dataset_name:
                continue
            true_grades.append(int(row["true_grade"]))
            pred_grades.append(int(row["pred_grade"]))

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_grades, pred_grades):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center", va="center",
                    color=color, fontsize=8)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    n = len(true_grades)
    acc = sum(1 for t, p in zip(true_grades, pred_grades) if t == p) / n if n else float("nan")
    ax.set_title(f"{dataset_name} classification confusion matrix (n={n}, acc={acc:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def parse_history_csv(path):
    epochs, kappas, accuracies = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(float(row["epoch"]))
            kappas.append(float(row["kappa"]) if row.get("kappa") not in (None, "") else None)
            accuracies.append(float(row["accuracy"]) if row.get("accuracy") not in (None, "") else None)
    return epochs, kappas, accuracies


def parse_drgnet_log_txt(path):
    """
    Best-effort parser for DRG-Net's own dr_classification/main.py training log (the file its
    'log_path' config points at). DRG-Net is an external, cloned repo (see module1/README.md)
    so its exact log format isn't vendored here -- this looks for lines carrying both an
    epoch number and kappa/accuracy values (e.g. "Epoch 12 ... kappa: 0.812 ... acc: 0.734"),
    case-insensitively. If your log's format doesn't match, prefer --history-csv instead (dump
    the same numbers to a CSV with columns epoch,kappa,accuracy) rather than editing this regex
    against a moving target.
    """
    epoch_re = re.compile(r"epoch[:\s]+(\d+)", re.IGNORECASE)
    kappa_re = re.compile(r"kappa[:\s]+([\d.]+)", re.IGNORECASE)
    acc_re = re.compile(r"acc(?:uracy)?[:\s]+([\d.]+)", re.IGNORECASE)

    epochs, kappas, accuracies = [], [], []
    with open(path) as f:
        for line in f:
            epoch_m = epoch_re.search(line)
            if not epoch_m:
                continue
            kappa_m = kappa_re.search(line)
            acc_m = acc_re.search(line)
            if not (kappa_m or acc_m):
                continue
            epochs.append(float(epoch_m.group(1)))
            kappas.append(float(kappa_m.group(1)) if kappa_m else None)
            accuracies.append(float(acc_m.group(1)) if acc_m else None)
    if not epochs:
        raise ValueError(
            f"No epoch+kappa/accuracy lines matched in {path} -- this log's format doesn't "
            "match the assumed pattern (see parse_drgnet_log_txt's docstring). Use "
            "--history-csv instead."
        )
    return epochs, kappas, accuracies


def plot_training_curve(epochs, kappas, accuracies, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if any(k is not None for k in kappas):
        ax.plot(epochs, kappas, marker="o", label="Quadratic Weighted Kappa")
    if any(a is not None for a in accuracies):
        ax.plot(epochs, accuracies, marker="s", label="Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def plot_segmentation_grid(fgadr_root, filtered_csv, checkpoint, lesion, n_examples, out_path,
                            image_size=512, device="cuda" if torch.cuda.is_available() else "cpu"):
    import segmentation_models_pytorch as smp

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=2)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    test_names = get_test_split(filtered_csv)
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    mask_folder = os.path.join(fgadr_root, LESION_TO_FOLDER[lesion])
    image_dir = os.path.join(fgadr_root, "Original_Images")

    examples = []
    for name in test_names:
        image_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_folder, name)
        if not (os.path.isfile(image_path) and os.path.isfile(mask_path)):
            continue
        gt = (np.array(Image.open(mask_path).convert("L")) > 127)
        if gt.sum() == 0:
            continue  # prefer examples where the lesion is actually visible
        examples.append((image_path, mask_path))
        if len(examples) >= n_examples:
            break

    if not examples:
        raise ValueError(f"No test images with a non-empty {lesion} mask found -- check "
                          "--fgadr-root/--filtered-csv.")

    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
    if len(examples) == 1:
        axes = axes[None, :]
    for row, (image_path, mask_path) in enumerate(examples):
        img = Image.open(image_path).convert("RGB")
        orig_size = img.size
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            pred = (probs[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
        pred_full = np.array(Image.fromarray(pred * 255).resize(orig_size, Image.NEAREST)) > 0
        gt_full = np.array(Image.open(mask_path).convert("L")) > 127

        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Image" if row == 0 else "")
        axes[row, 1].imshow(gt_full, cmap="gray")
        axes[row, 1].set_title("Ground truth" if row == 0 else "")
        axes[row, 2].imshow(pred_full, cmap="gray")
        axes[row, 2].set_title("Predicted" if row == 0 else "")
        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(f"Module 1 segmentation ({lesion}): qualitative examples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_roc = sub.add_parser("roc")
    p_roc.add_argument("--curve", action="append", default=[], metavar="LESION=PATH", required=True)
    p_roc.add_argument("--out", required=True)

    p_cm = sub.add_parser("confusion")
    p_cm.add_argument("--predictions-csv", required=True)
    p_cm.add_argument("--dataset-name", required=True)
    p_cm.add_argument("--out", required=True)

    p_tc = sub.add_parser("training-curve")
    p_tc.add_argument("--history-csv", default=None, help="CSV with columns epoch,kappa,accuracy")
    p_tc.add_argument("--log-txt", default=None, help="DRG-Net's own training log (best-effort parse)")
    p_tc.add_argument("--dataset-name", default="FGADR")
    p_tc.add_argument("--out", required=True)

    p_sg = sub.add_parser("seg-grid")
    p_sg.add_argument("--fgadr-root", required=True)
    p_sg.add_argument("--filtered-csv", required=True)
    p_sg.add_argument("--checkpoint", required=True)
    p_sg.add_argument("--lesion", required=True, choices=LESION_ORDER)
    p_sg.add_argument("--n-examples", type=int, default=3)
    p_sg.add_argument("--image-size", type=int, default=512)
    p_sg.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.command == "roc":
        curve_paths = {}
        for item in args.curve:
            lesion, path = item.split("=", 1)
            curve_paths[lesion] = path
        plot_roc_curves(curve_paths, args.out)
    elif args.command == "confusion":
        plot_confusion_matrix(args.predictions_csv, args.out, args.dataset_name)
    elif args.command == "training-curve":
        if args.history_csv:
            epochs, kappas, accuracies = parse_history_csv(args.history_csv)
        elif args.log_txt:
            epochs, kappas, accuracies = parse_drgnet_log_txt(args.log_txt)
        else:
            parser.error("training-curve needs --history-csv or --log-txt")
        plot_training_curve(epochs, kappas, accuracies, args.out,
                             title=f"{args.dataset_name} classification: kappa/accuracy vs. epoch")
    elif args.command == "seg-grid":
        plot_segmentation_grid(args.fgadr_root, args.filtered_csv, args.checkpoint, args.lesion,
                                args.n_examples, args.out, image_size=args.image_size)


if __name__ == "__main__":
    main()
