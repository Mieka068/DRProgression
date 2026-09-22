"""
Runs a trained Module 1 classification checkpoint over a classification pkl index's 'test'
split (the {'train','val','test': [(image_path, grade)]} format built by
dataprep/make_fgadr_classification_pkl.py / make_idrid_classification_pkl.py) and writes
per-image (true_grade, pred_grade) pairs to a CSV -- input for plot_figures.py's confusion
matrix figure. Reuses apply_to_progression_data.py's own classifier loading/inference code
(same resnet50 architecture, same checkpoint format) rather than going through DRG-Net's own
eval pipeline, matching this folder's "thin adapter" approach elsewhere.

Usage:
    python dump_classification_predictions.py \
        --pkl ./fgadr_classification_pkl.pkl \
        --checkpoint /path/to/final_weights.pt \
        --dataset-name FGADR \
        --out ./predictions_fgadr.csv
"""
import argparse
import csv
import pickle

import torch

from apply_to_progression_data import FGADR_MEAN, FGADR_STD, load_classifier, predict_grade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Classification pkl index (see module docstring)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", required=True, help="Label written into the CSV, e.g. FGADR or IDRiD")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mean", type=float, nargs=3, default=FGADR_MEAN,
                         help="Override for IDRiD (trained with 'auto' dataset stats -- see "
                              "apply_to_progression_data.py's own note on this)")
    parser.add_argument("--std", type=float, nargs=3, default=FGADR_STD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    items = data[args.split]
    print(f"Loaded {len(items)} ({args.split}) images from {args.pkl}")

    classifier = load_classifier(args.checkpoint, device=args.device)

    rows = []
    for image_path, true_grade in items:
        pred_grade = predict_grade(
            classifier, image_path, image_size=args.image_size, mean=args.mean, std=args.std, device=args.device
        )
        rows.append((args.dataset_name, image_path, int(true_grade), int(pred_grade)))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "image_path", "true_grade", "pred_grade"])
        writer.writerows(rows)

    n_correct = sum(1 for r in rows if r[2] == r[3])
    print(f"✓ Wrote {len(rows)} predictions to {args.out} (raw accuracy: {n_correct / len(rows):.4f})")


if __name__ == "__main__":
    main()
