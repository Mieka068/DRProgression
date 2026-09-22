"""
Bridge: run trained Module 1 outputs (grade classifier + per-lesion segmentation models) over
FIRE / LongDRScreening images, and cache {image_id: {grade, mask, lbs}} to disk. This is what
fire_dataset.py / longdr_dataset.py's new `module1_cache_path` argument consumes, replacing
today's placeholder empty-mask / fixed-Stage-2-grade with real Module 1 output.

Architectures are reconstructed to exactly match DRG-Net's own instantiation code (not
guessed): classifier = torchvision resnet50 + `fc` replaced with a `num_classes`-way linear
head (dr_classification/modules/builder.py::build_model, network: resnet50); segmentation =
segmentation_models_pytorch Unet w/ resnet50 encoder, 2 output classes
(dr_segmentation/train_fgadr.py). Checkpoint formats also match what those scripts save:
classifier checkpoints are a bare state_dict (utils/func.py::save_weights); segmentation
checkpoints are {'epoch','step','state_dict','optimizer'} (train_fgadr.py::train_model).

For lesion classes with no --seg-checkpoint given (e.g. HE/SE before their POC run finishes),
that channel is simply left out of the combined mask -- FIRE/LongDR have no ground truth to
substitute, so the combined mask honestly reflects only what's currently trained. Re-run with
more --seg-checkpoint entries once available; no code change needed.

Usage:
    python apply_to_progression_data.py \
        --images-dir ./FIRE_dataset/FIRE/Images \
        --classifier-checkpoint /path/to/final_weights.pt \
        --seg-checkpoint EX=/path/to/model_EX.pth.tar \
        --seg-checkpoint MA=/path/to/model_MA.pth.tar \
        --out ./module1_outputs_fire.pt

Dry run (no checkpoints needed, random-init weights -- for smoke-testing the plumbing before
any training has finished):
    python apply_to_progression_data.py --images-dir ./FIRE_dataset/FIRE/Images --dry-run \
        --out /tmp/dry_run.pt --limit 3
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from compute_lbs import compute_lbs, estimate_retinal_fov_mask

# Defaults match module1/configs/fgadr_poc.yaml's data.mean/data.std. Override with --mean/--std
# if the classifier checkpoint was instead trained on IDRiD (that config uses 'auto', i.e.
# dataset-computed statistics not reproduced here -- ImageNet defaults are a reasonable stand-in).
FGADR_MEAN = [0.4577976167201996, 0.2596849203109741, 0.13276457786560059]
FGADR_STD = [0.28394654393196106, 0.18262696266174316, 0.13211798667907715]

ALL_LESIONS = ["EX", "HE", "MA", "SE"]


def load_classifier(checkpoint_path, num_classes=5, device="cpu", dry_run=False):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if not dry_run:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def load_segmentation_model(checkpoint_path, device="cpu", dry_run=False):
    import segmentation_models_pytorch as smp

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=2)
    if not dry_run:
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def predict_grade(model, image_path, image_size=512, mean=FGADR_MEAN, std=FGADR_STD, device="cpu",
                   return_probs=False):
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        grade = int(torch.argmax(logits, dim=1).item())
    if return_probs:
        return grade, probs
    return grade


def predict_lesion_mask(model, image_path, image_size=512, device="cpu"):
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (W, H)
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        mask = (probs[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
    mask_img = Image.fromarray(mask * 255).resize(orig_size, Image.NEAREST)
    return (np.array(mask_img) > 0).astype(np.uint8)


def process_image(image_path, classifier, seg_models: dict, device="cpu"):
    grade, grade_probs = predict_grade(classifier, image_path, device=device, return_probs=True)
    rgb = np.array(Image.open(image_path).convert("RGB"))
    fov_mask = estimate_retinal_fov_mask(rgb)

    combined_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    trained_lesions = []
    for lesion_name, model in seg_models.items():
        if model is None:
            continue
        m = predict_lesion_mask(model, image_path, device=device)
        combined_mask = np.logical_or(combined_mask, m).astype(np.uint8)
        trained_lesions.append(lesion_name)

    lbs = compute_lbs(combined_mask, fov_mask)
    return {
        "grade": grade,
        "grade_probs": grade_probs,  # per-class softmax scores, for AUC-based consistency (evaluate_trajectory.py)
        "mask": combined_mask,
        "lbs": lbs,
        "trained_lesions": trained_lesions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="FIRE/LongDR Images folder to run inference over")
    parser.add_argument("--classifier-checkpoint", help="Required unless --dry-run")
    parser.add_argument(
        "--seg-checkpoint",
        action="append",
        default=[],
        metavar="LESION=PATH",
        help="Repeatable, e.g. --seg-checkpoint EX=/path/model_EX.pth.tar --seg-checkpoint MA=/path/...",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, help="Output cache path, e.g. module1_outputs_fire.pt")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N images (debug)")
    parser.add_argument("--dry-run", action="store_true", help="Random-init weights, no checkpoints needed")
    args = parser.parse_args()

    if not args.dry_run and not args.classifier_checkpoint:
        parser.error("--classifier-checkpoint is required unless --dry-run is set")

    device = args.device
    classifier = load_classifier(args.classifier_checkpoint, device=device, dry_run=args.dry_run)

    seg_models = {lesion: None for lesion in ALL_LESIONS}
    for entry in args.seg_checkpoint:
        lesion, path = entry.split("=", 1)
        seg_models[lesion] = load_segmentation_model(path, device=device, dry_run=args.dry_run)
    if args.dry_run and not args.seg_checkpoint:
        # exercise at least one segmentation head so the dry run covers both code paths
        seg_models["EX"] = load_segmentation_model(None, device=device, dry_run=True)

    # Walk recursively so this works both for FIRE's flat Images/ folder and LongDRScreening's
    # nested eye_XXX/visit_N_image_M.png layout. image_id is the path relative to images_dir
    # with the extension stripped and os.sep replaced by "__" (e.g. FIRE: "A01_1"; LongDR:
    # "eye_003__visit_1_image_1") -- longdr_dataset.py builds the same id when looking the
    # cache up, so keep the two in sync if this scheme ever changes.
    image_files = []
    for root, _dirs, files in os.walk(args.images_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".png")):
                image_files.append(os.path.relpath(os.path.join(root, f), args.images_dir))
    image_files.sort()
    if args.limit:
        image_files = image_files[: args.limit]

    cache = {}
    for relpath in image_files:
        image_path = os.path.join(args.images_dir, relpath)
        image_id = os.path.splitext(relpath)[0].replace(os.sep, "__")
        try:
            result = process_image(image_path, classifier, seg_models, device=device)
            cache[image_id] = {
                "grade": result["grade"],
                "mask": result["mask"],
                "lbs": result["lbs"],
                "trained_lesions": result["trained_lesions"],
            }
            print(
                f"✓ {image_id}: grade={result['grade']} lbs={result['lbs']:.5f} "
                f"(lesions: {result['trained_lesions']})"
            )
        except Exception as e:
            print(f"⚠ {image_id}: failed ({e})")

    torch.save(cache, args.out)
    print(f"✓ wrote {len(cache)} entries to {args.out}")


if __name__ == "__main__":
    main()
