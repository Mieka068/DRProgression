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

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from drive_io import local_twin, push_to_drive, restore_from_drive  # noqa: E402

# Classifier input normalisation must match the config the checkpoint was trained with:
# fgadr_poc.yaml (default) or eyepacs_poc.yaml (DRG-Net's eyepacs_ce.yaml stats). Select with
# --classifier-norm; idrid_poc.yaml uses 'auto' (dataset-computed stats DRG-Net prints at
# train time) -- pass those via --classifier-norm custom --mean ... --std ... if you use it.
FGADR_MEAN = [0.4577976167201996, 0.2596849203109741, 0.13276457786560059]
FGADR_STD = [0.28394654393196106, 0.18262696266174316, 0.13211798667907715]
EYEPACS_MEAN = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
EYEPACS_STD = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
CLASSIFIER_NORMS = {"fgadr": (FGADR_MEAN, FGADR_STD), "eyepacs": (EYEPACS_MEAN, EYEPACS_STD)}
# Module-level so callers that go through process_image() (trajectory_inference.py,
# generate_progression.py) pick up the same setting -- see set_classifier_norm().
CLASSIFIER_MEAN, CLASSIFIER_STD = FGADR_MEAN, FGADR_STD


def set_classifier_norm(name="fgadr", mean=None, std=None):
    global CLASSIFIER_MEAN, CLASSIFIER_STD
    if mean is not None and std is not None:
        CLASSIFIER_MEAN, CLASSIFIER_STD = list(mean), list(std)
    else:
        CLASSIFIER_MEAN, CLASSIFIER_STD = CLASSIFIER_NORMS[name]
    print(f"Classifier normalisation: {name} mean={CLASSIFIER_MEAN} std={CLASSIFIER_STD}")


def add_classifier_norm_args(parser):
    parser.add_argument("--classifier-norm", choices=list(CLASSIFIER_NORMS) + ["custom"], default="fgadr",
                        help="Normalisation the classifier checkpoint was trained with "
                             "(fgadr_poc.yaml = fgadr, eyepacs_poc.yaml = eyepacs)")
    parser.add_argument("--mean", type=float, nargs=3, default=None, help="With --classifier-norm custom")
    parser.add_argument("--std", type=float, nargs=3, default=None, help="With --classifier-norm custom")


def apply_classifier_norm_args(args):
    if args.classifier_norm == "custom" and (args.mean is None or args.std is None):
        raise SystemExit("--classifier-norm custom needs --mean R G B and --std R G B")
    set_classifier_norm(args.classifier_norm, args.mean, args.std)

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


def predict_grade(model, image_path, image_size=512, mean=None, std=None, device="cpu",
                   return_probs=False):
    mean = CLASSIFIER_MEAN if mean is None else mean
    std = CLASSIFIER_STD if std is None else std
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


def lesion_stats(lesion_mask: np.ndarray, fov_mask: np.ndarray) -> dict:
    """
    Per-lesion biomarkers for one binary lesion mask: area fraction of the retinal FOV (the
    same per-lesion LBS) and the number of connected components (lesion count). These are what
    stage_biomarker_profiles.py aggregates per DR stage on cross-sectional data (EyePACS etc.).
    """
    import cv2

    n_components, _ = cv2.connectedComponents(lesion_mask.astype(np.uint8), connectivity=8)
    return {
        "area_frac": compute_lbs(lesion_mask, fov_mask),
        "count": int(n_components - 1),  # label 0 is background
    }


def process_image(image_path, classifier, seg_models: dict, device="cpu"):
    grade, grade_probs = predict_grade(classifier, image_path, device=device, return_probs=True)
    rgb = np.array(Image.open(image_path).convert("RGB"))
    fov_mask = estimate_retinal_fov_mask(rgb)

    combined_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    trained_lesions = []
    per_lesion = {}
    for lesion_name, model in seg_models.items():
        if model is None:
            continue
        m = predict_lesion_mask(model, image_path, device=device)
        combined_mask = np.logical_or(combined_mask, m).astype(np.uint8)
        trained_lesions.append(lesion_name)
        per_lesion[lesion_name] = lesion_stats(m, fov_mask)

    lbs = compute_lbs(combined_mask, fov_mask)
    return {
        "grade": grade,
        "grade_probs": grade_probs,  # per-class softmax scores, for AUC-based consistency (evaluate_trajectory.py)
        "mask": combined_mask,
        "lbs": lbs,
        "trained_lesions": trained_lesions,
        "lesion_stats": per_lesion,  # {lesion: {area_frac, count}} -- only for trained lesions
    }


def downsample_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of a 0/1 mask to size x size (all Module 2 loaders resize to
    image_size anyway, so storing a full-resolution mask is only needed for LBS, which is
    already computed by then)."""
    resized = Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return (np.array(resized) > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True,
                        help="FIRE/LongDR/Tianjin/EyePACS images folder to run inference over")
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
    parser.add_argument(
        "--mask-size",
        type=int,
        default=None,
        help="Store each mask downsampled to N x N instead of full resolution. Recommended "
             "(e.g. 256) for large cross-sectional sets like EyePACS (~35k images), where "
             "full-res masks would make the cache tens of GB. LBS/lesion_stats are always "
             "computed at full resolution first.",
    )
    parser.add_argument(
        "--image-list",
        default=None,
        help="Optional text file of image file names/stems (one per line) to process instead of "
             "everything under --images-dir -- e.g. a per-grade EyePACS subset (notebook 09).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If --out already exists, load it and skip images already in it (for long runs "
             "that get cut off by a Colab disconnect).",
    )
    parser.add_argument("--save-every", "--checkpoint-every", dest="save_every", type=int, default=0,
                        help="Write the partial cache to --out every N new images (0 = only at the "
                             "end). --checkpoint-every is the same flag (notebook 04_05's name).")
    parser.add_argument("--drive-every", type=int, default=2000,
                        help="When --out is on Google Drive: the partial cache is written every "
                             "--save-every images to a copy on the VM disk, and copied to Drive only "
                             "every --drive-every new images and at the end (each Drive overwrite leaves "
                             "an old copy counted against the quota -- see drive_io.py).")
    parser.add_argument("--quiet", action="store_true", help="Only print progress every 100 images")
    add_classifier_norm_args(parser)
    args = parser.parse_args()
    apply_classifier_norm_args(args)

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
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.relpath(os.path.join(root, f), args.images_dir))
    image_files.sort()
    if args.image_list:
        with open(args.image_list) as f:
            wanted = {os.path.splitext(os.path.basename(line.strip()))[0] for line in f if line.strip()}
        image_files = [r for r in image_files if os.path.splitext(os.path.basename(r))[0] in wanted]
        print(f"--image-list: {len(image_files)} of {len(wanted)} listed images found under {args.images_dir}")
    if args.limit:
        image_files = image_files[: args.limit]

    cache = {}
    work = local_twin(args.out)  # == args.out unless --out is on Drive
    os.makedirs(os.path.dirname(os.path.abspath(work)), exist_ok=True)
    if args.resume:
        restore_from_drive(args.out)
        if os.path.isfile(work):
            cache = torch.load(work, weights_only=False)
            print(f"--resume: {len(cache)} images already cached in {args.out}, skipping them")

    n_new = 0
    for i, relpath in enumerate(image_files):
        image_path = os.path.join(args.images_dir, relpath)
        image_id = os.path.splitext(relpath)[0].replace(os.sep, "__")
        if image_id in cache:
            continue
        try:
            result = process_image(image_path, classifier, seg_models, device=device)
            mask = result["mask"]
            if args.mask_size:
                mask = downsample_mask(mask, args.mask_size)
            cache[image_id] = {
                "grade": result["grade"],
                "mask": mask,
                "lbs": result["lbs"],
                "trained_lesions": result["trained_lesions"],
                "lesion_stats": result["lesion_stats"],
            }
            n_new += 1
            if not args.quiet or n_new % 100 == 0:
                print(
                    f"✓ [{i + 1}/{len(image_files)}] {image_id}: grade={result['grade']} "
                    f"lbs={result['lbs']:.5f} (lesions: {result['trained_lesions']})"
                )
            if args.save_every and n_new % args.save_every == 0:
                torch.save(cache, work)
                if work != args.out and n_new % args.drive_every < args.save_every:
                    push_to_drive(args.out)
        except Exception as e:
            print(f"⚠ {image_id}: failed ({e})")

    torch.save(cache, work)
    push_to_drive(args.out)
    print(f"✓ wrote {len(cache)} entries to {args.out}")


if __name__ == "__main__":
    main()
