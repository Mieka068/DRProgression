"""
Autoregressive multi-stage trajectory synthesis -- the manuscript's "full multi-stage
trajectory from a single baseline image" claim, which the base DRForecastGAN/StarGAN
architecture does not do on its own (it only ever does one G(x, c) call; see
docs/IMPLEMENTATION_PLAN.md Task D and CLAUDE.md's "What this project is"). This is this
thesis's actual novelty contribution on top of the verified DRForecastGAN base in
base_model.py, not something recovered from upstream code.

Per the manuscript's Section 3.3.7 (docs/IMPLEMENTATION_PLAN.md's revised Task D), each
cascade step is a 4-part loop, NOT a bare repeated generator call:
  1. Generate the next-stage image from the current image + current mask.
  2. Re-segment the synthesized image with Module 1 to get the NEXT step's mask -- lesion
     distribution changes across stages, so carrying the baseline mask forward (an earlier,
     simpler version of this function did that) would misalign the conditioning signal with
     the synthesized image it's supposedly describing.
  3. Re-grade the synthesized image with Module 1's classifier as an internal consistency
     check: does the synthesized image actually look like the target stage to Module 1's own
     grader? Logged, not used to alter generation -- a real "does the model's own severity
     grader agree with what we told the generator to produce" measurement.
  4. Feed the new image + new mask into the next iteration.

This is confirmed absent from the official DRForecastGAN release and this project's prior
code -- new engineering, not something recovered from upstream.
"""
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module1"))
from apply_to_progression_data import process_image  # noqa: E402


def _tensor_to_pil(image_tensor):
    """
    Converts a single-image [1,3,H,W] or [3,H,W] tensor in the GAN's [-1,1] Tanh-output range
    to a PIL RGB image. This is a deliberate temp-file round trip (see _module1_predict)
    rather than a second, tensor-native reimplementation of Module 1's preprocessing --
    guarantees IDENTICAL preprocessing to Module 1's real-image predictions instead of risking
    the two pipelines silently drifting apart.
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    denorm = ((image_tensor + 1) / 2).clamp(0, 1)
    arr = (denorm.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _module1_predict(image_tensor, classifier, seg_models, device, mask_size):
    """
    Runs Module 1's classifier + segmentation heads on a synthesized image tensor, via a
    temp-file round trip through module1/apply_to_progression_data.py's own process_image()
    -- see _tensor_to_pil's docstring for why. Returns (predicted_grade, mask_tensor[1,1,H,W]).
    """
    pil_image = _tensor_to_pil(image_tensor)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        pil_image.save(tmp_path)
        result = process_image(tmp_path, classifier, seg_models, device=device)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    mask_np = result["mask"]
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
        (mask_size, mask_size), Image.NEAREST
    )
    mask_tensor = torch.from_numpy(np.array(mask_img) > 0).float().unsqueeze(0).unsqueeze(0)
    return result["grade"], mask_tensor


def synthesize_trajectory(generator, classifier, seg_models, baseline_image, baseline_mask,
                           start_stage, end_stage, c_dim, device):
    """
    Generates a full stage-by-stage trajectory from start_stage to end_stage, per the
    manuscript's Sec 3.3.7 4-step loop (see module docstring) -- NOT a bare repeated generator
    call, and NOT a baseline-mask-reuse simplification (an earlier version of this function
    did that; this version replaces it).

    Args:
        generator: a base_model.Generator instance (AdaIN-conditioned or not -- this function
            only depends on the G(x, c) call signature).
        classifier: a loaded Module 1 classifier
            (module1/apply_to_progression_data.py::load_classifier).
        seg_models: a {lesion_name: model_or_None} dict
            (module1/apply_to_progression_data.py::load_segmentation_model per lesion).
        baseline_image: [1, 3, H, W] real baseline fundus image tensor, in the generator's
            expected input range ([-1, 1], matching the existing tanh-output convention).
        baseline_mask: [1, 1, H, W] real Module 1 lesion mask for the baseline image. Used
            as-is for the baseline entry -- Module 1 is only re-run on SYNTHESIZED images,
            since the baseline already has a real mask.
        start_stage: int, the baseline image's own severity stage (0-indexed, ICDR 0-4).
        end_stage: int, the final stage to synthesize up to (inclusive). Must be > start_stage.
        c_dim: number of stages (5 for ICDR 0-4), matching the Generator's own c_dim.
        device: torch device to run inference on.

    Returns:
        list of dicts, one per stage from start_stage (the real baseline) through end_stage,
        each {'stage', 'image', 'mask', 'consistency'}. 'mask' is the real baseline_mask for
        the baseline entry and a fresh Module-1-predicted mask for every synthesized step.
        'consistency' is None for the baseline entry and a bool (predicted grade == target
        stage) for every synthesized step -- an internal Module-1-agreement signal, logged for
        evaluation, never used to alter generation.
    """
    if end_stage <= start_stage:
        raise ValueError(
            f"end_stage ({end_stage}) must be greater than start_stage ({start_stage}) -- "
            "there is nothing to synthesize otherwise."
        )
    if end_stage >= c_dim:
        raise ValueError(f"end_stage ({end_stage}) must be < c_dim ({c_dim}).")

    generator.eval()
    mask_size = baseline_mask.shape[-1]
    baseline_image = baseline_image.to(device)
    baseline_mask = baseline_mask.to(device)

    trajectory = [
        {"stage": start_stage, "image": baseline_image, "mask": baseline_mask, "consistency": None}
    ]
    current_image, current_mask = baseline_image, baseline_mask

    with torch.no_grad():
        for stage in range(start_stage + 1, end_stage + 1):
            c_target = torch.zeros(1, c_dim, device=device)
            c_target[0, stage] = 1.0
            x = torch.cat([current_image, current_mask], dim=1)
            synthesized = generator(x, c_target)

            predicted_grade, new_mask = _module1_predict(synthesized, classifier, seg_models, device, mask_size)
            new_mask = new_mask.to(device)
            consistent = predicted_grade == stage

            trajectory.append({"stage": stage, "image": synthesized, "mask": new_mask, "consistency": consistent})
            current_image, current_mask = synthesized, new_mask

    return trajectory


if __name__ == "__main__":
    sys.path.insert(0, ".")
    from base_model import Generator

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module1"))
    from apply_to_progression_data import ALL_LESIONS, load_classifier, load_segmentation_model

    print("Smoke-testing synthesize_trajectory (random weights/dry-run Module 1, no checkpoints)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(conv_dim=64, c_dim=5, repeat_num=6).to(device)
    classifier = load_classifier(None, device=device, dry_run=True)
    seg_models = {lesion: load_segmentation_model(None, device=device, dry_run=True) for lesion in ALL_LESIONS}

    baseline = torch.randn(1, 3, 128, 128)
    mask = torch.zeros(1, 1, 128, 128)
    trajectory = synthesize_trajectory(
        G, classifier, seg_models, baseline, mask, start_stage=0, end_stage=3, c_dim=5, device=device
    )

    for entry in trajectory:
        print(f"  stage {entry['stage']}: image {tuple(entry['image'].shape)}, "
              f"mask {tuple(entry['mask'].shape)}, consistency={entry['consistency']}")
    assert [e["stage"] for e in trajectory] == [0, 1, 2, 3]
    print("\n✓ synthesize_trajectory working!")
