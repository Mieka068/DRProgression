"""
Autoregressive multi-stage trajectory synthesis -- the manuscript's "full multi-stage
trajectory from a single baseline image" claim, which the base DRForecastGAN/StarGAN
architecture does not do on its own (it only ever does one G(x, c) call; see
docs/IMPLEMENTATION_PLAN.md Task D and CLAUDE.md's "What this project is"). This is this
thesis's actual novelty contribution on top of the verified DRForecastGAN base in
base_model.py, not something recovered from upstream code.

Mechanism: feed each generated image back in as the input for the next step's generation,
walking one target stage at a time from start_stage+1 to end_stage.
"""
import torch


def synthesize_trajectory(generator, baseline_image, mask, start_stage, end_stage, c_dim, device):
    """
    Generates a full stage-by-stage trajectory from start_stage to end_stage by feeding each
    generated image back in as the next step's input.

    Mask handling (a real, documented design choice, not an oversight): this v1 reuses the
    ORIGINAL baseline lesion mask at every step rather than re-segmenting each synthetic
    intermediate image with Module 1. Re-segmenting synthetic images is a reasonable future
    improvement but adds a second source of compounding error (Module 1 predictions on
    increasingly out-of-distribution synthetic images) on top of the generator's own
    compounding error across steps -- keeping those two error sources separate lets
    evaluation isolate which one dominates (see docs/IMPLEMENTATION_PLAN.md's "Open decisions"
    section; revisit only if per-step evaluation shows the reused mask is clearly the
    bottleneck, not by default).

    Args:
        generator: a base_model.Generator instance (AdaIN-conditioned or not -- this function
            only depends on the G(x, c) call signature, so it works with either).
        baseline_image: [1, 3, H, W] real baseline fundus image tensor, in the generator's
            expected input range (matches whatever the generator was trained on, e.g. [-1, 1]
            for the existing tanh-output convention).
        mask: [1, 1, H, W] lesion mask tensor for the baseline image (from Module 1, or an
            empty placeholder -- same conventions as elsewhere in this pipeline).
        start_stage: int, the baseline image's own severity stage (0-indexed, ICDR 0-4).
        end_stage: int, the final stage to synthesize up to (inclusive). Must be > start_stage.
        c_dim: number of stages (5 for ICDR 0-4), matching the Generator's own c_dim.
        device: torch device to run inference on.

    Returns:
        list of (stage, image_tensor) for start_stage+1 ... end_stage, in order. Does NOT
        include the baseline itself -- callers that want the full sequence for plotting can
        prepend (start_stage, baseline_image) themselves.
    """
    if end_stage <= start_stage:
        raise ValueError(
            f"end_stage ({end_stage}) must be greater than start_stage ({start_stage}) -- "
            "there is nothing to synthesize otherwise."
        )
    if end_stage >= c_dim:
        raise ValueError(f"end_stage ({end_stage}) must be < c_dim ({c_dim}).")

    generator.eval()
    trajectory = []
    current_image = baseline_image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        for stage in range(start_stage + 1, end_stage + 1):
            c_target = torch.zeros(1, c_dim, device=device)
            c_target[0, stage] = 1.0
            x = torch.cat([current_image, mask], dim=1)
            current_image = generator(x, c_target)
            trajectory.append((stage, current_image))
    return trajectory


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from base_model import Generator

    print("Smoke-testing synthesize_trajectory (random weights, no checkpoint)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(conv_dim=64, c_dim=5, repeat_num=6).to(device)

    baseline = torch.randn(1, 3, 128, 128)
    mask = torch.zeros(1, 1, 128, 128)
    trajectory = synthesize_trajectory(G, baseline, mask, start_stage=0, end_stage=3, c_dim=5, device=device)

    for stage, image in trajectory:
        print(f"  stage {stage}: {image.shape}")
    assert [s for s, _ in trajectory] == [1, 2, 3]
    print("\n✓ synthesize_trajectory working!")
