"""
Round 6: retina (field-of-view) masks for Module 2 tensors.

Why: after the round-5 crop fix the retina fills the frame, so an enlarging SIFT warp clips the
follow-up's circle into a rounded square and leaves black fill. Round 5 trained G with an
unmasked L1 against those targets and showed the same targets to the critic as "real", so G
learned square, shifted edges -- and the cascade fed them back in, step after step.

The fixes built on this module:
  * fov_mask(img)            -- [B,1,H,W] {0,1} mask of where an image shows retina.
  * apply_fov(img, mask)     -- black (-1) outside the mask.
  * masked_l1(a, b, region)  -- L1 over the region only.
  * FovMaskedGenerator(G)    -- G's output blacked out outside its INPUT's retina circle.
  * CopyBaselineGenerator()  -- the "copy the baseline" control: returns its input unchanged.

Round 7 (cascade clean-up, inference only -- training is unchanged):
  * shrink_soften(mask, erode_px, feather_px) -- pulls the baseline's retina mask in by a few
    pixels and fades its edge, so the thin bright/coloured bars G draws along the circle's left
    and right edges (round 6, nb 08/10) fall outside it.
  * match_color(img, ref, region)             -- per-channel mean/std of img matched to ref inside
    the retina, so cascade steps don't drift grey/desaturated step after step.
  * finish_step(img, ref, fov, color)         -- what synthesize_trajectory does to every step.

Images are in the Generator's [-1, 1] range throughout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# A pixel shows retina if its brightest channel is above this ([0, 1] scale; ~10/255, the same
# threshold fundus_preprocess.retina_box uses on the green channel of the full-size photo).
FOV_THRESHOLD = 0.04


def _fill_holes(m):
    """Fills every background region not connected to the image border (dark lesions, vessels,
    the dark blobs a Generator can draw) -- the retina is one disc, so anything enclosed by it is
    retina. m: [B,1,H,W] {0,1} float."""
    bg = 1.0 - m
    reach = torch.zeros_like(bg)
    reach[..., 0, :] = bg[..., 0, :]
    reach[..., -1, :] = bg[..., -1, :]
    reach[..., :, 0] = bg[..., :, 0]
    reach[..., :, -1] = bg[..., :, -1]
    for i in range(m.shape[-1] + m.shape[-2]):
        new = F.max_pool2d(reach, 3, stride=1, padding=1) * bg
        if i % 8 == 7 and torch.equal(new, reach):
            break
        reach = new
    return 1.0 - reach


@torch.no_grad()
def fov_mask(img, threshold=FOV_THRESHOLD):
    """img: [B,3,H,W] in [-1, 1] -> [B,1,H,W] float {0,1}: 1 where the image shows retina.
    Thresholds the brightest channel, removes 1-2 px specks of background noise (opening), then
    fills holes so dark content inside the retina never counts as background."""
    x01 = (img.detach().float() + 1.0) / 2.0
    m = (x01.amax(dim=1, keepdim=True) > threshold).float()
    m = -F.max_pool2d(-m, 3, stride=1, padding=1)  # erode
    m = F.max_pool2d(m, 3, stride=1, padding=1)  # dilate (opening removes isolated specks)
    return _fill_holes(m)


def apply_fov(img, mask):
    """Black ([-1]) outside mask; mask [B,1,H,W] broadcasts over the channels."""
    return img * mask + (mask - 1.0)


def masked_l1(a, b, region):
    """Mean |a - b| over the pixels (x 3 channels) inside region; plain L1 if region is None."""
    if region is None:
        return (a - b).abs().mean()
    denom = region.sum() * a.shape[1]
    return ((a - b).abs() * region).sum() / denom.clamp_min(1.0)


def shrink_soften(mask, erode_px=0, feather_px=0):
    """Round 7: mask [B,1,H,W] {0,1} -> pulled in by erode_px pixels, then a soft edge feather_px
    wide that fades from 1 (inside) to 0 at the eroded edge -- it never reaches outside the eroded
    circle. 0/0 = the mask unchanged (round 6). Pixels are at the Module 2 image size (128 px)."""
    m = mask.float()
    if erode_px > 0:
        m = -F.max_pool2d(-m, 2 * erode_px + 1, stride=1, padding=erode_px)
    if feather_px > 0:
        inner = -F.max_pool2d(-m, 2 * feather_px + 1, stride=1, padding=feather_px)
        m = F.avg_pool2d(inner, 2 * feather_px + 1, stride=1, padding=feather_px, count_include_pad=True)
    return m.clamp(0.0, 1.0)


@torch.no_grad()
def match_color(img, ref, region, eps=1e-4):
    """Round 7: per-channel mean/std of img matched to ref's inside region ([B,1,H,W], > 0.5 = in).
    Global per channel, so local changes G makes (lesions) keep their contrast; only the overall
    colour cast and saturation are pulled back to the real baseline. Pixels outside region are
    left as they are (finish_step blacks them out). Images in [-1, 1]."""
    r = (region > 0.5).float()
    n = r.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)

    def stats(x):
        mu = (x * r).sum(dim=(2, 3), keepdim=True) / n
        sd = (((x - mu) ** 2) * r).sum(dim=(2, 3), keepdim=True).div(n).sqrt()
        return mu, sd

    mu_i, sd_i = stats(img)
    mu_r, sd_r = stats(ref)
    out = (img - mu_i) / (sd_i + eps) * sd_r + mu_r
    return torch.where(r.bool().expand_as(img), out, img).clamp(-1.0, 1.0)


def finish_step(img, ref, fov=None, color_match=False):
    """Round 7: one synthesized cascade step -> colour-matched to the REAL baseline (ref; always the
    baseline, never the previous step, so the error can't build up) and blacked out outside fov
    (the baseline's possibly shrunk/softened retina mask). fov None and color_match False = raw G."""
    if color_match:
        img = match_color(img, ref, fov if fov is not None else fov_mask(ref))
    if fov is not None:
        img = apply_fov(img, fov)
    return img


class FovMaskedGenerator(nn.Module):
    """Wraps a Generator: same call signature G(x, c), output blacked out outside the retina
    circle of the input image (x[:, :3]). Parameters and state_dict stay those of the wrapped G
    (save/load the inner .G). The cascade (trajectory_inference.synthesize_trajectory) passes the
    baseline's mask explicitly instead, so the circle can't shrink from step to step."""

    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, x, c):
        out = self.G(x, c)
        return apply_fov(out, fov_mask(x[:, :3]))


class CopyBaselineGenerator(nn.Module):
    """The copy-the-baseline control: 'synthesizes' every stage by returning the input image
    unchanged. Scored through the same metrics, it is the floor a real Generator has to beat --
    a low FID/KID from copying only means the output is a real photo, not that it progressed."""

    def forward(self, x, c):
        return x[:, :3]


def add_cascade_args(parser):
    """Round 7 command-line options shared by evaluate_trajectory.py, visualize_trajectory.py and
    generate_progression.py (all off by default = round 6). --fov-erode/--fov-feather need --fov-mask."""
    parser.add_argument("--fov-erode", type=int, default=0,
                        help="Round 7: pull the baseline's retina mask in by this many pixels (at --image-size)")
    parser.add_argument("--fov-feather", type=int, default=0,
                        help="Round 7: fade the mask's edge to black over this many pixels")
    parser.add_argument("--color-match", action="store_true",
                        help="Round 7: match every synthesized step's colours to the real baseline's")
    parser.add_argument("--cascade-mode", choices=["cascade", "direct"], default="cascade",
                        help="Round 7: 'direct' generates every stage from the real baseline (evaluation control)")


def check_cascade_args(parser, args):
    if (args.fov_erode or args.fov_feather) and not args.fov_mask:
        parser.error("--fov-erode / --fov-feather need --fov-mask")


def cascade_fov(baseline, config):
    """The fov mask synthesize_trajectory gets for this baseline: None without --fov-mask, else the
    baseline's retina mask, shrunk/softened by --fov-erode/--fov-feather (round 7)."""
    if not getattr(config, "fov_mask", False):
        return None
    return shrink_soften(fov_mask(baseline), getattr(config, "fov_erode", 0), getattr(config, "fov_feather", 0))


def cascade_kwargs(baseline, config):
    """fov_mask / color_match / mode keyword arguments for synthesize_trajectory."""
    return {"fov_mask": cascade_fov(baseline, config),
            "color_match": bool(getattr(config, "color_match", False)),
            "mode": getattr(config, "cascade_mode", "cascade")}
