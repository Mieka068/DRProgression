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
