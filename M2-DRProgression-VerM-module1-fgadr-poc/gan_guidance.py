"""
Round-4 additions to Module 2's training (train_module2_poc.py): a frozen Module 1 grader in the
Generator's loss, and DiffAugment for the critic.

Why (round-3 numbers): the Discriminator's own stage head graded real held-out images at 17.5 %
accuracy (balanced 27 %, chance 20 %), yet "agreed" with the requested stage on 74-100 % of the
Generator's outputs -- G learned to trigger D's weak head, not to draw stage features. Module 1's
grader re-read the synthesized cascade as the target stage only 1.5 % / 2.6 % of the time.

FrozenGrader (item R4-3.1): Module 1's ResNet-50 grader, frozen (eval mode, no gradients to its
weights), scores G's output; G gets lambda_grader x CE(grader(fake), target stage). Two guards
against the Generator gaming a fixed network with imperceptible patterns instead of drawing lesions:
  * every forward pass applies a fresh random, differentiable augmentation first (scale/rotation/
    shift, flip, brightness/contrast, blur) -- a pattern that fools the grader must survive all of
    them, which pixel-level adversarial noise does not;
  * the run is SELECTED and EVALUATED with a different grader (train with the EyePACS grader,
    select/evaluate with the FGADR one -- notebook 06), so the reported agreement is not the
    number the Generator was optimised on.

diff_augment (item R4-3.2): Zhao et al. 2020, "Differentiable Augmentation for Data-Efficient GAN
Training" -- the same random colour / translation / cutout applied to real and fake images on
every critic call (and to G's fake on the G step), so the critic cannot memorise the ~35
progressing pairs. Re-implemented here (policy names as in the paper's code).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- DiffAugment

def _rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)


def _rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + mean


def _rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + mean


def _rand_translation(x, ratio=0.125):
    b, _c, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sh, sh + 1, size=[b, 1, 1], device=x.device)
    ty = torch.randint(-sw, sw + 1, size=[b, 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=x.device), torch.arange(h, device=x.device),
                                torch.arange(w, device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0], value=-1.0)  # -1 = black in Module 2's range
    return x_pad.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _rand_cutout(x, ratio=0.25):
    b, _c, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), size=[b, 1, 1], device=x.device)
    oy = torch.randint(0, w + (1 - cw % 2), size=[b, 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=x.device), torch.arange(ch, device=x.device),
                                torch.arange(cw, device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - ch // 2, min=0, max=h - 1)
    gy = torch.clamp(gy + oy - cw // 2, min=0, max=w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1) + (mask.unsqueeze(1) - 1)  # cut region -> -1 (black)


_POLICIES = {"color": [_rand_brightness, _rand_saturation, _rand_contrast],
             "translation": [_rand_translation], "cutout": [_rand_cutout]}


def diff_augment(x, policy="color,translation,cutout"):
    """Random differentiable augmentation of a [-1, 1] NCHW batch (a new draw on every call)."""
    if not policy:
        return x
    for name in policy.split(","):
        name = name.strip()
        if name not in _POLICIES:
            raise ValueError(f"unknown DiffAugment policy '{name}' (choose from {list(_POLICIES)})")
        for fn in _POLICIES[name]:
            x = fn(x)
    return x.contiguous()


# ----------------------------------------------------------------------------- frozen grader

def _gaussian_kernel(sigma, device):
    radius = max(1, int(math.ceil(2.5 * sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum(), radius


def grader_augment(x01, max_rot_deg=10.0, scale=(0.9, 1.1), max_shift=0.05, jitter=0.15, blur_p=0.3):
    """Random differentiable view of a [0, 1] NCHW batch: affine (rotation/scale/shift), h-flip,
    brightness/contrast, occasional Gaussian blur. Outside the image = black."""
    b, _c, _h, _w = x01.shape
    dev = x01.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    s = torch.empty(b, device=dev).uniform_(*scale)
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    tx, ty = [(torch.rand(b, device=dev) * 2 - 1) * max_shift * 2 for _ in range(2)]
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x01.shape), align_corners=False)
    x = F.grid_sample(x01, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    bright = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * jitter
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * jitter
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = ((x - mean) * contrast + mean) * bright
    if blur_p > 0 and torch.rand(()) < blur_p:
        k, r = _gaussian_kernel(float(torch.empty(()).uniform_(0.4, 1.2)), dev)
        c = x.shape[1]
        x = F.conv2d(F.pad(x, [r, r, 0, 0], mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
        x = F.conv2d(F.pad(x, [0, 0, r, r], mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x.clamp(0, 1)


class FrozenGrader(nn.Module):
    """Module 1's grader as a fixed, differentiable critic of G's output (see module docstring).
    forward(x) takes G's [-1, 1] output and returns the grader's logits."""

    def __init__(self, checkpoint_path, norm="eyepacs", input_size=512, device="cpu", augment=True):
        super().__init__()
        from apply_to_progression_data import CLASSIFIER_NORMS, load_classifier
        self.net = load_classifier(checkpoint_path, device=device)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        mean, std = CLASSIFIER_NORMS[norm]
        self.register_buffer("mean", torch.tensor(mean, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, device=device).view(1, 3, 1, 1))
        self.input_size = input_size
        self.augment = augment

    def train(self, mode=True):  # the grader stays in eval mode (frozen BatchNorm statistics)
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, x_pm1):
        x = ((x_pm1 + 1) / 2).clamp(0, 1)
        if self.augment:
            x = grader_augment(x)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)
