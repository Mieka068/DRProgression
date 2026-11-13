"""
Module 2 Generator / Discriminator (StarGAN / DRForecastGAN base, AdaIN stage conditioning).

Architecture changes on branch AllModuleEdits-PreFinals (CODE_FIXES items 3.4, 3.6, 4.1, 4.2), all
on by default -- MODEL_ARCHS['prefinals']:
  * Discriminator: spectral normalization on its conv layers (3.4; not on the stage class head).
  * Generator: InstanceNorm2d(track_running_stats=False) (3.6) -- with running stats, eval()
    normalised with dataset-average statistics instead of the image's own, unlike training.
  * Generator up-sampling: nearest-neighbour Upsample x2 + 3x3 conv instead of ConvTranspose2d (4.1)
    -- transposed convs with stride 2 produce the checkerboard pattern seen in the samples.
  * Generator: U-Net skip connections from each down-sampling stage to the matching up-sampling stage
    (4.2), so fine detail need not pass through the 32x32 (at 128 px) bottleneck; AdaIN stays in it.
These change the state_dict, so checkpoints from before them (mod-2-NonLongIntegration, including the
EyePACS pretraining) do NOT load into the default models -- load_checkpoint() then raises with a
pointer to --model-arch legacy, which rebuilds the earlier architecture (MODEL_ARCHS['legacy']) so
those checkpoints still load for before/after comparisons. Generator and Discriminator kwargs can
also be set one by one.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm

# Per-architecture constructor kwargs: (Generator kwargs, Discriminator kwargs).
MODEL_ARCHS = {
    "prefinals": ({"upsample": "nearest", "skips": True, "track_running_stats": False}, {"spectral_norm": True}),
    "legacy": ({"upsample": "deconv", "skips": False, "track_running_stats": True}, {"spectral_norm": False}),
}


def arch_kwargs(arch="prefinals"):
    """(generator_kwargs, discriminator_kwargs) for a MODEL_ARCHS name."""
    if arch not in MODEL_ARCHS:
        raise ValueError(f"Unknown model arch {arch!r}; choose from {list(MODEL_ARCHS)}")
    g, d = MODEL_ARCHS[arch]
    return dict(g), dict(d)


def add_arch_arg(parser):
    parser.add_argument("--model-arch", choices=list(MODEL_ARCHS), default="prefinals",
                        help="prefinals (default): spectral-norm D, upsample+conv G with skips, no IN "
                             "running stats; legacy: the earlier architecture (loads checkpoints made "
                             "before these changes)")


RUN_RESULT_FILES = ("poc_results.json", "nonlong_pretrain_results.json")


def check_run_config(ckpt_path, image_size=None, model_arch=None):
    """Fail loudly if the training run that saved ckpt_path (its poc_results.json /
    nonlong_pretrain_results.json in the same folder, when present) used a different image_size
    or --model-arch than the caller -- every Module 2 script must use the same image_size (item
    4.3; the Discriminator's class-head kernel and the Generator's learned scale depend on it).
    Runs from before these fields were recorded are 128 px / legacy."""
    import json
    import os

    for name in RUN_RESULT_FILES:
        path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), name)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            run = json.load(f)
        saved = {"image_size": run.get("image_size", 128), "model_arch": run.get("model_arch", "legacy")}
        wanted = {"image_size": image_size, "model_arch": model_arch}
        bad = {k: (saved[k], v) for k, v in wanted.items() if v is not None and saved[k] != v}
        if bad:
            raise ValueError(f"{ckpt_path} was trained with " + ", ".join(f"{k}={a}" for k, (a, _) in bad.items())
                             + " but this run uses " + ", ".join(f"{k}={b}" for k, (_, b) in bad.items())
                             + f" ({path}). Pass the same --image-size / --model-arch as that run.")
        return saved
    return None


def load_checkpoint(model, path, map_location=None, arch=None):
    """load_state_dict(torch.load(path)) that fails loudly, with the likely fix, when the checkpoint
    was saved by the other architecture (see the module docstring)."""
    state = torch.load(path, map_location=map_location)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        legacy_keys = (any(k.startswith("up.") or k.endswith("running_mean") for k in state)  # G
                       or "main.0.weight" in state)  # D without spectral norm
        hint = ("it looks like a checkpoint from BEFORE the AllModuleEdits-PreFinals architecture changes "
                "-- pass --model-arch legacy to load it, or retrain" if legacy_keys and arch != "legacy" else
                "check --model-arch and the conv dims / repeat nums / image size match the run that saved it")
        raise RuntimeError(f"{type(model).__name__} checkpoint {path} does not match this architecture "
                           f"(--model-arch {arch}): {hint}.\n{e}") from e
    return model


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, track_running_stats=False):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=track_running_stats))

    def forward(self, x):
        return x + self.main(x)


class AdaIN(nn.Module):
    """
    AdaIN(x, s) = gamma(s) * (x - mu(x)) / sigma(x) + beta(s). gamma/beta are produced from
    the stage-conditioning vector s by a small linear projection, following the standard
    AdaIN formulation (Huang & Belongie 2017, adopted by StarGAN v2).
    """
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResidualBlock(nn.Module):
    """Residual block using AdaIN instead of plain InstanceNorm2d for the style/stage
    injection, replacing ResidualBlock in the Generator's bottleneck."""
    def __init__(self, dim_in, dim_out, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain1 = AdaIN(style_dim, dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain2 = AdaIN(style_dim, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, s):
        h = self.relu(self.adain1(self.conv1(x), s))
        h = self.adain2(self.conv2(h), s)
        return x + h


class Generator(nn.Module):
    """
    Generator network. Bottleneck residual blocks use AdaIN conditioning on the target-stage
    vector (see AdaINResidualBlock above), additive to the spatial channel-concat conditioning
    on the down/up-sampling layers -- both mechanisms condition on the same target stage.

    Checkpoint compatibility: state_dict keys are self.down / self.bottleneck / self.up (not
    a single self.main), and each bottleneck block includes AdaIN's fc layers. The up-sampling path
    is self.up (one Sequential) for upsample='deconv' without skips (the legacy layout), else
    self.up_blocks + self.to_rgb. See the module docstring for upsample / skips /
    track_running_stats.
    """
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, style_dim=None, upsample="nearest", skips=True,
                 track_running_stats=False):
        super(Generator, self).__init__()
        if upsample not in ("nearest", "bilinear", "deconv"):
            raise ValueError(f"upsample must be 'nearest', 'bilinear' or 'deconv', not {upsample!r}")
        style_dim = style_dim or c_dim
        self.c_dim = c_dim
        self.style_dim = style_dim
        self.skips = skips

        def norm(dim):
            return nn.InstanceNorm2d(dim, affine=True, track_running_stats=track_running_stats)

        down_layers = []
        down_layers.append(nn.Conv2d(4+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)) # changed from 3+c_dim
        down_layers.append(norm(conv_dim))
        down_layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            down_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            down_layers.append(norm(curr_dim*2))
            down_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        self.down = nn.Sequential(*down_layers)
        # Outputs of the stem and of the first down-sampling stage (indices into self.down, after
        # their ReLU) -- the skip connections' sources; the second stage feeds the bottleneck.
        self._skip_taps = (2, 5)

        # Bottleneck layers -- AdaINResidualBlock needs the style vector `s` passed explicitly
        # on every forward call, so this is a ModuleList (per-block calls), not a Sequential.
        self.bottleneck = nn.ModuleList([
            AdaINResidualBlock(curr_dim, curr_dim, style_dim) for _ in range(repeat_num)
        ])

        # Up-sampling layers.
        def up_conv(dim_in, dim_out):
            if upsample == "deconv":
                return [nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False)]
            return [nn.Upsample(scale_factor=2, mode=upsample),
                    nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)]

        if upsample == "deconv" and not skips:  # legacy layout -- same state_dict keys as before
            up_layers = []
            for i in range(2):
                up_layers += up_conv(curr_dim, curr_dim // 2)
                up_layers += [norm(curr_dim // 2), nn.ReLU(inplace=True)]
                curr_dim = curr_dim // 2
            up_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
            up_layers.append(nn.Tanh())
            self.up = nn.Sequential(*up_layers)
            self.up_blocks = None
        else:
            # With skips, each block's output is concatenated with the matching down-sampling
            # stage's features (same channels and resolution) before the next block / to_rgb.
            blocks, in_dim = [], curr_dim
            for out_dim in (curr_dim // 2, curr_dim // 4):  # = the down-sampling stages' widths
                blocks.append(nn.Sequential(*up_conv(in_dim, out_dim), norm(out_dim), nn.ReLU(inplace=True)))
                in_dim = out_dim * 2 if skips else out_dim
            curr_dim = in_dim
            self.up_blocks = nn.ModuleList(blocks)
            self.to_rgb = nn.Sequential(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
                                        nn.Tanh())

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information (unchanged from the original
        # channel-concat conditioning -- AdaIN below is additive to this, not a replacement).
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c_spatial = c.view(c.size(0), c.size(1), 1, 1)
        c_spatial = c_spatial.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c_spatial], dim=1)
        feats = []
        for i, layer in enumerate(self.down):
            x = layer(x)
            if i in self._skip_taps:
                feats.append(x)
        for block in self.bottleneck:
            x = block(x, c)  # AdaIN style vector = the same target-stage one-hot vector
        if self.up_blocks is None:
            return self.up(x)
        for block in self.up_blocks:
            x = block(x)
            if self.skips:
                x = torch.cat([x, feats.pop()], dim=1)  # deepest skip first
        return self.to_rgb(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN. spectral_norm (default) wraps the down-sampling convs and
    the real/fake output conv in spectral normalization (item 3.4); the stage class head (conv2) is
    left plain, so it can be re-initialized on its own (train_module2_poc.py
    --init-discriminator-mode reset-cls-head). Its kernel depends on image_size, so D checkpoints
    only load at the image size they were trained at."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, spectral_norm=True):
        super(Discriminator, self).__init__()
        sn = _spectral_norm if spectral_norm else (lambda m: m)
        layers = []
        layers.append(sn(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(sn(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        if kernel_size < 1 or image_size % np.power(2, repeat_num):
            raise ValueError(f"image_size {image_size} must be a multiple of 2**repeat_num ({2 ** repeat_num})")
        self.main = nn.Sequential(*layers)
        self.conv1 = sn(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
