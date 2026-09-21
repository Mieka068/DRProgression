import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class AdaIN(nn.Module):
    """
    AdaIN(x, s) = gamma(s) * (x - mu(x)) / sigma(x) + beta(s), per the manuscript's Sec 3.2
    formula. gamma/beta are produced from the stage-conditioning vector s by a small linear
    projection, following the standard AdaIN formulation (Huang & Belongie 2017, adopted by
    StarGAN v2). Confirmed absent from the official DRForecastGAN release (see
    docs/IMPLEMENTATION_PLAN.md Task C) -- this is new engineering on top of that base, not
    something recovered from upstream code.
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
    vector (see AdaINResidualBlock above), ADDITIVE to the existing spatial channel-concat
    conditioning on the down/up-sampling layers -- both mechanisms reinforce the same target
    stage, rather than AdaIN replacing the original conditioning path wholesale. See
    docs/IMPLEMENTATION_PLAN.md Task C for why this reading of the manuscript's formula was
    chosen.

    CHECKPOINT COMPATIBILITY: this changes the state_dict keys (self.main -> self.down /
    self.bottleneck / self.up, and each bottleneck block gained AdaIN's fc layers). Any
    checkpoint trained under the old architecture will NOT load into this one -- retrain from
    scratch, don't attempt a partial load.
    """
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, style_dim=None):
        super(Generator, self).__init__()
        style_dim = style_dim or c_dim
        self.c_dim = c_dim
        self.style_dim = style_dim

        down_layers = []
        down_layers.append(nn.Conv2d(4+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)) # changed from 3+c_dim
        down_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        down_layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            down_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            down_layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            down_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        self.down = nn.Sequential(*down_layers)

        # Bottleneck layers -- AdaINResidualBlock needs the style vector `s` passed explicitly
        # on every forward call, so this is a ModuleList (per-block calls), not a Sequential.
        self.bottleneck = nn.ModuleList([
            AdaINResidualBlock(curr_dim, curr_dim, style_dim) for _ in range(repeat_num)
        ])

        # Up-sampling layers.
        up_layers = []
        for i in range(2):
            up_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            up_layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            up_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        up_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        up_layers.append(nn.Tanh())
        self.up = nn.Sequential(*up_layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information (unchanged from the original
        # channel-concat conditioning -- AdaIN below is additive to this, not a replacement).
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c_spatial = c.view(c.size(0), c.size(1), 1, 1)
        c_spatial = c_spatial.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c_spatial], dim=1)
        x = self.down(x)
        for block in self.bottleneck:
            x = block(x, c)  # AdaIN style vector = the same target-stage one-hot vector
        x = self.up(x)
        return x


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
