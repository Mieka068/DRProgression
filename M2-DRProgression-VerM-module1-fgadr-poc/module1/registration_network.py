"""
Training-only affine registration network: predicts a 2x3 affine matrix that warps a
follow-up image onto its baseline's coordinate frame, trained self-supervised via photometric
reconstruction loss (no manual correspondence labels required).

Used only during Module 2's training-data preparation, to spatially align baseline<->follow-up
pairs before they supervise the GAN -- not used at inference time, since inference only needs
a single baseline image with no follow-up to align against.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineRegistrationNet(nn.Module):
    def __init__(self, in_channels=6):  # baseline (3) + follow-up (3), stacked
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 6)  # 2x3 affine matrix, flattened
        # Initialize to identity transform so training starts from "no warp"
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, baseline, followup):
        x = torch.cat([baseline, followup], dim=1)
        features = self.localization(x).flatten(1)
        theta = self.fc(features).view(-1, 2, 3)
        grid = F.affine_grid(theta, followup.size(), align_corners=False)
        warped_followup = F.grid_sample(followup, grid, align_corners=False)
        return warped_followup, theta


def registration_photometric_loss(warped_followup, baseline):
    """L1 photometric loss between the warped follow-up and the baseline -- minimizing this
    is what teaches the network to align structures (vessels, optic disc) rather than
    intensity alone, since L1 on raw pixels still pulls toward matching retinal geometry
    given these images share the same underlying anatomy at two time points."""
    return F.l1_loss(warped_followup, baseline)


if __name__ == "__main__":
    print("Smoke-testing AffineRegistrationNet (random weights)...")
    net = AffineRegistrationNet()
    baseline = torch.randn(2, 3, 128, 128)
    followup = torch.randn(2, 3, 128, 128)

    warped, theta = net(baseline, followup)
    print(f"warped shape: {warped.shape}, theta shape: {theta.shape}")
    print(f"theta (should start near identity [1,0,0,0,1,0]):\n{theta[0]}")

    loss = registration_photometric_loss(warped, baseline)
    loss.backward()
    print(f"L1 loss: {loss.item():.4f}")
    print("\n✓ AffineRegistrationNet working!")
