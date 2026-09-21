"""
Module 3 -- EfficientNet-B4 backbone + Weibull survival head.

Independently trained from Module 1's classifier (a resnet50 + linear head) -- no shared
weights, per docs/DECISIONS.md's module-independence rule; this is a different architecture
entirely, not just a separate instance of the same one.

Takes the LBS stratum (see module1/compute_lbs.py, reused here via
module3/dataset.py::stratify_lbs_by_grade) AND the baseline eye's own severity grade (ICDR
0-4) as auxiliary inputs alongside the fundus image, since LBS is meant to do double duty as
both Module 2's fallback-synthesis target and Module 3's own survival-analysis input feature
(docs/DECISIONS.md). The baseline-grade embedding (added per
docs/IMPLEMENTATION_PLAN.md Task H) is what lets the model distinguish "mild->moderate" risk
from "severe->PDR" risk -- without it, the only severity signal available is whatever the
image implicitly encodes.

Outputs Weibull distribution parameters (shape k, scale lambda) per patient rather than a
point estimate -- train_module3_poc.py's module docstring explains why this only calibrates
reliably at Tianjin's single observed 2-year checkpoint for now.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

NUM_LBS_STRATA = 3  # low/medium/high -- see module1/compute_lbs.py::stratify_lbs
NUM_STAGES = 5  # ICDR 0-4, matching baseline_grade_icdr in module3/dataset.py


class EfficientNetWeibullSurvival(nn.Module):
    def __init__(self, pretrained=True, lbs_embedding_dim=8, grade_embedding_dim=8,
                 hidden_dim=128, dropout=0.3):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)
        backbone_out_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.lbs_embedding = nn.Embedding(NUM_LBS_STRATA, lbs_embedding_dim)
        self.grade_embedding = nn.Embedding(NUM_STAGES, grade_embedding_dim)

        self.head = nn.Sequential(
            nn.Linear(backbone_out_features + lbs_embedding_dim + grade_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [pre_shape, pre_scale] -- squashed positive below
        )

    def forward(self, image, lbs_stratum_idx, baseline_grade_icdr):
        features = self.backbone(image)
        lbs_emb = self.lbs_embedding(lbs_stratum_idx)
        grade_emb = self.grade_embedding(baseline_grade_icdr)
        combined = torch.cat([features, lbs_emb, grade_emb], dim=1)
        out = self.head(combined)
        # softplus keeps shape/scale strictly positive with better-behaved gradients near 0
        # than exp(); +eps avoids a degenerate zero-scale/zero-shape Weibull.
        shape = F.softplus(out[:, 0]) + 1e-3
        scale = F.softplus(out[:, 1]) + 1e-3
        return shape, scale

    @staticmethod
    def survival_function(shape, scale, t):
        """S(t) = exp(-(t/scale)^shape) -- probability of NOT having progressed by time t."""
        return torch.exp(-torch.pow(t / scale, shape))


def weibull_current_status_nll(shape, scale, t, event, eps=1e-6):
    """
    Negative log-likelihood for CURRENT-STATUS (case-1 interval-censored) survival data: each
    subject is observed exactly once, at time t, and we only know whether the event had
    already happened by then (event=1, likelihood = F(t) = 1-S(t)) or not (event=0,
    likelihood = S(t)) -- NOT the exact event time. This is what Tianjin's fixed
    baseline -> 2-year-follow-up structure actually gives us (see module3/dataset.py and
    docs/ROADMAP.md's "Consequence of the Tianjin timing structure" note). Using the Weibull
    density f(t) here, as ordinary right-censored survival NLL does for its observed events,
    would incorrectly assume we know the exact progression date, which we don't.
    """
    s_t = torch.exp(-torch.pow(t / scale, shape))
    cdf = torch.clamp(1 - s_t, min=eps, max=1 - eps)
    s_t = torch.clamp(s_t, min=eps, max=1 - eps)
    nll = -(event * torch.log(cdf) + (1 - event) * torch.log(s_t))
    return nll.mean()


if __name__ == "__main__":
    print("Smoke-testing EfficientNetWeibullSurvival (random weights, no download)...")
    model = EfficientNetWeibullSurvival(pretrained=False)
    image = torch.randn(2, 3, 380, 380)
    lbs_idx = torch.tensor([0, 2])
    baseline_grade = torch.tensor([1, 3])
    shape, scale = model(image, lbs_idx, baseline_grade)
    print(f"shape: {shape.tolist()}, scale: {scale.tolist()}")

    t = torch.tensor([2.0, 2.0])
    event = torch.tensor([1.0, 0.0])
    loss = weibull_current_status_nll(shape, scale, t, event)
    print(f"NLL: {loss.item():.4f}")
    s_2yr = model.survival_function(shape, scale, t)
    print(f"S(2yr): {s_2yr.tolist()}")
    print("\n✓ Module 3 model working!")
