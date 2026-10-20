"""
Methodology + per-module model diagrams for the write-up (matplotlib only, no data needed).

    python docs/diagrams/make_diagrams.py            # writes the PNGs (and PDFs) next to this file

Figures:
    methodology_overview      whole pipeline (Module 1 -> Modules 2 + 3 -> report)
    module1_model             DRG-Net grading + lesion segmentation + LBS
    module2_pretrain_step     stage-1 EyePACS pretraining step (train_module2_nonlong_pretrain.py)
    module2_finetune_step     stage-2 longitudinal fine-tuning step (train_module2_poc.py)
    module2_inference         inference cascade (synthesize_trajectory())
    module3_model             EfficientNet-B4 + Weibull survival head (module3/)
The fundus / mask thumbnails are drawn, not real images.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))

# overview palette (same as the earlier methodology figure)
M1_PANEL, M1_BOX, M1_EDGE = "#eef3fb", "#c9d9f2", "#5b7bb4"
M2_PANEL, M2_BOX, M2_EDGE = "#eef7ea", "#cfe8c6", "#5f9a55"
M3_PANEL, M3_BOX, M3_EDGE = "#fcf2e6", "#f3d4ab", "#c48a45"
OUT_BOX, OUT_EDGE = "#fdf3c4", "#c9a227"
RPT_PANEL = "#fffbe6"
IN_BOX, IN_EDGE = "#ece9f7", "#6b63a8"
GREY = "#555555"
# model-diagram palette (same as the earlier Module 2 figures)
G_FILL, G_EDGE = "#c8f5c8", "#3f8f5a"
D_FILL, D_EDGE = "#e3e3f5", "#4a4a9a"
R_FILL, R_EDGE = "#fae3c8", "#b06a2a"
B_EDGE = "#3a78d8"
LOSS_EDGE = "#3d3d9a"


# ---------------------------------------------------------------- drawing helpers
def new_fig(w_units, h_units, scale=0.11):
    fig = plt.figure(figsize=(w_units * scale, h_units * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_units)
    ax.set_ylim(0, h_units)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title=None, sub=None, fc="white", ec="black", ls="-", lw=1.1, r=0.8,
        tsize=9, ssize=7.6, family=None, align="center", z=2, gap=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, ls=ls, lw=lw, zorder=z))
    kw = dict(zorder=z + 1, family=family)
    tx = x + w / 2 if align == "center" else x + 1.2
    if title and sub:
        n = sub.count("\n") + 1
        g = gap if gap is not None else 1.25
        ax.text(tx, y + h / 2 + n * 0.62 * g * ssize / 7.6, title, ha=align, va="center",
                fontsize=tsize, fontweight="bold", **kw)
        ax.text(tx, y + h / 2 - 0.62 * g * ssize / 7.6, sub, ha=align, va="center",
                fontsize=ssize, linespacing=1.3, **kw)
    elif title:
        ax.text(tx, y + h / 2, title, ha=align, va="center", fontsize=tsize, fontweight="bold", **kw)
    elif sub:
        ax.text(tx, y + h / 2, sub, ha=align, va="center", fontsize=ssize, linespacing=1.3, **kw)


def panel(ax, x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=fc, ec=ec, lw=0.9, zorder=0))


def arrow(ax, p, q, color=GREY, lw=1.2, rad=0.0, z=4, ms=10):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def path(ax, pts, color=GREY, lw=1.2, z=4, ms=10):
    """Polyline through pts with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=color, lw=lw, zorder=z, solid_joinstyle="miter")
    arrow(ax, pts[-2], pts[-1], color, lw, z=z, ms=ms)


def label(ax, x, y, s, size=7.2, color="#333333", ha="left", va="center", style="italic", family=None, z=5,
          bg=None):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, style=style, zorder=z, family=family,
            bbox=dict(fc=bg, ec="none", pad=0.6) if bg else None)


def trapezoid(ax, x, y, w, h_left, h_right, fc, ec, text=None, size=9, family=None, bold=True):
    cy = y
    pts = [(x, cy - h_left / 2), (x + w, cy - h_right / 2), (x + w, cy + h_right / 2), (x, cy + h_left / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.3, zorder=2))
    if text:
        ax.text(x + w / 2, cy, text, ha="center", va="center", fontsize=size, zorder=3, family=family,
                fontweight="bold" if bold else "normal", linespacing=1.3)


def unet(ax, x, y, w, h, fc, ec, text=None, size=8, family=None):
    """Bow-tie: encoder narrowing, decoder widening."""
    half = w / 2
    trapezoid(ax, x, y, half, h, h * 0.35, fc, ec)
    trapezoid(ax, x + half, y, half, h * 0.35, h, fc, ec)
    if text:
        ax.text(x + half, y - h / 2 - 1.2, text, ha="center", va="top", fontsize=size, family=family,
                linespacing=1.3)


def fundus(ax, cx, cy, s, seed=0, synthetic=False, lesions=0):
    """Stylised colour fundus photo centred at (cx, cy), side s (data units)."""
    rng = np.random.default_rng(seed)
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s, boxstyle="square,pad=0", fc="black",
                                ec="none", zorder=3))
    r = s * 0.46
    for k, col in enumerate(["#7a1f0a", "#a8361a", "#c94f25", "#d9662f", "#e07a3a"]):
        ax.add_patch(Circle((cx, cy), r * (1 - k * 0.17), fc=col, ec="none", zorder=3.1 + k * 0.01))
    dx, dy = cx + r * 0.42, cy + r * 0.05
    ax.add_patch(Circle((dx, dy), r * 0.17, fc="#f7d67a", ec="none", zorder=3.3))
    for ang in np.linspace(0.6, 5.7, 7):
        t = np.linspace(0, 1, 20)
        bend = rng.uniform(-0.6, 0.6)
        px = dx + np.cos(ang + bend * t) * r * 0.95 * t
        py = dy + np.sin(ang + bend * t) * r * 0.95 * t
        ok = (px - cx) ** 2 + (py - cy) ** 2 < (r * 0.97) ** 2
        ax.plot(px[ok], py[ok], color="#5a0f05", lw=0.7, zorder=3.35)
    for _ in range(lesions):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.75) * r
        col = "#f4e27a" if rng.random() < 0.5 else "#4a0a02"
        ax.add_patch(Circle((cx + d * np.cos(a), cy + d * np.sin(a)), r * rng.uniform(0.03, 0.06), fc=col,
                            ec="none", zorder=3.4))
    if synthetic:
        ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s, boxstyle="square,pad=0", fc="white",
                                    alpha=0.08, ec="none", zorder=3.5))


def mask_thumb(ax, cx, cy, s, seed=1, n=9, color="white"):
    rng = np.random.default_rng(seed)
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s, boxstyle="square,pad=0", fc="black",
                                ec="none", zorder=3))
    for _ in range(n):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.05, 0.4) * s
        ax.add_patch(Circle((cx + d * np.cos(a), cy + d * np.sin(a)), s * rng.uniform(0.02, 0.045), fc=color,
                            ec="none", zorder=3.1))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".png"))


def title(ax, x, y, s, size=15):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, fontweight="bold", family="serif")


# ---------------------------------------------------------------- 1. methodology overview
def overview():
    fig, ax = new_fig(100, 124)
    # input
    box(ax, 29, 116.5, 42, 6.5, "Pipeline input: single baseline fundus image",
        "45° color fundus photograph, ≥ mild NPDR, no clinical metadata", IN_BOX, IN_EDGE, tsize=9.5)
    arrow(ax, (50, 116.5), (50, 113.2))

    # ---- Module 1
    panel(ax, 3, 82.5, 94, 30.5, M1_PANEL, "#a9bde0")
    box(ax, 22, 107.5, 56, 4.3, "Module 1: Severity Staging & Lesion Segmentation (DRG-Net)", None,
        M1_BOX, M1_EDGE, tsize=9.5)
    comps = [(6, "Grading", "ResNet-50 classifier, trained on FGADR\nICDR grade 0–4 + softmax probabilities"),
             (36, "Segmentation", "4 U-Nets (ResNet-50 encoder), one per lesion\nMA, HE, EX, SE · trained on FGADR + DDR"),
             (68, "Lesion Burden Score (LBS)", "lesion px / retinal FOV px")]
    widths = [28, 28, 26]
    ax.plot([20, 50], [104.8, 104.8], color=GREY, lw=1.2, zorder=4)
    ax.plot([50, 50], [107.5, 104.8], color=GREY, lw=1.2, zorder=4)
    for (x, t, s), w in zip(comps, widths):
        box(ax, x, 96.5, w, 6.6, t, s, M1_BOX, M1_EDGE, tsize=8.8, ssize=7.1)
        if t != "Lesion Burden Score (LBS)":
            arrow(ax, (x + w / 2, 104.8), (x + w / 2, 103.1))
    arrow(ax, (64, 99.8), (68, 99.8))
    outs = [(6, "Grade label", "baseline (start) stage"),
            (36, "Lesion mask", "union of the 4 lesion masks (1 channel)"),
            (68, "LBS + burden stratum", "low / medium / high, per-grade tertiles")]
    for (x, t, s), w in zip(outs, widths):
        box(ax, x, 85.5, w, 5.8, t, s, OUT_BOX, OUT_EDGE, tsize=8.8, ssize=7.1)
        arrow(ax, (x + w / 2, 96.5), (x + w / 2, 91.3))

    # ---- Module 1 -> 2 / 3 connectors
    arrow(ax, (20, 85.5), (20, 76.6))
    label(ax, 21, 80.0, "start stage")
    arrow(ax, (50, 85.5), (50, 76.6))
    label(ax, 51, 80.0, "baseline mask")
    # grade -> Module 3, hopping over the mask line
    ax.plot([20, 48.8], [81.8, 81.8], color=GREY, lw=1.2, zorder=4)
    t = np.linspace(np.pi, 0, 30)
    ax.plot(50 + 1.2 * np.cos(t), 81.8 + 1.2 * np.sin(t), color=GREY, lw=1.2, zorder=4)
    ax.plot([51.2, 74], [81.8, 81.8], color=GREY, lw=1.2, zorder=4)
    arrow(ax, (74, 81.8), (74, 76.6))
    label(ax, 75, 80.0, "grade")
    arrow(ax, (82, 85.5), (82, 76.6))
    label(ax, 83, 80.0, "LBS stratum")

    # ---- Module 2
    panel(ax, 3, 21.5, 57, 57.5, M2_PANEL, "#a8cf9c")
    box(ax, 6, 71.0, 51, 5.6, "Module 2: Autoregressive Multi-Stage Trajectory Synthesis",
        "StarGAN-style generator adapted from DRForecastGAN · inference cascade", M2_BOX, M2_EDGE,
        tsize=9, ssize=7.1)
    arrow(ax, (17.5, 71.0), (17.5, 69.8))
    box(ax, 6, 57.5, 23, 12.3, "Generator G(x, c)",
        "input: image + lesion mask (4 ch)\n+ target stage c (one-hot)\nAdaIN target-stage conditioning\n"
        "weights: best-G.ckpt", M2_BOX, M2_EDGE, tsize=8.8, ssize=7.1)
    box(ax, 34, 57.5, 23, 12.3, "Module 1: Segmentation",
        "4 U-Nets re-segment the\nsynthesized image\n→ next-step lesion mask", M2_BOX, M2_EDGE,
        tsize=8.8, ssize=7.1)
    box(ax, 34, 44.5, 23, 10.5, "Module 1: Grading",
        "ResNet-50 consistency check\nagainst the target stage\n(logged, does not alter output)", M2_BOX,
        M2_EDGE, tsize=8.8, ssize=7.1)
    box(ax, 6, 44.5, 23, 10.5, "Stage k+1 image + mask",
        "fed back as next input\nrepeat until PDR (stage 4)", M2_BOX, M2_EDGE, tsize=8.8, ssize=7.1)
    arrow(ax, (29, 63.6), (34, 63.6))
    arrow(ax, (45.5, 57.5), (45.5, 55.0))
    arrow(ax, (34, 49.7), (29, 49.7))
    arrow(ax, (17.5, 55.0), (17.5, 57.5))
    box(ax, 6, 30.5, 51, 12.3, None, None, "white", "#666666", ls=(0, (4, 3)), lw=1.0)
    ax.text(31.5, 41.3, "Training only (not used at inference)", ha="center", va="center", fontsize=8.6,
            fontweight="bold", zorder=4)
    ax.text(7.5, 36.0,
            "Stage 1 pretraining · EyePACS single graded images (40 ep):\n"
            "   stage-classification, cycle + identity L1, Module 1 biomarker-profile\n"
            "   and monotonic-burden losses; WGAN-GP PatchGAN discriminator\n"
            "Stage 2 fine-tuning · FIRE + LongDR + Tianjin pairs (50 ep):\n"
            "   affine registration aligns follow-up to baseline → L1 + WGAN-GP;\n"
            "   best epoch = lowest L1 on 10 % held-out eyes / patients",
            ha="left", va="center", fontsize=6.9, linespacing=1.3, zorder=4)
    box(ax, 6, 23.0, 24, 5.8, "Synthesized fundus images", "one per stage, up to PDR", OUT_BOX, OUT_EDGE,
        tsize=8.6, ssize=7.1)
    box(ax, 33, 23.0, 24, 5.8, "Stage label per step", "+ Module 1 consistency score", OUT_BOX, OUT_EDGE,
        tsize=8.6, ssize=7.1)

    # ---- Module 3
    panel(ax, 62, 21.5, 35, 57.5, M3_PANEL, "#e3bf93")
    box(ax, 65, 71.0, 29, 5.6, "Module 3: Probabilistic Time-to-Progression", "runs in parallel with Module 2",
        M3_BOX, M3_EDGE, tsize=8.3, ssize=7.1)
    box(ax, 65, 62.0, 29, 7.8, "EfficientNet-B4 backbone",
        "input: real baseline image only (380 px)\nImageNet init, independent weights\n(not shared with Module 1)",
        M3_BOX, M3_EDGE, tsize=8.6, ssize=7.0)
    box(ax, 65, 54.5, 29, 5.6, "Auxiliary embeddings", "LBS stratum (3) + baseline grade (5)", M3_BOX, M3_EDGE,
        tsize=8.6, ssize=7.1)
    box(ax, 65, 46.4, 29, 6.2, "Weibull survival head", "shape k, scale λ:  S(t) = exp(−(t/λ)^k)", M3_BOX,
        M3_EDGE, tsize=8.6, ssize=7.1)
    for a, b in ((71.0, 69.8), (62.0, 60.1), (54.5, 52.6)):
        arrow(ax, (79.5, a), (79.5, b))
    box(ax, 65, 32.5, 29, 11.5, None, None, "white", "#666666", ls=(0, (4, 3)), lw=1.0)
    ax.text(79.5, 41.8, "Training only", ha="center", va="center", fontsize=8.6, fontweight="bold", zorder=4)
    ax.text(79.5, 36.6, "Tianjin, fixed 2-year follow-up (30 ep)\ncurrent-status (interval-censored)\n"
            "Weibull NLL; best epoch = lowest\nval NLL (best-model.ckpt)", ha="center", va="center",
            fontsize=6.9, linespacing=1.3, zorder=4)
    box(ax, 65, 23.0, 29, 6.8, "P(progression within 2 years)", "+ survival curve S(t)", OUT_BOX, OUT_EDGE,
        tsize=8.6, ssize=7.1)
    arrow(ax, (79.5, 46.4), (79.5, 44.0))
    arrow(ax, (79.5, 32.5), (79.5, 29.8))

    # ---- side lines: real baseline image to Modules 2 and 3
    path(ax, [(29, 119.7), (1.5, 119.7), (1.5, 63.6), (6, 63.6)])
    label(ax, 3, 121.0, "real baseline image to Module 2", size=6.9)
    path(ax, [(71, 119.7), (98.5, 119.7), (98.5, 66.0), (94, 66.0)])
    label(ax, 97, 121.0, "real baseline image to Module 3", size=6.9, ha="right")

    # ---- report
    panel(ax, 3, 1.5, 94, 17.0, RPT_PANEL, "#e6d48a")
    box(ax, 33, 12.2, 34, 4.4, "Complete Forecast Report", None, OUT_BOX, OUT_EDGE, tsize=9.5)
    path(ax, [(31.5, 23.0), (31.5, 18.5), (40, 18.5), (40, 16.6)])
    path(ax, [(79.5, 23.0), (79.5, 18.5), (60, 18.5), (60, 16.6)])
    box(ax, 6, 3.0, 42, 6.3, "Synthesized future-stage fundus images up to PDR",
        "each captioned with its Module 1 grade check", OUT_BOX, OUT_EDGE, tsize=8.6, ssize=7.1)
    box(ax, 52, 3.0, 42, 6.3, "Est. P(further progression within 2 years) per stage",
        "estimates on synthesized images flagged as untested", OUT_BOX, OUT_EDGE, tsize=8.6, ssize=7.1)
    path(ax, [(50, 12.2), (50, 10.8), (27, 10.8), (27, 9.3)])
    path(ax, [(50, 10.8), (73, 10.8), (73, 9.3)])
    save(fig, "methodology_overview")


# ---------------------------------------------------------------- 2. Module 1 model
def module1():
    fig, ax = new_fig(120, 62)
    title(ax, 60, 58.5, "Module 1 - severity grading & lesion segmentation (DRG-Net)")
    F = "serif"
    # input
    box(ax, 2, 24, 17, 16, None, None, "white", "black", ls=(0, (2, 2)), lw=1.3, r=1.2)
    fundus(ax, 7, 32, 7, seed=3, lesions=8)
    ax.text(11.2, 32, "X\nbaseline\nfundus\nimage (RGB)", fontsize=8, va="center", family=F, linespacing=1.25)
    ax.text(11.2, 36.6, "X", fontsize=9, fontweight="bold", family=F, va="center", alpha=0)

    # grading branch
    path(ax, [(19, 36), (24, 36), (24, 47), (30, 47)], color="black")
    label(ax, 20, 49.5, "512 × 512, FGADR\nnormalisation", size=7, family=F, style="normal")
    trapezoid(ax, 30, 47, 13, 14, 6, G_FILL, G_EDGE, "ResNet-50", size=9, family=F)
    arrow(ax, (43, 47), (49, 47), color="black")
    box(ax, 49, 43.5, 13, 7, None, "FC 2048 → 5\nsoftmax", "white", G_EDGE, ssize=8, family=F, r=1)
    arrow(ax, (62, 47), (68, 47), color=G_EDGE)
    box(ax, 68, 42.5, 21, 9, "ĝ  grade", "ICDR 0–4 +\nclass probabilities", "white", G_EDGE, ls=(0, (3, 2)),
        tsize=8.5, ssize=7.6, family=F, r=1.2)

    # segmentation branch
    path(ax, [(19, 30), (27, 30)], color="black")
    for k in range(3, -1, -1):
        unet(ax, 27 + k * 0.8, 30 + k * 0.8, 15, 11, "#d6ecf8" if k else "#bfe0f5", "#2f6f9f")
    ax.text(35, 21.5, "U-Net × 4  (ResNet-50 encoder,\n2-class output), one per lesion", ha="center", va="top",
            fontsize=7.6, family=F, linespacing=1.3)
    lesions = ["MA", "HE", "EX", "SE"]
    for i, les in enumerate(lesions):
        yy = 39 - i * 6
        arrow(ax, (45.5, 30 + (1.5 - i) * 1.6), (50.5, yy), color="#2f6f9f", lw=1.0, ms=8)
        mask_thumb(ax, 53.5, yy, 5, seed=10 + i, n=3 + 2 * i)
        ax.text(56.6, yy, les, fontsize=7.5, va="center", family=F)
        arrow(ax, (59.5, yy), (64.2, 30), color="#2f6f9f", lw=1.0, ms=8)
    ax.add_patch(Circle((66, 30), 1.9, fc="white", ec="#2f6f9f", lw=1.3, zorder=4))
    ax.text(66, 30, "∪", ha="center", va="center", fontsize=11, zorder=5, family=F)
    arrow(ax, (67.9, 30), (72, 30), color="#2f6f9f")
    box(ax, 72, 22.5, 17, 15, None, None, "white", "#2f6f9f", ls=(0, (3, 2)), r=1.2)
    mask_thumb(ax, 76.5, 30, 6, seed=21, n=14)
    ax.text(80.3, 30, "M\nlesion\nmask\n(1 ch)", fontsize=7.8, va="center", family=F, linespacing=1.25)

    # FOV + LBS
    path(ax, [(10.5, 24), (10.5, 9), (30, 9)], color="black")
    box(ax, 30, 5.5, 18, 7, None, "FOV mask\n(green channel > 10)", "white", "#777777", ssize=7.8, family=F, r=1)
    arrow(ax, (48, 9), (60, 9), color="black")
    path(ax, [(80.5, 22.5), (80.5, 16), (66, 16), (66, 12.5)], color="#2f6f9f")
    box(ax, 60, 5.5, 12, 7, None, "LBS =\n|M| / |FOV|", "#fff6d6", OUT_EDGE, ssize=8, family=F, r=1)
    arrow(ax, (72, 9), (77, 9), color="black")
    box(ax, 77, 5.0, 18, 8, None, "burden stratum\nlow / med / high\n(per-grade tertiles)", "#fff6d6", OUT_EDGE,
        ssize=7.6, family=F, r=1)
    path(ax, [(89, 46), (93, 46), (93, 13)], color=G_EDGE)
    label(ax, 93.6, 35, "per-grade\nthresholds", size=6.8, family=F)

    # downstream + training
    box(ax, 97, 41.5, 21, 11, None, "to Module 2:\nstart stage ĝ, mask M\nto Module 3:\nĝ, LBS stratum",
        "white", "black", ssize=7.6, family=F, r=1.2)
    arrow(ax, (89, 47), (97, 47), color="black")
    box(ax, 97, 8, 21, 28.5, None, None, "white", LOSS_EDGE, lw=1.3, r=1.5)
    ax.text(107.5, 33.5, "Training (notebook 11)", ha="center", va="center", fontsize=8.2, fontweight="bold",
            family=F)
    ax.text(98.3, 20.5, "Grading: FGADR\n  weighted cross-entropy\n  SGD, lr 1e-4, 20 ep\n\n"
            "Segmentation:\n  FGADR + DDR combined\n  train 240 + 383 images\n  eval 81 + 149 images\n"
            "  pixel cross-entropy, 100 ep\n  best epoch = highest eval AP", fontsize=7.1, va="center",
            family=F, linespacing=1.3)
    save(fig, "module1_model")


# ---------------------------------------------------------------- 3. Module 2 stage-1 pretraining step
def module2_pretrain():
    fig, ax = new_fig(120, 72)
    F = "serif"
    title(ax, 60, 68.5, "Module 2 - stage 1 pretraining step (train_module2_nonlong_pretrain.py)")
    # inputs
    box(ax, 2, 41, 21, 18, None, None, "white", "black", ls=(0, (2, 2)), lw=1.3, r=1.2)
    fundus(ax, 6.5, 54, 6, seed=4, lesions=5)
    mask_thumb(ax, 6.5, 46, 6, seed=5, n=6)
    ax.text(10.5, 50, "X  EyePACS image\n(3ch, grade s)\n+ Module 1\nmask (1ch)", fontsize=7.6, va="center",
            family=F, linespacing=1.25)
    box(ax, 2, 27, 21, 9, None, "c  target stage, one-hot\ns+1 with p = 0.5,\nelse any stage", "white", "black",
        ssize=7.4, family=F, r=1.2)
    path(ax, [(23, 50), (35, 50)], color="black")
    label(ax, 23.5, 52, "concat, 4+c_dim ch", size=7, family=F, style="normal")
    path(ax, [(23, 31.5), (28, 31.5), (28, 46), (35, 46)], color="black")
    trapezoid(ax, 35, 48, 11, 9, 20, G_FILL, G_EDGE, "Generator", family=F)
    arrow(ax, (46, 48), (52, 48), color=G_EDGE)
    box(ax, 52, 41, 18, 14, None, None, "white", G_EDGE, ls=(0, (3, 2)), r=1.2)
    fundus(ax, 56.5, 48, 6, seed=4, lesions=12, synthetic=True)
    ax.text(60.3, 48, "G(x,c)\nstage-c\nimage", fontsize=7.6, va="center", family=F, linespacing=1.25)

    # discriminator with cls head
    path(ax, [(70, 51), (80, 51)], color=G_EDGE)
    trapezoid(ax, 80, 51, 13, 20, 9, D_FILL, D_EDGE, "Discriminator\n(PatchGAN\n+ grade head)", size=8, family=F)
    path(ax, [(12.5, 59), (12.5, 63), (86.5, 63), (86.5, 60.8)], color="black")
    label(ax, 30, 64.4, "real X, graded with its own stage s", size=7, family=F)
    # cycle
    path(ax, [(61, 41), (61, 30), (52, 30)], color=G_EDGE)
    box(ax, 33, 25, 19, 10, None, "G(G(x,c), s)\ncycle back to the\nsource stage", "white", G_EDGE,
        ls=(0, (3, 2)), ssize=7.4, family=F, r=1.2)
    path(ax, [(42.5, 25), (42.5, 22.6)], color=G_EDGE)
    ax.add_patch(Circle((42.5, 20.3), 2.3, fc="white", ec="black", lw=1.1, zorder=4))
    ax.text(42.5, 20.3, "L1\ncyc", ha="center", va="center", fontsize=6.8, zorder=5, family=F)
    path(ax, [(2, 47), (0.8, 47), (0.8, 20.3), (40.2, 20.3)], color="black")
    label(ax, 3, 18.6, "X", size=7, family=F)

    # biomarker regressor
    path(ax, [(66, 41), (66, 33), (72, 33)], color=G_EDGE)
    trapezoid(ax, 72, 33, 11, 9, 5, R_FILL, R_EDGE, "Biomarker\nregressor R", size=7, family=F)
    arrow(ax, (83, 33), (87, 33), color=R_EDGE)
    box(ax, 87, 26, 14, 14, None, "R(G(x,c)):\nEX / HE / MA / SE\narea + count,\nLBS (9 values)", "white",
        R_EDGE, ls=(0, (3, 2)), ssize=7, family=F, r=1.2)
    box(ax, 66, 8, 21, 11, None, "stage profiles from\nModule 1 on EyePACS\n(stage_biomarker_\nprofiles.json)",
        "#fff6d6", OUT_EDGE, ssize=7, family=F, r=1.2)
    path(ax, [(87, 13.5), (106, 13.5), (106, 17)], color=OUT_EDGE)
    ax.add_patch(Circle((106, 20), 3.0, fc="white", ec="black", lw=1.1, zorder=4))
    ax.text(106, 20, "bio +\nmono", ha="center", va="center", fontsize=6.8, zorder=5, family=F)
    path(ax, [(94, 26), (94, 20), (103, 20)], color=R_EDGE)
    label(ax, 66, 4.5, "R is trained on real images to predict Module 1's biomarkers; mono: lesion burden of "
          "G(x,c) ≥ burden of X when c > s", size=6.8, family=F)

    # losses
    arrow(ax, (93, 51), (99, 51), color=D_EDGE)
    box(ax, 99, 42, 20, 22, None, None, "white", LOSS_EDGE, lw=1.3, r=1.5)
    ax.text(100.3, 53, "d_loss = fake − real\n  + 10·GP + CE(D_cls(X), s)\n\n"
            "g_loss = adv\n  + CE(D_cls(G(x,c)), c)\n  + 10·L1_cyc + 5·L1_id\n  + 1·L_bio + 1·L_mono\n\n"
            "40 epochs · 20,000\nimages per epoch", fontsize=6.9, va="center", family=F, linespacing=1.3)
    save(fig, "module2_pretrain_step")


# ---------------------------------------------------------------- 4. Module 2 stage-2 fine-tuning step
def module2_finetune():
    fig, ax = new_fig(120, 62)
    F = "serif"
    title(ax, 60, 58.5, "Module 2 - stage 2 fine-tuning step (train_module2_poc.py)")
    box(ax, 2, 33, 21, 18, None, None, "white", "black", ls=(0, (2, 2)), lw=1.3, r=1.2)
    fundus(ax, 6.5, 46, 6, seed=6, lesions=4)
    mask_thumb(ax, 6.5, 38, 6, seed=7, n=5)
    ax.text(10.5, 42, "X\nbaseline image\n(3ch) + Module 1\nmask (1ch)", fontsize=7.6, va="center", family=F,
            linespacing=1.25)
    path(ax, [(23, 44), (34, 44)], color="black")
    label(ax, 23.5, 46, "concat, 4+c_dim ch", size=7, family=F, style="normal")
    box(ax, 2, 22, 19, 7, None, "c  stage, one-hot\n(currently the baseline grade)", "white", "black", ssize=7.4,
        family=F, r=1.2)
    path(ax, [(21, 25.5), (29, 25.5), (29, 40), (34, 40)], color="black")
    ax.add_patch(FancyArrowPatch((12, 22), (36, 36), connectionstyle="arc3,rad=0.45", arrowstyle="-|>",
                                 mutation_scale=10, color="black", lw=1.2, zorder=4))
    label(ax, 22, 19.3, "s = c (AdaIN style vector)", size=7, family=F, style="normal")
    trapezoid(ax, 34, 42, 13, 9, 20, G_FILL, G_EDGE, "Generator", family=F)
    label(ax, 34, 54.3, "initialised from stage 1\n(final-G.ckpt / final-D.ckpt)", size=6.9, family=F,
          color=G_EDGE)
    arrow(ax, (47, 42), (54, 42), color=G_EDGE)
    box(ax, 54, 35, 19, 14, None, None, "white", G_EDGE, ls=(0, (3, 2)), r=1.2)
    fundus(ax, 58.5, 42, 6, seed=6, lesions=10, synthetic=True)
    ax.text(62.3, 42, "G(x)\nsynthesized\nimage, target\nstage (3ch)", fontsize=7.4, va="center", family=F,
            linespacing=1.25)
    path(ax, [(73, 45), (85, 45), (85, 38)], color=G_EDGE)
    trapezoid(ax, 81, 30, 12, 18, 9, D_FILL, D_EDGE, "Discriminator", size=8, family=F)
    arrow(ax, (93, 30), (99, 30), color=D_EDGE)
    box(ax, 99, 23, 20, 14, None, "d_loss = fake − real + 10·GP\ng_loss = adv + λ_rec·L1(G(x), Reg(Y))\n"
        "λ_rec = 10, λ_adv = 1", "white", LOSS_EDGE, lw=1.3, ssize=6.6, family=F, r=1.5)
    # registration
    box(ax, 2, 3, 17, 12, None, None, "white", "black", ls=(0, (2, 2)), lw=1.3, r=1.2)
    fundus(ax, 6.5, 9, 6, seed=8, lesions=9)
    ax.text(10.3, 9, "Y\nreal follow-\nup (raw)", fontsize=7.4, va="center", family=F, linespacing=1.25)
    arrow(ax, (19, 9), (27, 9), color="black")
    trapezoid(ax, 27, 9, 11, 12, 6, R_FILL, R_EDGE, "Affine\nregistration", size=7.5, family=F)
    arrow(ax, (38, 9), (54, 9), color=R_EDGE)
    box(ax, 54, 2, 19, 14, None, None, "white", R_EDGE, ls=(0, (3, 2)), r=1.2)
    fundus(ax, 58.5, 9, 6, seed=8, lesions=9)
    ax.text(62.3, 9, "Reg(Y)\nfollow-up\nwarped to\nX's geometry", fontsize=7.2, va="center", family=F,
            linespacing=1.25)
    arrow(ax, (63.5, 35), (63.5, 16), color=G_EDGE)
    path(ax, [(73, 12), (87, 12), (87, 21.4)], color=R_EDGE)
    path(ax, [(73, 6), (94, 6)], color=R_EDGE)
    ax.add_patch(Circle((96.3, 6), 2.3, fc="white", ec="black", lw=1.1, zorder=4))
    ax.text(96.3, 6, "L1\nrec", ha="center", va="center", fontsize=6.8, zorder=5, family=F)
    # best epoch
    box(ax, 99, 40, 20, 14, None, "after every epoch:\nL1(G(x), Reg(Y)) on 10 %\nheld-out eyes / patients\n"
        "lowest → best-G.ckpt\n(50 epochs)", "#fff6d6", OUT_EDGE, ssize=6.8, family=F, r=1.5)
    label(ax, 27, 1.0, "pairs: FIRE + LongDR + Tianjin; registration net trained once, self-supervised "
          "(photometric loss)", size=6.6, family=F)
    save(fig, "module2_finetune_step")


# ---------------------------------------------------------------- 5. Module 2 inference cascade
def module2_inference():
    fig, ax = new_fig(120, 56)
    F = "serif"
    title(ax, 60, 52.5, "Module 2 - inference cascade (synthesize_trajectory())")
    box(ax, 2, 28, 24, 18, None, None, "white", "black", ls=(0, (2, 2)), lw=1.3, r=1.2)
    fundus(ax, 6.5, 41, 6, seed=9, lesions=4)
    mask_thumb(ax, 6.5, 33, 6, seed=10, n=5)
    ax.text(10.5, 37, "Xₙ, Mₙ\ncurrent image +\nmask (n=0: real\nbaseline)\n+ target stage c",
            fontsize=7.4, va="center", family=F, linespacing=1.25)
    arrow(ax, (26, 37), (36, 37), color=G_EDGE)
    trapezoid(ax, 36, 37, 13, 9, 20, G_FILL, G_EDGE, "Generator\n(best-G.ckpt)", size=8.5, family=F)
    arrow(ax, (49, 37), (58, 37), color=G_EDGE)
    box(ax, 58, 30, 20, 14, None, None, "white", G_EDGE, ls=(0, (3, 2)), r=1.2)
    fundus(ax, 62.5, 37, 6, seed=9, lesions=10, synthetic=True)
    ax.text(66.3, 37, "Xₙ₊₁\nsynthesized\nimage, stage\n= target c", fontsize=7.4, va="center", family=F,
            linespacing=1.25)
    arrow(ax, (68, 30), (68, 22), color=G_EDGE)
    box(ax, 50, 4, 36, 18, None, None, "white", B_EDGE, lw=1.4, r=1.5)
    ax.text(51.5, 13.5, "4 U-Nets re-segment Xₙ₊₁ → Mₙ₊₁\n"
            "ResNet-50 re-grades Xₙ₊₁ → predicted grade\nconsistency = (predicted grade == c),\nlogged only",
            fontsize=7.6, va="center", family=F, linespacing=1.3)
    ax.text(51.5, 19.6, "Module 1 (re-run)", fontsize=8, fontweight="bold", va="center", family=F, zorder=4,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.add_patch(FancyArrowPatch((50, 13), (14, 28), connectionstyle="arc3,rad=-0.3", arrowstyle="-|>",
                                 mutation_scale=10, color=B_EDGE, lw=1.3, zorder=4))
    label(ax, 3, 19, "feeds back as next Xₙ, Mₙ\n(c ← c + 1)", size=7.4, family=F, style="normal")
    arrow(ax, (86, 13), (93, 13), color=B_EDGE)
    box(ax, 93, 4, 25, 18, None, "stop when c = end stage\n(default 4, PDR)\ntrajectory =\n[baseline, X₁ … X_end]\n"
        "→ report (Module 3\nestimate per stage)", "white", "black", lw=1.4, ssize=7.3, family=F, r=1.5)
    save(fig, "module2_inference")


# ---------------------------------------------------------------- 6. Module 3 model
def module3():
    fig, ax = new_fig(120, 60)
    F = "serif"
    title(ax, 60, 56.5, "Module 3 - time-to-progression model (module3/model.py)")
    box(ax, 2, 31, 18, 16, None, None, "white", "black", ls=(0, (2, 2)), lw=1.3, r=1.2)
    fundus(ax, 6.8, 39, 6.5, seed=11, lesions=6)
    ax.text(10.8, 39, "X\nreal baseline\nimage only\n(380 × 380)", fontsize=7.4, va="center", family=F,
            linespacing=1.25)
    arrow(ax, (20, 39), (26, 39), color="black")
    trapezoid(ax, 26, 39, 16, 20, 8, M3_BOX, M3_EDGE, "EfficientNet-B4\n(ImageNet init,\nfine-tuned)", size=8,
              family=F)
    arrow(ax, (42, 39), (49, 39), color=M3_EDGE)
    label(ax, 42.5, 41.2, "1792-d", size=7, family=F, style="normal")
    box(ax, 2, 17, 18, 8, None, "LBS stratum\nlow / med / high\n(from Module 1)", "white", "black", ssize=7.2,
        family=F, r=1.2)
    box(ax, 2, 5, 18, 8, None, "baseline grade\nICDR 0–4\n(from Module 1)", "white", "black", ssize=7.2,
        family=F, r=1.2)
    box(ax, 26, 17, 16, 8, None, "Embedding\n3 → 8", "white", M3_EDGE, ssize=7.6, family=F, r=1.2)
    box(ax, 26, 5, 16, 8, None, "Embedding\n5 → 8", "white", M3_EDGE, ssize=7.6, family=F, r=1.2)
    arrow(ax, (20, 21), (26, 21), color="black")
    arrow(ax, (20, 9), (26, 9), color="black")
    path(ax, [(42, 21), (51, 21), (51, 33.5)], color=M3_EDGE)
    path(ax, [(42, 9), (53, 9), (53, 33.5)], color=M3_EDGE)
    box(ax, 49, 33.5, 8, 11, None, "concat\n1808", "white", M3_EDGE, ssize=7.2, family=F, r=1)
    arrow(ax, (57, 39), (61, 39), color=M3_EDGE)
    box(ax, 61, 31, 17, 16, None, "Linear 1808 → 128\nReLU\nDropout 0.3\nLinear 128 → 2\nsoftplus",
        "white", M3_EDGE, ssize=7.4, family=F, r=1.2)
    arrow(ax, (78, 39), (83, 39), color=M3_EDGE)
    box(ax, 83, 33, 14, 12, None, "shape k\nscale λ", "#fff6d6", OUT_EDGE, ssize=8.2, family=F, r=1.2)
    arrow(ax, (97, 39), (101, 39), color=OUT_EDGE)
    box(ax, 101, 30, 17, 18, None, "S(t) = exp(−(t/λ)^k)\n\nP(progression\nwithin 2 years)\n= 1 − S(2)\n"
        "+ survival curve", "white", "black", lw=1.4, ssize=7.3, family=F, r=1.5)
    # loss
    path(ax, [(90, 33), (90, 19)], color=OUT_EDGE)
    box(ax, 62, 4, 56, 15, None, None, "white", LOSS_EDGE, lw=1.3, r=1.5)
    ax.text(63.5, 11.5, "Training (Tianjin, baseline → fixed 2-year follow-up):\n"
            "event e = 1 if follow-up grade > baseline grade, observed only at t = 2 years\n"
            "current-status Weibull NLL = −[ e·log(1 − S(2)) + (1 − e)·log S(2) ]\n"
            "Adam lr 1e-4, 30 epochs, patient-level train / val split\n"
            "best epoch = lowest val NLL → best-model.ckpt; reported: AUROC + Brier at 2 y",
            fontsize=7.1, va="center", family=F, linespacing=1.4)
    save(fig, "module3_model")


if __name__ == "__main__":
    overview()
    module1()
    module2_pretrain()
    module2_finetune()
    module2_inference()
    module3()
