"""
Module 2 stage-1 pretraining on a NON-longitudinal graded dataset (EyePACS / APTOS / ...).

The longitudinal pair sets (FIRE + LongDR + Tianjin) are too small to teach the Generator what
each DR stage looks like on their own. This script pretrains the SAME Generator/Discriminator
(DRForestGAN-v2/base_model.py) on thousands of single graded images instead, then
train_module2_poc.py --init-generator/--init-discriminator fine-tunes them on the real pairs:

    stage 1 (this script):  cross-sectional, unpaired  -> learns stage appearance + biomarkers
    stage 2 (train_module2_poc.py): longitudinal pairs -> learns eye-specific progression

How a dataset without pairs can still supervise "make this eye look like stage k":
  * Stage classification (StarGAN, Choi et al. 2018): the Discriminator's existing out_cls head
    is trained to grade REAL images with their true label; the Generator is trained so its
    output is graded as the requested target stage.
  * Cycle reconstruction: G(G(x, k), source_stage) ~= x, so the eye's own anatomy (vessels,
    optic disc) is preserved -- the property a follow-up image must have.
  * Identity: when target == source stage, G(x, k) ~= x.
  * Biomarker-profile loss (the "learn the biomarker patterns per stage" part): a small
    BiomarkerRegressor is trained on real images to predict Module 1's per-lesion biomarkers
    (stage_biomarker_profiles.py). The Generator is penalised when the biomarkers of its
    stage-k output fall outside stage k's population profile (hinge on |z| > bio_margin).
  * Monotonic progression prior: when target > source stage, predicted lesion burden of the
    output should not be lower than the input's (lesions accumulate as DR worsens).
Target stages are sampled with probability p_adjacent as source+1 (the one-step transition the
trajectory cascade uses -- DRForestGAN-v2/trajectory_inference.py), otherwise uniformly.

Without --module1-cache / --profiles the biomarker terms are switched off and this reduces to
plain StarGAN-style stage translation, which still works -- the biomarker terms are an
addition, not a requirement.

Usage (Colab):
    # 0) Module 1 over EyePACS (per-lesion stats + downsampled masks):
    python module1/apply_to_progression_data.py --images-dir /content/data/eyepacs \
        --classifier-checkpoint .../final_weights.pt \
        --seg-checkpoint EX=... --seg-checkpoint HE=... --seg-checkpoint MA=... --seg-checkpoint SE=... \
        --mask-size 256 --out /content/drive/MyDrive/.../module1_outputs_eyepacs.pt
    # 1) Stage biomarker profiles:
    python stage_biomarker_profiles.py --module1-cache .../module1_outputs_eyepacs.pt \
        --labels-csv /content/data/eyepacs/all_labels.csv --out ./stage_biomarker_profiles.json
    # 2) Pretrain:
    python train_module2_nonlong_pretrain.py --dataset-dir /content/data/eyepacs --preset eyepacs_resized \
        --module1-cache .../module1_outputs_eyepacs.pt --profiles ./stage_biomarker_profiles.json \
        --save-dir ./DRForestGAN-v2/stargan/models_nonlong/
    # 3) Fine-tune on longitudinal pairs:
    python train_module2_poc.py --tianjin-dir ... --tianjin-module1-cache ... \
        --init-generator ./DRForestGAN-v2/stargan/models_nonlong/final-G.ckpt \
        --init-discriminator ./DRForestGAN-v2/stargan/models_nonlong/final-D.ckpt \
        --save-dir ./DRForestGAN-v2/stargan/models_poc_from_nonlong/

Dry run (synthetic images + labels, no data needed -- plumbing check only):
    python train_module2_nonlong_pretrain.py --dry-run --save-dir /tmp/nonlong_dry
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "DRForestGAN-v2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_model import Discriminator, Generator  # noqa: E402
from crosssectional_dataset import NUM_STAGES, get_crosssectional_loaders  # noqa: E402
from stage_biomarker_profiles import BIOMARKER_NAMES, N_BIOMARKERS, profile_tensors  # noqa: E402
from train_module2_poc import denorm, gradient_penalty  # noqa: E402


class Config:
    # Data
    dataset_dir = "./eyepacs"
    preset = "eyepacs_resized"
    module1_cache_path = None
    profiles_path = None
    image_size = 128
    batch_size = 16
    num_workers = 0
    val_fraction = 0.1
    max_per_grade = None
    samples_per_epoch = 20000  # grade-balanced draws per epoch (EyePACS has ~35k images)
    dataset_kwargs = {}  # labels_csv/images_dir/image_col/label_col/ext overrides (preset 'generic')

    # Model -- must match train_module2_poc.py so checkpoints transfer
    g_conv_dim = 64
    d_conv_dim = 64
    c_dim = NUM_STAGES
    style_dim = None
    g_repeat_num = 6
    d_repeat_num = 6

    # Training. Adam settings match train_module2_poc.py (DRForecastGAN's reported values).
    num_epochs = 20
    g_lr = 0.0002
    d_lr = 0.0002
    r_lr = 0.0002
    beta1 = 0.0
    beta2 = 0.9
    n_critic = 1

    lambda_adv = 1.0
    lambda_gp = 10.0
    lambda_cls = 1.0
    lambda_cyc = 10.0
    lambda_id = 5.0
    lambda_bio = 1.0
    lambda_mono = 1.0
    bio_margin = 1.0  # |z| within this many stage-SDs of the stage mean is not penalised
    bio_warmup_epochs = 1  # train the BiomarkerRegressor alone first so it isn't random
    p_adjacent = 0.5

    save_dir = "./DRForestGAN-v2/stargan/models_nonlong/"
    sample_dir = "./training_samples_nonlong/"
    eval_batches = 20


class BiomarkerRegressor(nn.Module):
    """Small CNN: fundus image in [-1,1] -> transformed biomarker vector (stage_biomarker_profiles).
    Deliberately image-only (no mask channel), since synthesized images have no mask."""

    def __init__(self, n_out=N_BIOMARKERS, width=32):
        super().__init__()
        layers, c = [], 3
        for w in (width, width * 2, width * 4, width * 8, width * 8):
            layers += [nn.Conv2d(c, w, 4, 2, 1), nn.InstanceNorm2d(w, affine=True), nn.LeakyReLU(0.2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_out)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def sample_targets(src, c_dim, p_adjacent):
    """Per-sample target stage: source+1 (capped) with prob p_adjacent, else uniform."""
    uniform = torch.randint(0, c_dim, src.shape, device=src.device)
    adjacent = (src + 1).clamp(max=c_dim - 1)
    use_adj = torch.rand(src.shape, device=src.device) < p_adjacent
    return torch.where(use_adj, adjacent, uniform)


def onehot(idx, c_dim):
    return F.one_hot(idx, c_dim).float()


def masked_mse(pred, target, mask):
    mask = mask.float()
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def biomarker_profile_loss(pred, trg, mu, sd, margin):
    """Hinge on how many stage-SDs pred sits from the target stage's population mean.
    Huber-shaped (quadratic just past the margin, linear further out) so a few far-off
    predictions early in training can't swamp the adversarial/cycle terms."""
    m, s = mu[trg], sd[trg]
    valid = torch.isfinite(m)
    z = (pred - torch.nan_to_num(m)) / torch.nan_to_num(s, nan=1.0)
    excess = F.relu(z.abs() - margin)
    pen = F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none")
    return (pen * valid).sum() / valid.sum().clamp(min=1)


def monotonic_loss(pred_fake, pred_real, src, trg):
    """When moving to a worse stage, predicted biomarkers shouldn't drop below the input's."""
    worse = (trg > src).float().unsqueeze(1)
    pen = F.relu(pred_real.detach() - pred_fake) * worse
    return pen.sum() / (worse.sum() * pred_fake.size(1)).clamp(min=1)


@torch.no_grad()
def evaluate(G, D, R, loader, device, config, mu, sd, use_bio):
    """
    On the held-out (stratified) val split:
      real_cls_acc         D's grading accuracy on real val images (is the stage head useful?)
      fake_cls_agreement   per target stage, fraction of G's outputs D grades as that stage
      bio_mean_abs_z       per target stage, mean |z| of R's biomarkers vs that stage's profile
      lbs_monotonic_rate   fraction of val images whose predicted LBS is non-decreasing
                           across synthesized stages 0..4
    D and R are trained jointly with G, so the fake-side numbers are self-consistency signals,
    not independent validation -- use evaluate_trajectory.py with Module 1 for that.
    """
    G.eval(); D.eval(); R.eval()
    n_real = n_correct = 0
    agree = np.zeros(config.c_dim); seen = np.zeros(config.c_dim)
    absz = np.zeros(config.c_dim); absz_n = np.zeros(config.c_dim)
    mono_hits = mono_n = 0
    lbs_idx = BIOMARKER_NAMES.index("lbs")
    for b, batch in enumerate(loader):
        if b >= config.eval_batches:
            break
        x = batch["image"].to(device)
        mask = batch["mask"].to(device)
        src = batch["grade"].to(device)
        _, cls_real = D(x)
        n_correct += int((cls_real.argmax(1) == src).sum()); n_real += x.size(0)
        lbs_per_stage = []
        for k in range(config.c_dim):
            trg = torch.full_like(src, k)
            fake = G(torch.cat([x, mask], 1), onehot(trg, config.c_dim))
            _, cls_fake = D(fake)
            agree[k] += int((cls_fake.argmax(1) == k).sum()); seen[k] += x.size(0)
            if use_bio:
                pred = R(fake)
                m, s = mu[trg], sd[trg]
                valid = torch.isfinite(m)
                z = ((pred - torch.nan_to_num(m)) / torch.nan_to_num(s, nan=1.0)).abs()
                absz[k] += float((z * valid).sum()); absz_n[k] += float(valid.sum())
                lbs_per_stage.append(pred[:, lbs_idx])
        if use_bio and lbs_per_stage:
            L = torch.stack(lbs_per_stage, 1)
            mono_hits += int((L[:, 1:] >= L[:, :-1] - 1e-3).all(1).sum()); mono_n += L.size(0)
    G.train(); D.train(); R.train()
    return {
        "n_val_images": n_real,
        "real_cls_acc": n_correct / max(n_real, 1),
        "fake_cls_agreement": {str(k): agree[k] / seen[k] if seen[k] else None for k in range(config.c_dim)},
        "bio_mean_abs_z": ({str(k): absz[k] / absz_n[k] if absz_n[k] else None for k in range(config.c_dim)}
                           if use_bio else None),
        "lbs_monotonic_rate": mono_hits / mono_n if mono_n else None,
    }


@torch.no_grad()
def save_stage_grid(G, batch, device, c_dim, path, n=4):
    """Rows: one val image each; columns: source, then synthesized stage 0..c_dim-1."""
    G.eval()
    x = batch["image"][:n].to(device)
    mask = batch["mask"][:n].to(device)
    cols = [denorm(x.clone())]
    for k in range(c_dim):
        trg = torch.full((x.size(0),), k, device=device, dtype=torch.long)
        cols.append(denorm(G(torch.cat([x, mask], 1), onehot(trg, c_dim))))
    grid = torch.stack(cols, 1).reshape(-1, *x.shape[1:])
    save_image(grid, path, nrow=c_dim + 1, padding=4)
    G.train()


def make_dry_run_data(root, n_per_grade=6, size=64):
    """Writes a tiny fake dataset in the resized-EyePACS layout (train/ + test/ subfolders, one
    merged all_labels.csv, relpath-style Module 1 cache keys) + profiles, for plumbing tests."""
    from PIL import Image

    from stage_biomarker_profiles import build_profiles

    rng = np.random.RandomState(0)
    rows, cache, labels = [], {}, {}
    for g in range(NUM_STAGES):
        for i in range(n_per_grade):
            split = "train" if i % 2 == 0 else "test"
            os.makedirs(os.path.join(root, split), exist_ok=True)
            name = f"{g}{i}_left"
            arr = (rng.rand(size, size, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(root, split, name + ".jpeg"))
            rows.append(f"{name}.jpeg,{g},{split}")
            labels[name] = g
            m = (rng.rand(32, 32) < 0.02 * (g + 1)).astype(np.uint8)
            cache[f"{split}__{name}"] = {
                "grade": g, "mask": m, "lbs": float(m.mean()), "trained_lesions": ["EX", "MA"],
                "lesion_stats": {"EX": {"area_frac": 0.001 * g + rng.rand() * 1e-4, "count": g * 3},
                                 "MA": {"area_frac": 0.0005 * g + rng.rand() * 1e-4, "count": g * 5}},
            }
    with open(os.path.join(root, "all_labels.csv"), "w") as f:
        f.write("image,level,split\n" + "\n".join(rows) + "\n")
    torch.save(cache, os.path.join(root, "module1_cache.pt"))
    stem_cache = {k.split("__")[-1]: v for k, v in cache.items()}
    with open(os.path.join(root, "profiles.json"), "w") as f:
        json.dump(build_profiles(stem_cache, labels), f)
    return root


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Module 2 stage-1 pretraining on cross-sectional data")
    print("=" * 70)
    print(f"Device: {device} | dataset: {config.dataset_dir} ({config.preset})")
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    print("\n[1/4] Loading cross-sectional dataset...")
    train_loader, val_loader, _ = get_crosssectional_loaders(
        config.dataset_dir, preset=config.preset, batch_size=config.batch_size,
        image_size=config.image_size, module1_cache_path=config.module1_cache_path,
        val_fraction=config.val_fraction, max_per_grade=config.max_per_grade,
        samples_per_epoch=config.samples_per_epoch, num_workers=config.num_workers,
        **config.dataset_kwargs,
    )

    use_bio = bool(config.module1_cache_path and config.profiles_path)
    if use_bio:
        with open(config.profiles_path) as f:
            profiles = json.load(f)
        mu_np, sd_np = profile_tensors(profiles, config.c_dim)
        mu, sd = torch.from_numpy(mu_np).to(device), torch.from_numpy(sd_np).to(device)
        print(f"✓ Biomarker profiles loaded ({config.profiles_path}) -- biomarker losses ON")
    else:
        mu = sd = None
        print("⚠ No --module1-cache/--profiles -- biomarker losses OFF (plain stage translation)")

    print("\n[2/4] Initializing models...")
    G = Generator(conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num,
                  style_dim=config.style_dim).to(device)
    D = Discriminator(image_size=config.image_size, conv_dim=config.d_conv_dim, c_dim=config.c_dim,
                      repeat_num=config.d_repeat_num).to(device)
    R = BiomarkerRegressor().to(device)
    g_opt = optim.Adam(G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
    d_opt = optim.Adam(D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
    r_opt = optim.Adam(R.parameters(), lr=config.r_lr, betas=(0.9, 0.999))

    print(f"\n[3/4] Training for {config.num_epochs} epochs ({len(train_loader)} batches/epoch)...")
    start = time.time()
    history = []
    for epoch in range(config.num_epochs):
        bio_on = use_bio and epoch >= config.bio_warmup_epochs
        sums = dict(d=0.0, g=0.0, cls=0.0, cyc=0.0, bio=0.0, mono=0.0, r=0.0)
        n_g = 0
        for i, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            mask = batch["mask"].to(device)
            src = batch["grade"].to(device)
            c_src = onehot(src, config.c_dim)
            trg = sample_targets(src, config.c_dim, config.p_adjacent)
            c_trg = onehot(trg, config.c_dim)
            x_in = torch.cat([x, mask], 1)

            # --- Biomarker regressor on real images (Module 1 biomarkers as labels) ---
            if use_bio:
                bio_mask = batch["bio_mask"].to(device)
                r_loss = masked_mse(R(x), batch["biomarkers"].to(device), bio_mask)
                r_opt.zero_grad(); r_loss.backward(); r_opt.step()
                sums["r"] += r_loss.item()

            # --- Discriminator: real/fake + grade real images with their true stage ---
            out_src, out_cls = D(x)
            d_real = -out_src.mean()
            d_cls = F.cross_entropy(out_cls, src)
            with torch.no_grad():
                fake = G(x_in, c_trg)
            out_fake, _ = D(fake)
            gp = gradient_penalty(D, x, fake, device)
            d_loss = d_real + out_fake.mean() + config.lambda_gp * gp + config.lambda_cls * d_cls
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()
            sums["d"] += d_loss.item()

            # --- Generator ---
            if i % config.n_critic == 0:
                fake = G(x_in, c_trg)
                out_fake, cls_fake = D(fake)
                g_adv = -out_fake.mean()
                g_cls = F.cross_entropy(cls_fake, trg)
                # Cycle back to the source stage with the same (source) mask -- the synthesized
                # image has no Module 1 mask of its own during training.
                rec = G(torch.cat([fake, mask], 1), c_src)
                g_cyc = F.l1_loss(rec, x)
                same = trg == src
                g_id = F.l1_loss(fake[same], x[same]) if same.any() else fake.new_zeros(())
                g_loss = (config.lambda_adv * g_adv + config.lambda_cls * g_cls
                          + config.lambda_cyc * g_cyc + config.lambda_id * g_id)
                if bio_on:
                    pred_fake = R(fake)
                    with torch.no_grad():
                        pred_real = R(x)
                    g_bio = biomarker_profile_loss(pred_fake, trg, mu, sd, config.bio_margin)
                    g_mono = monotonic_loss(pred_fake, pred_real, src, trg)
                    g_loss = g_loss + config.lambda_bio * g_bio + config.lambda_mono * g_mono
                    sums["bio"] += g_bio.item(); sums["mono"] += g_mono.item()
                g_opt.zero_grad(); g_loss.backward(); g_opt.step()
                # g_loss.backward() also reaches R's params through R(fake); R is only updated
                # by its own real-image loss, so drop those gradients.
                r_opt.zero_grad()
                sums["g"] += g_loss.item(); sums["cls"] += g_cls.item(); sums["cyc"] += g_cyc.item()
                n_g += 1

        nb = max(len(train_loader), 1)
        rec_ = {"epoch": epoch + 1, "d_loss": sums["d"] / nb, "r_loss": sums["r"] / nb,
                **{f"g_{k}": sums[k] / max(n_g, 1) for k in ("g", "cls", "cyc", "bio", "mono")},
                "biomarker_losses_on": bio_on}
        history.append(rec_)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] D {rec_['d_loss']:.3f} | G {rec_['g_g']:.3f} "
              f"(cls {rec_['g_cls']:.3f}, cyc {rec_['g_cyc']:.3f}, bio {rec_['g_bio']:.3f}, "
              f"mono {rec_['g_mono']:.3f}) | R {rec_['r_loss']:.3f} | {(time.time() - start) / 60:.1f}min")

        val_batch = next(iter(val_loader), None)
        if val_batch is not None:
            path = os.path.join(config.sample_dir, f"epoch_{epoch + 1}.jpg")
            save_stage_grid(G, val_batch, device, config.c_dim, path)
        torch.save(G.state_dict(), os.path.join(config.save_dir, f"{epoch + 1}-G.ckpt"))

    print("\n[4/4] Evaluating on the held-out val split...")
    metrics = evaluate(G, D, R, val_loader, device, config, mu, sd, use_bio)
    print(json.dumps(metrics, indent=2))

    torch.save(G.state_dict(), os.path.join(config.save_dir, "final-G.ckpt"))
    torch.save(D.state_dict(), os.path.join(config.save_dir, "final-D.ckpt"))
    torch.save(R.state_dict(), os.path.join(config.save_dir, "final-R.ckpt"))
    results = {
        "stage": "nonlong_pretrain",
        "dataset_dir": config.dataset_dir, "preset": config.preset,
        "module1_cache_path": config.module1_cache_path, "profiles_path": config.profiles_path,
        "num_epochs": config.num_epochs, "image_size": config.image_size,
        "lambdas": {k: getattr(config, k) for k in dir(config) if k.startswith("lambda_")},
        "bio_margin": config.bio_margin, "p_adjacent": config.p_adjacent,
        "metrics": metrics, "loss_history": history,
        "total_time_min": (time.time() - start) / 60,
    }
    with open(os.path.join(config.save_dir, "nonlong_pretrain_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved final-G/D/R.ckpt + nonlong_pretrain_results.json to {config.save_dir}")
    print("  Next: train_module2_poc.py --init-generator <save_dir>/final-G.ckpt "
          "--init-discriminator <save_dir>/final-D.ckpt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=Config.dataset_dir)
    p.add_argument("--preset", default=Config.preset, choices=["eyepacs_resized", "eyepacs", "aptos", "generic"])
    p.add_argument("--labels-csv", default=None, help="Override the preset's labels CSV (relative to --dataset-dir)")
    p.add_argument("--images-subdir", default=None, help="Override the preset's images folder")
    p.add_argument("--image-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--ext", default=None, help="Only index images with this extension, e.g. .jpeg")
    p.add_argument("--module1-cache", default=None)
    p.add_argument("--profiles", default=None, help="stage_biomarker_profiles.py JSON")
    p.add_argument("--image-size", type=int, default=Config.image_size)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--num-epochs", type=int, default=Config.num_epochs)
    p.add_argument("--samples-per-epoch", type=int, default=Config.samples_per_epoch)
    p.add_argument("--max-per-grade", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument("--eval-batches", type=int, default=Config.eval_batches)
    for name in ("lambda_cls", "lambda_cyc", "lambda_id", "lambda_bio", "lambda_mono", "bio_margin", "p_adjacent"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(Config, name))
    p.add_argument("--bio-warmup-epochs", type=int, default=Config.bio_warmup_epochs)
    p.add_argument("--save-dir", default=Config.save_dir)
    p.add_argument("--sample-dir", default=Config.sample_dir)
    p.add_argument("--dry-run", action="store_true",
                   help="Generate a tiny synthetic dataset under --save-dir and run 2 short epochs")
    args = p.parse_args()

    cfg = Config()
    for k, v in vars(args).items():
        if k not in ("dry_run", "profiles", "module1_cache") and hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.module1_cache_path = args.module1_cache
    cfg.profiles_path = args.profiles
    cfg.dataset_kwargs = {k: v for k, v in {"labels_csv": args.labels_csv, "images_dir": args.images_subdir,
                                            "image_col": args.image_col, "label_col": args.label_col,
                                            "ext": args.ext}.items() if v is not None}

    if args.dry_run:
        root = make_dry_run_data(os.path.join(args.save_dir, "dry_run_data"))
        cfg.dataset_dir, cfg.preset = root, "eyepacs_resized"
        cfg.module1_cache_path = os.path.join(root, "module1_cache.pt")
        cfg.profiles_path = os.path.join(root, "profiles.json")
        cfg.image_size, cfg.batch_size, cfg.num_epochs = 64, 4, 2
        cfg.samples_per_epoch, cfg.eval_batches, cfg.val_fraction = 12, 2, 0.2
        cfg.g_conv_dim = cfg.d_conv_dim = 16
        cfg.g_repeat_num = 2
        cfg.sample_dir = os.path.join(args.save_dir, "samples")

    train(cfg)
