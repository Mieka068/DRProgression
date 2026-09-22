"""
Module 2 proof-of-concept training run, using REAL Module 1 conditioning (grade + lesion
mask from module1/apply_to_progression_data.py's cache) instead of the empty-mask/Stage-2
placeholder that train_fire.py / train_combined.py fall back to.

This is a thin variant of train_combined.py, not a rewrite -- same Generator/Discriminator
(DRForestGAN-v2/base_model.py), same WGAN-GP-style training loop, same 4-channel
(image + mask) conditioning input train_combined.py already implements. The only things that
change:
  1. fire_module1_cache_path / longdr_module1_cache_path / tianjin_dir /
     tianjin_module1_cache_path are threaded through to get_combined_loader() ->
     FIREDataset/LongDRScreeningDataset/TianjinLongitudinalDataset, so real grade/mask flow
     in. tianjin_dir is optional (default None) -- omit it to train on FIRE+LongDR only,
     exactly as before Tianjin was wired in.
  2. num_epochs is reduced to 5 for a short POC run (see Config.num_epochs below), well below
     the 50 epochs train_fire.py/train_combined.py use, so a full run finishes and produces
     samples/metrics in a single session.
  3. Optimizer settings are changed FROM train_combined.py's existing (lr=1e-4, β1=0.5,
     β2=0.999) TO lr=2e-4, β1=0.0, β2=0.9, matching DRForecastGAN's reported Adam
     hyperparameters.
  4. At the end of training, computes FID/PSNR/SSIM (torchmetrics + torch-fidelity) between
     synthesized and real follow-up images on a held-out slice of the loader -- small/short-run
     numbers, not a claim of matching DRForecastGAN's published benchmark
     (FID 27.3/PSNR 25.3/SSIM 0.93). This single-step number does not report per-cascade-step
     quality -- run evaluate_trajectory.py separately afterward for that.
  5. The Generator (DRForestGAN-v2/base_model.py) uses AdaIN stage conditioning in its
     bottleneck, additive to the original channel-concat conditioning -- this changes its
     state_dict keys, so a checkpoint from before this change will not load here.
  6. registration_cache_path, if given, substitutes a pre-registered (baseline-aligned)
     follow-up image wherever module1/train_registration.py produced one, in place of the raw
     follow-up image. None (the default) uses raw follow-up images everywhere.

Usage (Colab, after Module 1 checkpoints + cache files exist):
    python train_module2_poc.py \
        --fire-module1-cache /content/drive/MyDrive/.../module1_outputs_fire.pt \
        --longdr-module1-cache /content/drive/MyDrive/.../module1_outputs_longdr.pt \
        --num-epochs 5

With Tianjin included (see notebooks/05_tianjin_data_prep_colab.ipynb and
notebooks/06_module2_poc_with_tianjin_colab.ipynb):
    python train_module2_poc.py \
        --fire-module1-cache /content/drive/MyDrive/.../module1_outputs_fire.pt \
        --longdr-module1-cache /content/drive/MyDrive/.../module1_outputs_longdr.pt \
        --tianjin-dir /content/data/retinal-dr-longitudinal \
        --tianjin-module1-cache /content/drive/MyDrive/.../module1_outputs_tianjin.pt \
        --num-epochs 5
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

sys.path.insert(0, "./DRForestGAN-v2")
from base_model import Generator, Discriminator  # noqa: E402
from combined_dataset import get_combined_loader  # noqa: E402


class Config:
    # Data
    fire_dir = "./FIRE_dataset"
    longdr_dir = "./LongDRScreening_20150209"
    tianjin_dir = None  # optional -- e.g. "./retinal-dr-longitudinal"; None skips Tianjin
    image_size = 128
    batch_size = 4
    augment = True
    num_workers = 0
    fire_module1_cache_path = None
    longdr_module1_cache_path = None
    tianjin_module1_cache_path = None
    tianjin_min_pair_quality = None
    registration_cache_path = None  # optional -- see module1/train_registration.py

    # Model
    g_conv_dim = 64
    d_conv_dim = 64
    c_dim = 5
    style_dim = None  # AdaIN style vector size; None -> defaults to c_dim (see base_model.py)
    g_repeat_num = 6
    d_repeat_num = 6

    # Training -- POC-scale (5 epochs). See module docstring point 2.
    num_epochs = 5
    # Adam settings matching DRForecastGAN's reported hyperparameters, overriding
    # train_combined.py's own defaults (1e-4, 0.5, 0.999) -- see docstring point 3.
    g_lr = 0.0002
    d_lr = 0.0002
    beta1 = 0.0
    beta2 = 0.9
    n_critic = 1

    lambda_rec = 10.0
    lambda_adv = 1.0

    save_dir = "./DRForestGAN-v2/stargan/models_poc/"
    sample_dir = "./training_samples_poc/"
    log_step = 1
    save_step = 1
    eval_batches = 5  # how many held-out batches to use for FID/PSNR/SSIM at the end
    min_n_for_fid = 50  # below this, FID (feature=2048) is unreliable -- report None instead


def gradient_penalty(D, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def denorm(x):
    return ((x + 1) / 2).clamp_(0, 1)


def compute_poc_metrics(G, loader, device, n_batches, image_size, min_n_for_fid=50):
    """
    FID/PSNR/SSIM between synthesized follow-ups and real follow-ups, over a handful of
    batches -- a small/short-run number, explicitly not a claim of matching DRForecastGAN's
    published benchmark (see module docstring point 4). Requires torchmetrics AND
    torch-fidelity (pip install torchmetrics torch-fidelity on Colab -- FID specifically
    needs the latter, PSNR/SSIM don't).

    FID uses feature=2048 (standard Inception pool features), which needs more real samples
    to be numerically stable than the feature=64 variant does -- below min_n_for_fid samples,
    'fid' is reported as None (with 'fid_note' explaining why) rather than as an unstable
    number, the same pattern evaluate_trajectory.py uses for its per-cascade-step FID.
    """
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.fid import FrechetInceptionDistance

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # FID's InceptionV3 backbone expects >=75x75 images; our POC image_size (128) is fine.
    # feature=2048 (the standard Inception pool features, Heusel et al. 2017) matches the
    # convention DRForecastGAN's own reported FID is assumed to use -- feature=64 is a
    # low-sample-friendly variant unlikely to be what a published FID number reflects.
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    G.eval()
    n_seen = 0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            if n_seen >= n_batches:
                break
            baseline = batch["baseline"].to(device)
            follow_up = batch["follow_up"].to(device)
            mask = batch["mask"].to(device)
            target_grade = batch["target_grade"].to(device)

            x_real = torch.cat([baseline, mask], dim=1)
            fake = G(x_real, target_grade)

            real_01 = denorm(follow_up)
            fake_01 = denorm(fake)

            psnr_metric.update(fake_01, real_01)
            ssim_metric.update(fake_01, real_01)
            fid_metric.update(real_01, real=True)
            fid_metric.update(fake_01, real=False)
            n_seen += 1
            n_samples += baseline.size(0)
    G.train()

    result = {
        "psnr": float(psnr_metric.compute()),
        "ssim": float(ssim_metric.compute()),
        "n_batches_used": n_seen,
        "n_samples_used": n_samples,
    }
    if n_samples >= min_n_for_fid:
        result["fid"] = float(fid_metric.compute())
    else:
        result["fid"] = None
        result["fid_note"] = (
            f"n_samples ({n_samples}) < min_n_for_fid ({min_n_for_fid}) -- FID (feature=2048) "
            "is unreliable on this few samples, so it's omitted here rather than reported "
            "misleadingly."
        )
    return result


def train(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Module 2 POC training (real Module 1 conditioning)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"fire_module1_cache_path: {config.fire_module1_cache_path}")
    print(f"longdr_module1_cache_path: {config.longdr_module1_cache_path}")
    print(f"tianjin_dir: {config.tianjin_dir}")
    print(f"tianjin_module1_cache_path: {config.tianjin_module1_cache_path}")

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    print("\n[1/4] Loading combined dataset (with real Module 1 conditioning where cached)...")
    loader, num_pairs = get_combined_loader(
        fire_dir=config.fire_dir,
        longdr_dir=config.longdr_dir,
        tianjin_dir=config.tianjin_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        augment=config.augment,
        fire_module1_cache_path=config.fire_module1_cache_path,
        longdr_module1_cache_path=config.longdr_module1_cache_path,
        tianjin_module1_cache_path=config.tianjin_module1_cache_path,
        tianjin_min_pair_quality=config.tianjin_min_pair_quality,
        registration_cache_path=config.registration_cache_path,
        num_workers=config.num_workers,
    )
    print(f"✓ Real pairs: {num_pairs}, effective samples: {len(loader.dataset)}, batches/epoch: {len(loader)}")

    print("\n[2/4] Initializing models...")
    G = Generator(conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num,
                  style_dim=config.style_dim).to(device)
    D = Discriminator(
        image_size=config.image_size, conv_dim=config.d_conv_dim, c_dim=config.c_dim, repeat_num=config.d_repeat_num
    ).to(device)

    g_optimizer = optim.Adam(G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
    d_optimizer = optim.Adam(D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
    l1_loss = nn.L1Loss()

    print(f"\n[3/4] Training for {config.num_epochs} epochs (POC scale -- see module docstring)...")
    start_time = time.time()
    x_real = target_grade = baseline = follow_up = None  # for the post-loop sample save
    loss_history = []  # per-epoch {epoch, d_loss, g_loss, rec_loss} -- for the G/D loss curve figure

    for epoch in range(config.num_epochs):
        epoch_g_loss = epoch_d_loss = epoch_rec_loss = 0
        for i, batch in enumerate(loader):
            baseline = batch["baseline"].to(device)
            follow_up = batch["follow_up"].to(device)
            mask = batch["mask"].to(device)
            target_grade = batch["target_grade"].to(device)

            x_real = torch.cat([baseline, mask], dim=1)

            out_real, _ = D(follow_up)
            d_loss_real = -torch.mean(out_real)
            with torch.no_grad():
                fake = G(x_real, target_grade)
            out_fake, _ = D(fake)
            d_loss_fake = torch.mean(out_fake)
            gp = gradient_penalty(D, follow_up, fake, device)
            d_loss = d_loss_real + d_loss_fake + 10 * gp

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            if i % config.n_critic == 0:
                fake = G(x_real, target_grade)
                out_fake, _ = D(fake)
                g_loss_adv = -torch.mean(out_fake)
                g_loss_rec = l1_loss(fake, follow_up)
                g_loss = config.lambda_adv * g_loss_adv + config.lambda_rec * g_loss_rec

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                epoch_g_loss += g_loss.item()
                epoch_rec_loss += g_loss_rec.item()
            epoch_d_loss += d_loss.item()

        elapsed = time.time() - start_time
        mean_d_loss = epoch_d_loss / len(loader)
        mean_g_loss = epoch_g_loss / len(loader)
        mean_rec_loss = epoch_rec_loss / len(loader)
        loss_history.append({
            "epoch": epoch + 1, "d_loss": mean_d_loss, "g_loss": mean_g_loss, "rec_loss": mean_rec_loss,
        })
        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}] | D_loss: {mean_d_loss:.4f} | "
            f"G_loss: {mean_g_loss:.4f} | Rec_loss: {mean_rec_loss:.4f} | "
            f"Elapsed: {elapsed / 60:.1f}min"
        )

        if (epoch + 1) % config.save_step == 0 or epoch == config.num_epochs - 1:
            G.eval()
            with torch.no_grad():
                sample_fake = G(x_real, target_grade)
                samples = torch.cat(
                    [denorm(baseline[:2]), denorm(sample_fake[:2]), denorm(follow_up[:2])], dim=3
                )
                sample_path = os.path.join(config.sample_dir, f"epoch_{epoch + 1}.jpg")
                save_image(samples, sample_path, nrow=1, padding=10)
                print(f"  ✓ Saved samples to {sample_path}")
            G.train()
            torch.save(G.state_dict(), os.path.join(config.save_dir, f"{epoch + 1}-G.ckpt"))

    print("\n[4/4] Computing POC metrics (FID/PSNR/SSIM on a held-out slice)...")
    try:
        metrics = compute_poc_metrics(
            G, loader, device, config.eval_batches, config.image_size, min_n_for_fid=config.min_n_for_fid
        )
        fid_str = f"{metrics['fid']:.3f}" if metrics["fid"] is not None else f"None ({metrics.get('fid_note', '')})"
        print(f"✓ PSNR: {metrics['psnr']:.3f} | SSIM: {metrics['ssim']:.3f} | FID: {fid_str} "
              f"(n_batches={metrics['n_batches_used']}, n_samples={metrics['n_samples_used']})")
    except ImportError:
        print("⚠ torchmetrics/torch-fidelity not installed -- skipping FID/PSNR/SSIM "
              "(pip install torchmetrics torch-fidelity)")
        metrics = None

    final_g_path = os.path.join(config.save_dir, "final-G.ckpt")
    torch.save(G.state_dict(), final_g_path)

    results = {
        "num_epochs": config.num_epochs,
        "num_pairs": num_pairs,
        "effective_samples": len(loader.dataset),
        "g_lr": config.g_lr,
        "d_lr": config.d_lr,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "fire_module1_cache_path": config.fire_module1_cache_path,
        "longdr_module1_cache_path": config.longdr_module1_cache_path,
        "tianjin_dir": config.tianjin_dir,
        "tianjin_module1_cache_path": config.tianjin_module1_cache_path,
        "registration_cache_path": config.registration_cache_path,
        "metrics": metrics,
        "loss_history": loss_history,
        "total_time_min": (time.time() - start_time) / 60,
    }
    results_path = os.path.join(config.save_dir, "poc_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Wrote {results_path}")
    print(f"✓ Final Generator saved to {final_g_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire-dir", default=Config.fire_dir)
    parser.add_argument("--longdr-dir", default=Config.longdr_dir)
    parser.add_argument("--tianjin-dir", default=Config.tianjin_dir,
                         help="Optional path to the extracted Tianjin dataset; omit to skip it")
    parser.add_argument("--fire-module1-cache", default=None)
    parser.add_argument("--longdr-module1-cache", default=None)
    parser.add_argument("--tianjin-module1-cache", default=None)
    parser.add_argument("--tianjin-min-pair-quality", type=float, default=None)
    parser.add_argument("--registration-cache", default=None,
                         help="Optional path to a module1/train_registration.py cache; omit to "
                              "use raw follow-up images")
    parser.add_argument("--style-dim", type=int, default=None,
                         help="AdaIN style vector size; omit to default to --c-dim")
    parser.add_argument("--num-epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers,
                         help="DataLoader worker processes for image decode/resize/augment. "
                              "0 (the default) loads serially on the main process, which can "
                              "make the GPU sit idle between batches -- try 2 on Colab.")
    parser.add_argument("--save-dir", default=Config.save_dir,
                         help="Where to write {epoch}-G.ckpt / final-G.ckpt / poc_results.json. "
                              "Give each run its own directory -- this script always trains G/D "
                              "from scratch and will silently overwrite an earlier run's "
                              "checkpoints left at the same path.")
    parser.add_argument("--min-n-for-fid", type=int, default=Config.min_n_for_fid,
                         help="Minimum eval samples before FID (feature=2048) is reported "
                              "instead of None.")
    args = parser.parse_args()

    cfg = Config()
    cfg.fire_dir = args.fire_dir
    cfg.longdr_dir = args.longdr_dir
    cfg.tianjin_dir = args.tianjin_dir
    cfg.fire_module1_cache_path = args.fire_module1_cache
    cfg.longdr_module1_cache_path = args.longdr_module1_cache
    cfg.tianjin_module1_cache_path = args.tianjin_module1_cache
    cfg.tianjin_min_pair_quality = args.tianjin_min_pair_quality
    cfg.registration_cache_path = args.registration_cache
    cfg.style_dim = args.style_dim
    cfg.num_epochs = args.num_epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.save_dir = args.save_dir
    cfg.min_n_for_fid = args.min_n_for_fid

    train(cfg)
