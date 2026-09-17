"""
Train DRForecastGAN on FIRE Dataset
Simple training script for paired image-to-image translation

Usage:
    python train_fire.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import time

# Add DRForecastGAN to path
sys.path.insert(0, './DRForestGAN-v2')
from base_model import Generator, Discriminator
from fire_dataset import FIREDataset


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Data
    fire_dir = './FIRE_dataset'
    image_size = 128
    batch_size = 2
    
    # Model
    g_conv_dim = 64
    d_conv_dim = 64
    c_dim = 5  # Number of severity stages
    g_repeat_num = 6
    d_repeat_num = 6
    
    # Training
    num_epochs = 100  # Start small, increase if needed
    g_lr = 0.0001
    d_lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    n_critic = 1  # Train D once per G update
    
    # Loss weights
    lambda_rec = 10.0  # Reconstruction loss
    lambda_adv = 1.0   # Adversarial loss
    
    # Logging & Saving
    save_dir = './DRForestGAN-v2/stargan/models/'
    sample_dir = './training_samples/'
    log_step = 5
    save_step = 25  # Save every 25 epochs


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def gradient_penalty(D, real, fake, device):
    """Compute gradient penalty for WGAN-GP style training"""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    d_interpolates, _ = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def denorm(x):
    """Denormalize from [-1, 1] to [0, 1]"""
    return ((x + 1) / 2).clamp_(0, 1)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train():
    config = Config()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("DRForecastGAN Training on FIRE Dataset")
    print("="*70)
    print(f"Device: {device}")
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    
    # Load dataset
    print("\n[1/4] Loading FIRE dataset...")
    dataset = FIREDataset(
        config.fire_dir,
        image_size=config.image_size,
        category='A'  # Disease progression pairs
    )
    
    if len(dataset) == 0:
        print("❌ No image pairs found! Check FIRE_dataset folder structure.")
        return
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    print(f"✓ Loaded {len(dataset)} pairs")
    print(f"✓ Batches per epoch: {len(loader)}")
    
    # Initialize models
    print("\n[2/4] Initializing models...")
    G = Generator(conv_dim=config.g_conv_dim, c_dim=config.c_dim, repeat_num=config.g_repeat_num).to(device)
    D = Discriminator(image_size=config.image_size, conv_dim=config.d_conv_dim, 
                      c_dim=config.c_dim, repeat_num=config.d_repeat_num).to(device)
    
    print(f"✓ Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
    print(f"✓ Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")
    
    # Optimizers
    g_optimizer = optim.Adam(G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
    d_optimizer = optim.Adam(D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
    
    # Loss functions
    l1_loss = nn.L1Loss()
    
    # Training loop
    print(f"\n[3/4] Starting training for {config.num_epochs} epochs...")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_rec_loss = 0
        
        for i, batch in enumerate(loader):
            baseline = batch['baseline'].to(device)      # [B, 3, H, W]
            follow_up = batch['follow_up'].to(device)    # [B, 3, H, W]
            mask = batch['mask'].to(device)              # [B, 1, H, W]
            target_grade = batch['target_grade'].to(device)  # [B, 5]
            
            batch_size = baseline.size(0)
            
            # ============================================
            # Concat image + mask for 4-channel input
            # ============================================
            x_real = torch.cat([baseline, mask], dim=1)  # [B, 4, H, W]
            
            # ============================================
            # Train Discriminator
            # ============================================
            # Real images
            out_real, _ = D(follow_up)
            d_loss_real = -torch.mean(out_real)
            
            # Fake images
            with torch.no_grad():
                fake = G(x_real, target_grade)
            
            out_fake, _ = D(fake)
            d_loss_fake = torch.mean(out_fake)
            
            # Gradient penalty
            gp = gradient_penalty(D, follow_up, fake, device)
            
            d_loss = d_loss_real + d_loss_fake + 10 * gp
            
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # ============================================
            # Train Generator
            # ============================================
            if i % config.n_critic == 0:
                fake = G(x_real, target_grade)
                
                # Adversarial loss
                out_fake, _ = D(fake)
                g_loss_adv = -torch.mean(out_fake)
                
                # Reconstruction loss (L1 between fake and real follow-up)
                g_loss_rec = l1_loss(fake, follow_up)
                
                g_loss = config.lambda_adv * g_loss_adv + config.lambda_rec * g_loss_rec
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_rec_loss += g_loss_rec.item()
            
            epoch_d_loss += d_loss.item()
        
        # Logging
        avg_g_loss = epoch_g_loss / len(loader)
        avg_d_loss = epoch_d_loss / len(loader)
        avg_rec_loss = epoch_rec_loss / len(loader)
        
        elapsed = time.time() - start_time
        
        if (epoch + 1) % config.log_step == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{config.num_epochs}] | "
                  f"D_loss: {avg_d_loss:.4f} | "
                  f"G_loss: {avg_g_loss:.4f} | "
                  f"Rec_loss: {avg_rec_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")
        
        # Save samples
        if (epoch + 1) % config.save_step == 0 or epoch == config.num_epochs - 1:
            G.eval()
            with torch.no_grad():
                sample_fake = G(x_real, target_grade)
                samples = torch.cat([
                    denorm(baseline[:2]),
                    denorm(sample_fake[:2]),
                    denorm(follow_up[:2])
                ], dim=3)
                
                sample_path = os.path.join(config.sample_dir, f'epoch_{epoch+1}.jpg')
                save_image(samples, sample_path, nrow=1, padding=10)
                print(f"  ✓ Saved samples to {sample_path}")
            G.train()
            
            # Save model checkpoints
            g_path = os.path.join(config.save_dir, f'{epoch+1}-G.ckpt')
            d_path = os.path.join(config.save_dir, f'{epoch+1}-D.ckpt')
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"  ✓ Saved checkpoints: {g_path}")
    
    # Final save
    print(f"\n[4/4] Training complete!")
    print("="*70)
    
    final_g_path = os.path.join(config.save_dir, 'final-G.ckpt')
    final_d_path = os.path.join(config.save_dir, 'final-D.ckpt')
    torch.save(G.state_dict(), final_g_path)
    torch.save(D.state_dict(), final_d_path)
    
    # Also save with the standard name for inference compatibility
    standard_path = os.path.join(config.save_dir, '200000-G.ckpt')
    torch.save(G.state_dict(), standard_path)
    
    print(f"✓ Final Generator saved to: {final_g_path}")
    print(f"✓ Final Discriminator saved to: {final_d_path}")
    print(f"✓ Inference-compatible weights saved to: {standard_path}")
    print(f"\n📊 Total training time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"\n🎉 You can now run inference with these trained weights!")


if __name__ == '__main__':
    train()