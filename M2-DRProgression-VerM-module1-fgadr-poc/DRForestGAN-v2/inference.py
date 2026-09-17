"""
Inference script for DRForecastGAN
Loads a fundus image + lesion mask and synthesizes next disease stage
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import os
from base_model import Generator


def load_image_and_mask(image_path, mask_path, image_size=128):
    """
    Load fundus image and lesion mask.
    
    Args:
        image_path: path to fundus image
        mask_path: path to lesion mask (binary)
        image_size: resize to this size
    
    Returns:
        image_tensor: [1, 3, H, W] normalized image
        mask_tensor: [1, 1, H, W] binary mask
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_size, image_size), Image.BILINEAR)
    
    # Load mask
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    
    # Normalize image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    
    # Binarize mask
    mask_array = np.array(mask) > 127
    mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    return image_tensor, mask_tensor


def denorm(x):
    """Denormalize image from [-1, 1] to [0, 1]"""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def run_inference(config):
    """
    Run inference: load image+mask, synthesize next disease stage
    
    Args:
        config: configuration object with image_path, mask_path, model weights, etc.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("DRForecastGAN - Inference Mode")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(config.result_dir, exist_ok=True)

    # --- Step 1: Load image and mask ---
    print(f"\n[1/4] Loading image and mask...")
    print(f"  Image: {config.image_path}")
    print(f"  Mask:  {config.mask_path}")
    
    try:
        image_tensor, mask_tensor = load_image_and_mask(
            config.image_path, 
            config.mask_path, 
            image_size=config.image_size
        )
        print(f"  ✓ Image shape: {image_tensor.shape}")
        print(f"  ✓ Mask shape: {mask_tensor.shape}")
    except Exception as e:
        print(f"  ✗ Error loading image/mask: {e}")
        return
    
    image_tensor = image_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    
    # --- Step 2: Load generator ---
    print(f"\n[2/4] Loading generator...")
    
    generator = Generator(
        conv_dim=config.g_conv_dim,
        c_dim=config.c_dim,
        repeat_num=config.g_repeat_num
    )
    generator.to(device)
    generator.eval()
    
    # Try to load pretrained weights
    model_save_dir = config.model_save_dir
    test_iters = config.test_iters
    G_path = os.path.join(model_save_dir, f'{test_iters}-G.ckpt')
    
    if os.path.exists(G_path):
        print(f"  Loading weights from: {G_path}")
        try:
            generator.load_state_dict(torch.load(G_path, map_location=device))
            print(f"  ✓ Weights loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading weights: {e}")
            print(f"  ⚠ Using random initialization")
    else:
        print(f"  ⚠ Weights not found at {G_path}")
        print(f"  ⚠ Using random initialization")
        print(f"  💡 Tip: Train the model first or provide trained weights")
    
    # --- Step 3: Prepare input and target condition ---
   
    print(f"\n[3/4] Preparing inference inputs...")
    

    # Concatenate image + mask
    x = torch.cat([image_tensor, mask_tensor], dim=1)  # [1, 4, H, W]
    print(f"  ✓ Input (image + mask): {x.shape}")
 

    # Use just image (no mask for now)
    # x = image_tensor  # [1, 3, H, W]
    
    # Create target grade condition (one-hot)
    current_grade = 1  # Assume input is at stage 1
    target_grade = config.target_grade
    
    c_target = torch.zeros(1, config.c_dim).to(device)
    c_target[0, target_grade] = 1.0
    print(f"  ✓ Current grade: {current_grade}")
    print(f"  ✓ Target grade: {target_grade}")
    print(f"  ✓ Target condition shape: {c_target.shape}")
    
    # --- Step 4: Run inference ---

    print(f"\n[4/4] Running inference...")
    
    with torch.no_grad():
        try:
            synthesized = generator(x, c_target)
            print(f"  ✓ Synthesis complete")
            print(f"  ✓ Output shape: {synthesized.shape}")
        except Exception as e:
            print(f"  ✗ Error during synthesis: {e}")
            return
    
    # --- Step 5: Save results ---

    print(f"\n[5/5] Saving results...")
    
    # Concatenate original and synthesized for comparison
    result_image = torch.cat([denorm(image_tensor), denorm(synthesized)], dim=3)
    
    result_path = os.path.join(config.result_dir, 'inference_result.jpg')
    save_image(result_image, result_path, nrow=1, padding=10)
    print(f"  ✓ Result saved to: {result_path}")
    
    # Also save just the synthesized image
    synth_path = os.path.join(config.result_dir, 'synthesized_stage_{}.jpg'.format(target_grade))
    save_image(denorm(synthesized), synth_path)
    print(f"  ✓ Synthesized image saved to: {synth_path}")
    
    # Save the mask for reference
    mask_path_out = os.path.join(config.result_dir, 'lesion_mask.jpg')
    save_image(mask_tensor, mask_path_out)
    print(f"  ✓ Mask saved to: {mask_path_out}")
    
    print(f"\n{'='*60}")
    print("✓ Inference complete!")
    print(f"{'='*60}\n")