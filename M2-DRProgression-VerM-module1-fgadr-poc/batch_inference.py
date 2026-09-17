"""
Batch Inference Script for Module 2 (GAN Synthesis)
Runs inference on multiple images from IDRiD dataset
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import os
from pathlib import Path
import sys

# Add DRForestGAN to path
sys.path.insert(0, './DRForestGAN-v2')

from base_model import Generator


def load_image_and_mask(image_path, mask_path, image_size=128):
    """Load fundus image and lesion mask"""
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
    """Denormalize from [-1, 1] to [0, 1]"""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def run_batch_inference(image_dir, mask_dir, output_dir, num_images=5, target_grade=2):
    """
    Run inference on multiple images
    
    Args:
        image_dir: Directory containing fundus images
        mask_dir: Directory containing lesion masks
        output_dir: Where to save results
        num_images: How many images to process
        target_grade: Target severity stage (0-4)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("BATCH INFERENCE - Module 2 (GAN Synthesis)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Processing {num_images} images...")
    print(f"Target Grade: {target_grade}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    print("[Loading Generator]")
    generator = Generator(conv_dim=64, c_dim=5, repeat_num=6)
    generator.to(device)
    generator.eval()
    
    # Try to load weights
    weights_path = './DRForestGAN-v2/stargan/models/200000-G.ckpt'
    if os.path.exists(weights_path):
        print(f"✓ Loading weights from: {weights_path}")
        generator.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"⚠ Weights not found. Using random initialization.")
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])[:num_images]
    
    print(f"\n{'='*70}")
    print(f"Found {len(image_files)} images to process")
    print(f"{'='*70}\n")
    
    results = []
    
    # Process each image
    for idx, img_filename in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_filename}")
        print("-" * 70)
        
        # Construct mask filename (assuming naming like IDRiD_01.jpg → IDRiD_01_EX.tif)
        base_name = Path(img_filename).stem
        mask_filename = f"{base_name}_EX.tif"
        
        image_path = os.path.join(image_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        
        # Check if files exist
        if not os.path.exists(image_path):
            print(f"  ✗ Image not found: {image_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"  ✗ Mask not found: {mask_path}")
            continue
        
        try:
            # Load image and mask
            image_tensor, mask_tensor = load_image_and_mask(image_path, mask_path)
            image_tensor = image_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            
            print(f"  ✓ Image loaded: {image_tensor.shape}")
            print(f"  ✓ Mask loaded: {mask_tensor.shape}")
            
            # Prepare input
            x = torch.cat([image_tensor, mask_tensor], dim=1)  # [1, 4, 128, 128]
            
            # Create condition (one-hot for target grade)
            c_target = torch.zeros(1, 5).to(device)
            c_target[0, target_grade] = 1.0
            
            # Run inference
            with torch.no_grad():
                synthesized = generator(x, c_target)
            
            print(f"  ✓ Synthesis complete: {synthesized.shape}")
            
            # Save original and synthesized side-by-side
            result_image = torch.cat([denorm(image_tensor), denorm(synthesized)], dim=3)
            
            result_path = os.path.join(output_dir, f'{idx:02d}_{base_name}_comparison.jpg')
            save_image(result_image, result_path, nrow=1, padding=10)
            print(f"  ✓ Saved: {result_path}")
            
            # Save just synthesized
            synth_path = os.path.join(output_dir, f'{idx:02d}_{base_name}_synthesized.jpg')
            save_image(denorm(synthesized), synth_path)
            print(f"  ✓ Saved: {synth_path}")
            
            results.append({
                'image': img_filename,
                'status': 'SUCCESS',
                'input_shape': str(x.shape),
                'output_shape': str(synthesized.shape)
            })
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                'image': img_filename,
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'SUCCESS')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'FAILED')}")
    print(f"\nResults saved to: {output_dir}\n")
    
    return results


if __name__ == '__main__':
    # Configure paths
    IMAGE_DIR = './IDRiD_dataset/A. Segmentation/1. Original Images/a. Training Set/'
    MASK_DIR = './IDRiD_dataset/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates/'
    OUTPUT_DIR = './DRForestGAN-v2/stargan/results/batch_inference/'
    
    # Run batch inference
    results = run_batch_inference(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        output_dir=OUTPUT_DIR,
        num_images=5,
        target_grade=2  # Synthesize to Stage 2 (Moderate DR)
    )
    
    # Print results table
    print("\nDetailed Results:")
    print("-" * 70)
    for r in results:
        print(f"  {r['image']:<30} {r['status']:<10}")
