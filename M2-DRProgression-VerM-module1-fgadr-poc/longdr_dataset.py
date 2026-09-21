"""
LongDRScreening Dataset Loader (Rotterdam Eye Hospital)
Loads longitudinal fundus image pairs

Structure:
- FundusImagesColor/eye_XXX/visit_N_image_M.png
- 140 eyes, 2 visits each, 4 images per visit
- Inter-visit pairs (visit_1 → visit_2) for longitudinal training
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import re


class LongDRScreeningDataset(Dataset):
    """
    Loads LongDRScreening inter-visit pairs for longitudinal GAN training.
    
    For each eye, pairs corresponding images from visit_1 to visit_2
    (e.g., visit_1_image_1 → visit_2_image_1).
    
    This gives us up to 4 pairs per eye × 140 eyes = 560 pairs.
    """
    
    SOURCE_NAME = "LongDRScreening"

    def __init__(self, dataset_dir, image_size=128, use_normalized=True, module1_cache_path=None,
                 registration_cache_path=None):
        """
        Args:
            dataset_dir: Path to LongDRScreening_20150209 folder
            image_size: Resize images to this size
            use_normalized: Use FundusImagesNormalized (recommended) or FundusImagesColor
            module1_cache_path: Optional path to a cache produced by
                module1/apply_to_progression_data.py ({image_id: {grade, mask, lbs}}), keyed
                as "<eye_folder>__<baseline_filename_stem>" to match that script's recursive
                directory walk. Same fallback behavior as FIREDataset -- see its docstring.
            registration_cache_path: Optional path to a cache produced by
                module1/train_registration.py ({(source, baseline_path): warped_followup_uint8}).
                Same fallback behavior as FIREDataset -- see its docstring.
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size

        self.module1_cache = None
        if module1_cache_path:
            self.module1_cache = torch.load(module1_cache_path, weights_only=False)
            print(f"✓ Loaded Module 1 cache: {len(self.module1_cache)} images ({module1_cache_path})")

        self.registration_cache = None
        if registration_cache_path:
            self.registration_cache = torch.load(registration_cache_path, weights_only=False)
            print(f"✓ Loaded registration cache: {len(self.registration_cache)} pairs ({registration_cache_path})")

        # Choose which folder to use
        folder_name = 'FundusImagesNormalized' if use_normalized else 'FundusImagesColor'
        self.images_dir = os.path.join(dataset_dir, folder_name)
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Could not find {folder_name} in {dataset_dir}")
        
        # Find all valid pairs
        self.pairs = self._find_pairs()
        
        print(f"✓ Found {len(self.pairs)} pairs in LongDRScreening ({folder_name})")
        
        # Image transformation with augmentation
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    def _find_pairs(self):
        """Find all inter-visit pairs across all eyes"""
        pairs = []
        
        # Get all eye folders
        eye_folders = sorted([d for d in os.listdir(self.images_dir) 
                             if d.startswith('eye_') and os.path.isdir(os.path.join(self.images_dir, d))])
        
        for eye_folder in eye_folders:
            eye_path = os.path.join(self.images_dir, eye_folder)
            files = os.listdir(eye_path)
            
            # Build dictionary of files by (visit, image_num)
            visit_images = {1: {}, 2: {}}
            
            for f in files:
                if not f.lower().endswith('.png'):
                    continue
                
                # Match patterns: visit_1_image_1.png, Visit_1_image_1.png
                match = re.match(r'[Vv]isit_(\d+)_image_(\d+)\.png', f)
                if match:
                    visit_num = int(match.group(1))
                    image_num = int(match.group(2))
                    
                    if visit_num in visit_images:
                        visit_images[visit_num][image_num] = os.path.join(eye_path, f)
            
            # Pair corresponding images from visit_1 → visit_2
            for img_num in visit_images[1].keys():
                if img_num in visit_images[2]:
                    pairs.append({
                        'eye': eye_folder,
                        'baseline': visit_images[1][img_num],
                        'follow_up': visit_images[2][img_num],
                        'image_num': img_num
                    })
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Returns dict with baseline, follow_up, mask, target_grade"""
        pair = self.pairs[idx]
        
        # Load images
        try:
            img1 = Image.open(pair['baseline']).convert('RGB')
            warped = self.registration_cache.get((self.SOURCE_NAME, pair['baseline'])) if self.registration_cache else None
            if warped is not None:
                # Pre-registered (baseline-aligned) follow-up image.
                img2 = Image.fromarray(np.transpose(warped, (1, 2, 0)))  # CHW uint8 -> HWC for PIL
            else:
                img2 = Image.open(pair['follow_up']).convert('RGB')
        except Exception as e:
            # If image fails to load, return next valid pair
            return self.__getitem__((idx + 1) % len(self.pairs))

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        # Look up real Module 1 output for the baseline image, if a cache was provided.
        # Falls back to the placeholder below otherwise -- see FIREDataset.__getitem__ for the
        # identical pattern.
        image_id = f"{pair['eye']}__{Path(pair['baseline']).stem}"
        module1_output = self.module1_cache.get(image_id) if self.module1_cache else None

        if module1_output is not None:
            mask_np = module1_output['mask']
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
            mask = torch.from_numpy(np.array(mask_img) > 0).float().unsqueeze(0)

            target_grade = torch.zeros(5)
            target_grade[int(module1_output['grade'])] = 1.0
        else:
            # RODREP doesn't have lesion masks
            # In real training, run Module 1 to generate masks
            mask = torch.zeros(1, self.image_size, self.image_size)

            # RODREP doesn't have severity stage labels
            # Default to Stage 2 for conditioning
            target_grade = torch.zeros(5)
            target_grade[2] = 1.0

        return {
            'baseline': img1,
            'follow_up': img2,
            'mask': mask,
            'target_grade': target_grade,
            'source': 'LongDRScreening',
            'eye_id': pair['eye']
        }


if __name__ == '__main__':
    # Test the loader
    print("="*70)
    print("Testing LongDRScreening Data Loader")
    print("="*70)
    
    dataset = LongDRScreeningDataset('./LongDRScreening_20150209', image_size=128)
    
    print(f"\nTotal pairs: {len(dataset)}")
    print(f"Sample pair info:")
    
    sample = dataset[0]
    print(f"  Baseline shape: {sample['baseline'].shape}")
    print(f"  Follow-up shape: {sample['follow_up'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Target grade: {sample['target_grade']}")
    print(f"  Eye ID: {sample['eye_id']}")
    
    print("\n✓ LongDRScreening loader working!")
    