"""
FIRE Dataset Loader
Loads paired fundus images from FIRE dataset for GAN training
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


class FIREDataset(Dataset):
    """
    Loads FIRE image pairs for longitudinal training.
    
    FIRE structure:
    - Images/A01_1.jpg (Time 1)
    - Images/A01_2.jpg (Time 2)  
    - Categories: A (anatomical changes - DR progression), S (super-resolution), P (panorama)
    
    For our GAN, we want Category A pairs (disease progression).
    """
    
    SOURCE_NAME = "FIRE"

    def __init__(self, fire_dir, image_size=128, category='A', module1_cache_path=None,
                 registration_cache_path=None):
        """
        Args:
            fire_dir: Path to FIRE_dataset folder
            image_size: Resize images to this size
            category: 'A' for anatomical changes (best for DR progression)
            module1_cache_path: Optional path to a cache produced by
                module1/apply_to_progression_data.py ({image_id: {grade, mask, lbs}}).
                When given and an entry exists for a pair's baseline image, its real
                mask/grade are used instead of the placeholder empty-mask/Stage-2 default
                below. Falls back to the placeholder for any image not in the cache (or if
                no cache_path is given at all) -- existing behavior is unchanged either way.
            registration_cache_path: Optional path to a cache produced by
                module1/train_registration.py ({(source, baseline_path): warped_followup_uint8}).
                When given and an entry exists for this pair's baseline path, the pre-warped,
                baseline-aligned follow-up image is used instead of the raw one. Falls back to
                the raw follow-up image for any pair not in the cache (or if no cache_path is
                given at all).
        """
        self.fire_dir = fire_dir
        self.image_size = image_size
        self.category = category

        self.module1_cache = None
        if module1_cache_path:
            self.module1_cache = torch.load(module1_cache_path, weights_only=False)
            print(f"✓ Loaded Module 1 cache: {len(self.module1_cache)} images ({module1_cache_path})")

        self.registration_cache = None
        if registration_cache_path:
            self.registration_cache = torch.load(registration_cache_path, weights_only=False)
            print(f"✓ Loaded registration cache: {len(self.registration_cache)} pairs ({registration_cache_path})")

        # Find all image pairs in the specified category
        self.pairs = self._find_pairs()

        print(f"✓ Found {len(self.pairs)} pairs in FIRE Category {category}")
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    def _find_pairs(self):
        """Find all paired images in the dataset"""
        pairs = []
        
        # Search for images directory
        images_dir = None
        for root, dirs, files in os.walk(self.fire_dir):
            if 'Images' in dirs:
                images_dir = os.path.join(root, 'Images')
                break
        
        if not images_dir:
            # Try alternative: just look for image files
            for root, dirs, files in os.walk(self.fire_dir):
                jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
                if len(jpg_files) > 10:  # Likely image directory
                    images_dir = root
                    break
        
        if not images_dir:
            raise FileNotFoundError(f"Could not find Images folder in {self.fire_dir}")
        
        print(f"  Searching: {images_dir}")
        
        # FIRE naming: A01_1.jpg and A01_2.jpg are pairs
        all_files = sorted(os.listdir(images_dir))
        
        # Group files by their base name (without _1, _2 suffix)
        pair_dict = {}
        for f in all_files:
            if not f.lower().endswith(('.jpg', '.png')):
                continue
            
            # Filter by category if specified
            if self.category and not f.startswith(self.category):
                continue
            
            # Get base name (e.g., A01 from A01_1.jpg)
            base = f.rsplit('_', 1)[0] if '_' in f else f.rsplit('.', 1)[0]
            
            if base not in pair_dict:
                pair_dict[base] = []
            pair_dict[base].append(os.path.join(images_dir, f))
        
        # Keep only complete pairs
        for base, files in pair_dict.items():
            if len(files) == 2:
                pairs.append(sorted(files))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Returns (baseline_image, follow_up_image, dummy_mask, target_grade)"""
        img1_path, img2_path = self.pairs[idx]
        
        # Load images
        img1 = Image.open(img1_path).convert('RGB')

        # Use the pre-registered (baseline-aligned) follow-up image if a registration cache
        # was supplied and has an entry for this pair, else fall back to the raw follow-up image.
        warped = self.registration_cache.get((self.SOURCE_NAME, img1_path)) if self.registration_cache else None
        if warped is not None:
            img2 = Image.fromarray(np.transpose(warped, (1, 2, 0)))  # CHW uint8 -> HWC for PIL
        else:
            img2 = Image.open(img2_path).convert('RGB')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        # Look up real Module 1 output (grade + lesion mask) for the baseline image, if a
        # cache was provided and has an entry for it. Falls back to the placeholder below
        # otherwise -- FIRE has no ground-truth grade/mask of its own either way.
        image_id = Path(img1_path).stem
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
            # FIRE doesn't have lesion masks, so use empty mask
            # In real training, you'd run Module 1 to generate masks
            mask = torch.zeros(1, self.image_size, self.image_size)

            # FIRE doesn't have severity labels, so use random grade for conditioning
            # In real training, Module 1 would assign labels
            target_grade = torch.zeros(5)  # 5 stages
            target_grade[2] = 1.0  # Default to Stage 2 (Moderate)

        return {
            'baseline': img1,
            'follow_up': img2,
            'mask': mask,
            'target_grade': target_grade,
            # Matches LongDRScreeningDataset's return keys so ConcatDataset batches that mix
            # samples from both datasets collate correctly (default_collate requires every
            # sample dict in a batch to have the same keys) -- see combined_dataset.py.
            'source': 'FIRE',
            'eye_id': 'n/a'
        }


def get_fire_loader(fire_dir, batch_size=4, image_size=128, num_workers=2, module1_cache_path=None):
    """Get DataLoader for FIRE dataset"""
    dataset = FIREDataset(fire_dir, image_size=image_size, category='A', module1_cache_path=module1_cache_path)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    return loader


if __name__ == '__main__':
    # Test the loader
    print("Testing FIRE Data Loader")
    print("="*70)
    
    loader = get_fire_loader('./FIRE_dataset', batch_size=2)
    
    for batch in loader:
        print(f"\nBatch shapes:")
        print(f"  Baseline: {batch['baseline'].shape}")
        print(f"  Follow-up: {batch['follow_up'].shape}")
        print(f"  Mask: {batch['mask'].shape}")
        print(f"  Target Grade: {batch['target_grade'].shape}")
        break
    
    print("\n✓ Data loader working!")