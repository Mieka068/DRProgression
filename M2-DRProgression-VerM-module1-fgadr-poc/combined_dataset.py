"""
Combined FIRE + LongDRScreening + Tianjin Dataset Loader with Augmentation
Maximizes training data for the GAN

How FIRE and LongDRScreening actually flow through training, precisely: this file wraps
FIREDataset and LongDRScreeningDataset (each reading real longitudinal image pairs from their
own raw data directories -- no real DR grade available in either) and concatenates them with
TianjinLongitudinalDataset into one combined dataset that train_module2_poc.py trains on.
Neither FIRE nor LongDRScreening carries a real grade, so both fall back to a placeholder
Stage-2 one-hot target during training whenever no module1_cache_path entry is given for a
pair (see fire_dataset.py's __getitem__ / longdr_dataset.py's equivalent) -- meaning every
FIRE/LongDR training pair without a cache entry teaches the generator "make this image look
like Stage 2," regardless of what stage the pair might actually represent. Their contribution
is real image-pair volume and photographic diversity for the adversarial/reconstruction
losses, not genuine stage-transition supervision. Because of this, FIRE/LongDR pairs cannot be
scored against a real follow-up grade the way Tianjin's can (see evaluate_trajectory.py's
--self-consistency-dirs option for the metric this file's sources use instead).
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random

from fire_dataset import FIREDataset
from longdr_dataset import LongDRScreeningDataset
from tianjin_dataset import TianjinLongitudinalDataset


class AugmentedPairDataset(Dataset):
    """
    Wraps a paired dataset with synchronized augmentation.
    Same augmentation is applied to baseline and follow-up to preserve correspondence.
    """
    
    def __init__(self, base_dataset, image_size=128, augment=True):
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.augment = augment
    
    def __len__(self):
        # Multiply effective dataset size with augmentation
        return len(self.base_dataset) * (4 if self.augment else 1)
    
    def __getitem__(self, idx):
        # Wrap around to original dataset
        real_idx = idx % len(self.base_dataset)
        sample = self.base_dataset[real_idx]
        
        baseline = sample['baseline']
        follow_up = sample['follow_up']
        
        if self.augment:
            # Apply same random augmentation to both images
            baseline, follow_up = self._sync_augment(baseline, follow_up)
        
        sample['baseline'] = baseline
        sample['follow_up'] = follow_up

        # TianjinLongitudinalDataset samples carry three extra keys (grade_is_real,
        # patient_id, pair_quality) that FIRE/LongDR samples don't set. A DataLoader batch can
        # mix samples from different sources (ConcatDataset + shuffle), and
        # torch.utils.data.default_collate requires every sample dict in a batch to have the
        # same keys -- it iterates the *first* sample's keys and indexes every other sample by
        # them, so a missing key crashes collation as soon as a batch happens to start with a
        # Tianjin sample. Default them here so every sample this dataset yields has the same
        # shape regardless of source.
        sample.setdefault('grade_is_real', False)
        sample.setdefault('patient_id', sample.get('eye_id', 'n/a'))
        sample.setdefault('pair_quality', -1.0)
        sample.setdefault('grade_is_eye_specific', False)

        return sample
    
    def _sync_augment(self, img1, img2):
        """
        Apply same augmentation to both images to preserve pair correspondence.
        
        Uses MEDICALLY-APPROPRIATE augmentations only:
        - Horizontal flip (left-eye ↔ right-eye is anatomically valid)
        - Small rotations (±15°)
        - Mild brightness/contrast (camera variability)
        
        AVOIDS:
        - Vertical flips (changes optic nerve anatomy)
        - 90/180° rotations (unnatural retinal orientation)
        - Heavy color jittering (could distort lesion appearance)
        """
        
        # Random horizontal flip (50% chance) - MEDICALLY VALID
        # Equivalent to swapping left/right eye perspective
        if random.random() > 0.5:
            img1 = torch.flip(img1, dims=[2])
            img2 = torch.flip(img2, dims=[2])
        
        # Small random rotation (±15 degrees) - SAFE
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            # Rotate using affine transformation
            from torchvision.transforms.functional import rotate
            img1 = rotate(img1, angle)
            img2 = rotate(img2, angle)
        
        # Mild brightness variation (mimics camera differences) - SAFE
        if random.random() > 0.5:
            brightness = 0.9 + random.random() * 0.2  # 0.9 to 1.1 (gentler)
            img1 = (img1 * brightness).clamp(-1, 1)
            img2 = (img2 * brightness).clamp(-1, 1)
        
        return img1, img2


def get_combined_loader(fire_dir='./FIRE_dataset',
                        longdr_dir='./LongDRScreening_20150209',
                        tianjin_dir=None,
                        image_size=128,
                        batch_size=4,
                        augment=True,
                        num_workers=0,
                        fire_module1_cache_path=None,
                        longdr_module1_cache_path=None,
                        tianjin_module1_cache_path=None,
                        tianjin_min_pair_quality=None,
                        registration_cache_path=None):
    """
    Get a DataLoader combining FIRE + LongDRScreening + (optionally) Tianjin, with augmentation.

    Args:
        fire_module1_cache_path: Optional module1/apply_to_progression_data.py cache
            (see FIREDataset docstring) -- real grade/mask conditioning instead of the
            placeholder empty-mask/Stage-2 default when given.
        longdr_module1_cache_path: Same, for LongDRScreeningDataset.
        tianjin_dir: Optional path to the extracted Tianjin (usama10/retinal-dr-longitudinal)
            dataset. None (the default) skips Tianjin entirely, so existing FIRE+LongDR-only
            callers are unaffected.
        tianjin_module1_cache_path: Same fallback contract as the other two, for
            TianjinLongitudinalDataset -- only used for the lesion mask, since Tianjin's real
            clinical grade always wins over a Module 1 prediction (see its docstring).
        tianjin_min_pair_quality: Optional float; drop Tianjin pairs below this
            corrected_manifest.csv registration-quality threshold. None keeps every pair.
        registration_cache_path: Optional path to a single cache produced by
            module1/train_registration.py, shared across all three sources (it's keyed by
            (source, baseline_path), see that script's docstring) -- substitutes a
            pre-registered, baseline-aligned follow-up image in place of the raw one wherever
            an entry exists. None (the default) uses raw follow-up images everywhere.

    Returns:
        DataLoader yielding batches from all loaded datasets
    """
    from torch.utils.data import DataLoader

    datasets = []

    # Try to load FIRE
    try:
        fire = FIREDataset(fire_dir, image_size=image_size, category='A',
                            module1_cache_path=fire_module1_cache_path,
                            registration_cache_path=registration_cache_path)
        if len(fire) > 0:
            datasets.append(fire)
            print(f"✓ FIRE: {len(fire)} pairs")
    except Exception as e:
        print(f"⚠ Could not load FIRE: {e}")

    # Try to load LongDRScreening
    try:
        longdr = LongDRScreeningDataset(longdr_dir, image_size=image_size, use_normalized=True,
                                         module1_cache_path=longdr_module1_cache_path,
                                         registration_cache_path=registration_cache_path)
        if len(longdr) > 0:
            datasets.append(longdr)
            print(f"✓ LongDRScreening: {len(longdr)} pairs")
    except Exception as e:
        print(f"⚠ Could not load LongDRScreening: {e}")

    # Try to load Tianjin (only if a directory was given -- keeps this backward compatible
    # with callers that don't know about Tianjin yet)
    if tianjin_dir:
        try:
            tianjin = TianjinLongitudinalDataset(
                tianjin_dir, image_size=image_size,
                min_pair_quality=tianjin_min_pair_quality,
                module1_cache_path=tianjin_module1_cache_path,
                registration_cache_path=registration_cache_path,
            )
            if len(tianjin) > 0:
                datasets.append(tianjin)
                print(f"✓ Tianjin: {len(tianjin)} pairs")
        except Exception as e:
            print(f"⚠ Could not load Tianjin: {e}")

    if not datasets:
        raise RuntimeError("No datasets loaded!")
    
    # Combine datasets
    combined = ConcatDataset(datasets)
    print(f"✓ Combined: {len(combined)} pairs")
    
    # Wrap with augmentation
    augmented = AugmentedPairDataset(combined, image_size=image_size, augment=augment)
    print(f"✓ Augmented: {len(augmented)} effective samples")
    
    # Create loader
    loader = DataLoader(
        augmented,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    return loader, len(combined)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fire-dir', default='./FIRE_dataset')
    parser.add_argument('--longdr-dir', default='./LongDRScreening_20150209')
    parser.add_argument('--tianjin-dir', default=None, help='Omit to test FIRE+LongDR only')
    args = parser.parse_args()

    print("="*70)
    print("Testing Combined FIRE + LongDRScreening + Tianjin Loader")
    print("="*70)

    loader, num_pairs = get_combined_loader(
        fire_dir=args.fire_dir,
        longdr_dir=args.longdr_dir,
        tianjin_dir=args.tianjin_dir,
        image_size=128,
        batch_size=4,
        augment=True
    )
    
    print(f"\nReal pairs: {num_pairs}")
    print(f"Batches per epoch: {len(loader)}")
    
    for batch in loader:
        print(f"\nBatch shapes:")
        print(f"  Baseline: {batch['baseline'].shape}")
        print(f"  Follow-up: {batch['follow_up'].shape}")
        print(f"  Mask: {batch['mask'].shape}")
        print(f"  Target Grade: {batch['target_grade'].shape}")
        print(f"  Sources: {batch['source']}")
        break
    
    print("\n✓ Combined loader working!")
    