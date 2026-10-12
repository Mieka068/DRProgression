"""
Cross-sectional (non-longitudinal) DR dataset loader for Module 2 pretraining.

Why this exists: Module 2's longitudinal pairs (FIRE + LongDR + Tianjin) are small, and only
Tianjin carries real grades. Cross-sectional sets like EyePACS (~35k graded images) have no
baseline -> follow-up pairs, but they have MANY images per ICDR stage. That is enough to learn
(a) what each stage looks like (StarGAN-style unpaired stage translation, see
train_module2_nonlong_pretrain.py) and (b) which biomarkers characterise each stage and change
between adjacent stages (see stage_biomarker_profiles.py). The pretrained Generator is then
fine-tuned on the real longitudinal pairs with train_module2_poc.py --init-generator.

Supported label layouts (pick with --preset, or pass --labels-csv/--image-col/--label-col):
    eyepacs:  Kaggle "Diabetic Retinopathy Detection"
              trainLabels.csv (image, level), images in train/<image>.jpeg
    aptos:    Kaggle "APTOS 2019 Blindness Detection"
              train.csv (id_code, diagnosis), images in train_images/<id_code>.png
    generic:  any CSV with an image-id column and an ICDR 0-4 label column

All three use the ICDR 0-4 scale already, the same scale as target_grade elsewhere in this
repo, so no grade remapping is applied. Rows with a label outside 0-4 are dropped.

Module 1 cache key scheme: module1/apply_to_progression_data.py keys its cache by the image
path relative to --images-dir with the extension stripped (and os.sep -> "__"). Point
--images-dir at the same folder this loader's `images_dir` points at, and EyePACS keys come out
as "10_left", matching the CSV's image column directly.

Each sample is a single image (no follow-up):
    image        [3,H,W] in [-1,1] (same normalisation as the longitudinal loaders)
    mask         [1,H,W] Module 1 lesion mask, or zeros when no cache entry exists
    grade        int ICDR stage (real label from the CSV)
    grade_onehot [c_dim] one-hot of `grade`
    biomarkers   [N_BIOMARKERS] transformed biomarker vector (see stage_biomarker_profiles),
                 0 where unavailable
    bio_mask     [N_BIOMARKERS] bool -- False for features with no Module 1 lesion_stats
                 (image not in the cache, or that lesion's segmenter wasn't trained)
    image_id     str

License note: EyePACS (Kaggle) and APTOS are competition data under their own terms --
research use only, do not redistribute the raw images.
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms.functional import rotate

from stage_biomarker_profiles import N_BIOMARKERS, biomarker_vector

NUM_STAGES = 5

PRESETS = {
    "eyepacs": {"labels_csv": "trainLabels.csv", "images_subdir": "train",
                "image_col": "image", "label_col": "level", "ext": ".jpeg"},
    "aptos": {"labels_csv": "train.csv", "images_subdir": "train_images",
              "image_col": "id_code", "label_col": "diagnosis", "ext": ".png"},
}


def load_labels(labels_csv, image_col, label_col):
    """Returns a (image_id, grade) frame with grades restricted to ICDR 0-4."""
    df = pd.read_csv(labels_csv)
    for col in (image_col, label_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not in {labels_csv}. Actual columns: {list(df.columns)}")
    df = df[[image_col, label_col]].rename(columns={image_col: "image_id", label_col: "grade"})
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    before = len(df)
    df = df[df["grade"].between(0, NUM_STAGES - 1)].copy()
    df["grade"] = df["grade"].astype(int)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with a missing/out-of-range grade")
    return df.reset_index(drop=True)


class CrossSectionalDRDataset(Dataset):
    """Single graded fundus images (no pairs). See module docstring for the sample dict."""

    def __init__(self, dataset_dir, preset="eyepacs", labels_csv=None, images_dir=None,
                 image_col=None, label_col=None, ext=None, image_size=128, augment=True,
                 module1_cache_path=None, max_per_grade=None, seed=0):
        """
        Args:
            dataset_dir: root folder of the dataset (relative labels_csv/images_dir resolve
                against it).
            preset: 'eyepacs', 'aptos', or 'generic' (then labels_csv/image_col/label_col/
                images_dir/ext are required). Explicit args override preset values.
            module1_cache_path: optional module1/apply_to_progression_data.py cache over this
                dataset's images (ideally built with --mask-size 256). Supplies the lesion mask
                channel and the per-lesion biomarkers; images missing from it get a zero mask
                and an all-False bio_mask. The CSV grade is always used for conditioning -- Module 1's
                predicted grade is ignored here because the label is real.
            max_per_grade: optional cap on images kept per grade (random, seeded) -- EyePACS is
                ~73% grade 0, so capping is a cheap way to cut epoch time without losing the
                rare stages.
        """
        cfg = dict(PRESETS.get(preset, {}))
        for key, val in (("labels_csv", labels_csv), ("images_subdir", images_dir),
                         ("image_col", image_col), ("label_col", label_col), ("ext", ext)):
            if val is not None:
                cfg[key] = val
        missing = [k for k in ("labels_csv", "images_subdir", "image_col", "label_col") if k not in cfg]
        if missing:
            raise ValueError(f"preset '{preset}' needs explicit {missing}")

        self.images_dir = os.path.join(dataset_dir, cfg["images_subdir"])
        self.ext = cfg.get("ext", "")
        self.image_size = image_size
        self.augment = augment

        df = load_labels(os.path.join(dataset_dir, cfg["labels_csv"]), cfg["image_col"], cfg["label_col"])
        df["path"] = df["image_id"].map(self._resolve_path)
        n_missing = int(df["path"].isna().sum())
        if n_missing:
            print(f"  ⚠ {n_missing} labelled images not found under {self.images_dir} -- skipped")
        df = df[df["path"].notna()]
        if max_per_grade:
            df = (df.groupby("grade", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), max_per_grade), random_state=seed)))
        self.df = df.reset_index(drop=True)

        self.module1_cache = None
        if module1_cache_path:
            self.module1_cache = torch.load(module1_cache_path, weights_only=False)
            print(f"✓ Loaded Module 1 cache: {len(self.module1_cache)} images ({module1_cache_path})")

        counts = self.df["grade"].value_counts().reindex(range(NUM_STAGES), fill_value=0)
        print(f"✓ Cross-sectional ({preset}): {len(self.df)} images, per grade {counts.tolist()}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def _resolve_path(self, image_id):
        # Labels CSVs sometimes include the extension, sometimes not.
        for name in (image_id + self.ext, image_id):
            p = os.path.join(self.images_dir, name)
            if os.path.isfile(p):
                return p
        return None

    @property
    def grades(self):
        return self.df["grade"].to_numpy()

    def __len__(self):
        return len(self.df)

    def _augment(self, img, mask):
        """Same medically-appropriate set as combined_dataset.AugmentedPairDataset: h-flip,
        small rotation, mild brightness -- applied jointly to image and mask, brightness to the
        image only."""
        if random.random() > 0.5:
            img, mask = torch.flip(img, dims=[2]), torch.flip(mask, dims=[2])
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img, mask = rotate(img, angle), rotate(mask, angle)
        if random.random() > 0.5:
            img = (img * (0.9 + random.random() * 0.2)).clamp(-1, 1)
        return img, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = self.transform(Image.open(row["path"]).convert("RGB"))
        except Exception:
            return self.__getitem__((idx + 1) % len(self.df))

        entry = self.module1_cache.get(row["image_id"]) if self.module1_cache else None
        if entry is not None:
            mask_img = Image.fromarray((entry["mask"] * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask_img) > 0).float().unsqueeze(0)
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        bio = biomarker_vector(entry) if entry is not None else None
        if bio is None:
            bio = np.full(N_BIOMARKERS, np.nan, dtype=np.float32)
        bio_mask = torch.from_numpy(np.isfinite(bio))
        bio = torch.from_numpy(np.nan_to_num(bio, nan=0.0)).float()

        if self.augment:
            img, mask = self._augment(img, mask)

        grade = int(row["grade"])
        onehot = torch.zeros(NUM_STAGES)
        onehot[grade] = 1.0
        return {
            "image": img,
            "mask": mask,
            "grade": grade,
            "grade_onehot": onehot,
            "biomarkers": bio,
            "bio_mask": bio_mask,
            "image_id": row["image_id"],
        }


def get_crosssectional_loaders(dataset_dir, preset="eyepacs", batch_size=16, image_size=128,
                               module1_cache_path=None, val_fraction=0.1, max_per_grade=None,
                               balance_grades=True, samples_per_epoch=None, num_workers=0,
                               seed=0, **dataset_kwargs):
    """
    Stratified train/val split + loaders. With balance_grades=True the train loader samples
    each grade equally often (WeightedRandomSampler), so the rare severe/PDR stages are seen as
    often as stage 0 -- otherwise the Generator mostly learns "stage 0" on EyePACS.

    Returns (train_loader, val_loader, dataset).
    """
    train_ds = CrossSectionalDRDataset(dataset_dir, preset=preset, image_size=image_size, augment=True,
                                       module1_cache_path=module1_cache_path,
                                       max_per_grade=max_per_grade, seed=seed, **dataset_kwargs)
    # Separate non-augmented view over the same rows for validation (shares the cache object).
    val_ds = CrossSectionalDRDataset.__new__(CrossSectionalDRDataset)
    val_ds.__dict__.update(train_ds.__dict__)
    val_ds.augment = False

    rng = np.random.RandomState(seed)
    grades = train_ds.grades
    train_idx, val_idx = [], []
    for g in range(NUM_STAGES):
        idx = np.where(grades == g)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    if balance_grades:
        train_grades = grades[train_idx]
        counts = np.bincount(train_grades, minlength=NUM_STAGES).astype(float)
        weights = 1.0 / counts[train_grades]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=samples_per_epoch or len(train_idx),
                                        replacement=True)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    print(f"✓ Split: {len(train_idx)} train / {len(val_idx)} val "
          f"({'grade-balanced' if balance_grades else 'natural'} sampling)")
    return train_loader, val_loader, train_ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--preset", default="eyepacs", choices=["eyepacs", "aptos", "generic"])
    parser.add_argument("--module1-cache", default=None)
    args = parser.parse_args()

    train_loader, val_loader, _ = get_crosssectional_loaders(
        args.dataset_dir, preset=args.preset, batch_size=4, module1_cache_path=args.module1_cache)
    batch = next(iter(train_loader))
    print(f"  image {tuple(batch['image'].shape)} mask {tuple(batch['mask'].shape)} "
          f"grades {batch['grade'].tolist()} bio features available {int(batch['bio_mask'].sum())}")
    print("✓ Cross-sectional loader working!")
