"""
Module 3 -- Tianjin survival dataset.

Builds (baseline_image, LBS stratum, progression event, follow-up time) samples from the same
Tianjin corrected_manifest.csv + Organized_Data of Patients.xlsx that Module 2's
tianjin_dataset.py reads, plus the module1_outputs_tianjin.pt cache built by
notebooks/05_tianjin_data_prep_colab.ipynb. This is a SEPARATE loader, not a reuse of
TianjinLongitudinalDataset -- Module 3 needs the "2-Year Follow-up" sheet's grade too (to
compute the progression event label), which Module 2 never touches (Module 2 only conditions
on the BASELINE grade; see tianjin_dataset.py's own docstring for why).

Per docs/DECISIONS.md's module-independence rule, Module 3 takes the REAL baseline fundus
image directly, in parallel with Module 2 -- never Module 2's synthesized output.

Per docs/ROADMAP.md's "Consequence of the Tianjin timing structure" note: every Tianjin
patient's follow-up interval is a FIXED 2 years, not a variable one, and we only observe
whether progression happened by that single checkpoint -- never the exact progression date
within it. This is "current status" / case-1 interval-censored data, not exact-event-time
survival data. `time_to_followup` is still threaded through as a per-sample field (not
hardcoded into the loss) so this interface doesn't need to change if a future source
(DCCT/EDIC, Moorfields) supplies real variable inter-visit timing.

Unlike tianjin_dataset.py's placeholder-tolerant grade fallback (Module 2's conditioning
degrades gracefully to a Stage-2 default when a grade is missing), Module 3 needs a REAL
progression label to train on at all -- a row with an excluded/missing baseline or follow-up
grade is dropped here, not defaulted. This is a real, intentional divergence from Module 2's
loader, not an inconsistency to "fix" back into agreement with it.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_MODULE1_DIR = os.path.join(_REPO_ROOT, "module1")
for _p in (_REPO_ROOT, _MODULE1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compute_lbs import stratify_lbs_by_grade  # noqa: E402
from tianjin_dataset import (  # noqa: E402
    EXCLUDED_GRADES,
    _find_column,
    _module1_image_id_candidates,
    _normalize_colname,
    tianjin_grade_to_icdr,
)

# ImageNet normalization stats -- correct choice here (unlike the [-1,1] range Module 2's
# loaders use) because this feeds an ImageNet-pretrained EfficientNet-B4 backbone, not a GAN.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STRATUM_TO_IDX = {"low": 0, "medium": 1, "high": 2}


class TianjinSurvivalDataset(Dataset):
    """
    One sample per corrected_manifest.csv row (i.e. per baseline<->follow-up eye pair) whose
    patient has BOTH a usable baseline grade and a usable follow-up grade. Multiple rows can
    belong to the same patient (e.g. both eyes) -- use patient_level_split() below rather than
    a plain random split to avoid leaking a patient across train/val.
    """

    def __init__(self, dataset_dir, module1_cache_path, image_size=380, fixed_followup_years=2.0):
        """
        Args:
            dataset_dir: path to the extracted usama10/retinal-dr-longitudinal dataset (same
                layout tianjin_dataset.py expects).
            module1_cache_path: REQUIRED (unlike Module 2's optional cache) -- path to the
                module1/apply_to_progression_data.py cache built over Tianjin's baseline
                images (notebooks/05_tianjin_data_prep_colab.ipynb). Module 3's LBS input
                feature comes entirely from this cache's precomputed 'lbs' field; there's no
                placeholder fallback because an LBS stratum is the model's only non-image
                input, not an optional conditioning channel.
            image_size: resize images to this size. Default 380 matches EfficientNet-B4's
                standard ImageNet input resolution.
            fixed_followup_years: the observation time for every sample. Tianjin's follow-up
                interval is fixed at 2 years for the whole cohort (see module docstring) --
                exposed as a parameter, not hardcoded, so a future variable-interval source
                can reuse this same class/interface.
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size

        if not module1_cache_path:
            raise ValueError(
                "module1_cache_path is required for Module 3 -- its LBS stratum input has no "
                "placeholder fallback, unlike Module 2's grade/mask conditioning."
            )
        cache = torch.load(module1_cache_path, weights_only=False)
        print(f"✓ Loaded Module 1 cache: {len(cache)} images ({module1_cache_path})")

        manifest_path = os.path.join(dataset_dir, "corrected_manifest.csv")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Could not find corrected_manifest.csv in {dataset_dir}")
        manifest = pd.read_csv(manifest_path)
        manifest["patient_id"] = manifest["patient_id"].astype(str).str.strip()

        labels = self._load_progression_labels(dataset_dir)
        manifest = manifest.merge(labels, on="patient_id", how="inner")
        print(f"  Patients with both a baseline and follow-up grade row: {manifest['patient_id'].nunique()}")

        before = len(manifest)
        manifest = manifest[~manifest["baseline_grade_raw"].isin(EXCLUDED_GRADES)]
        manifest = manifest[~manifest["followup_grade_raw"].isin(EXCLUDED_GRADES)]
        manifest = manifest.dropna(subset=["baseline_grade_raw", "followup_grade_raw"])
        print(f"  Excluded/missing baseline or follow-up grade: {before} -> {len(manifest)} pairs")

        manifest["baseline_icdr"] = manifest["baseline_grade_raw"].astype(int).map(tianjin_grade_to_icdr)
        manifest["followup_icdr"] = manifest["followup_grade_raw"].astype(int).map(tianjin_grade_to_icdr)
        # Event = progressed by at least one ICDR stage within the fixed follow-up window.
        # NOT clinically the same claim as "progressed to sight-threatening disease" -- any
        # upward stage move counts. Flag if this thesis's definition of "progression" should
        # instead require a specific stage threshold (e.g. reaching PDR).
        manifest["event"] = (manifest["followup_icdr"] > manifest["baseline_icdr"]).astype(int)
        manifest["time_to_followup"] = fixed_followup_years

        def lookup_lbs(baseline_path):
            for candidate_id in _module1_image_id_candidates(baseline_path):
                if candidate_id in cache:
                    return cache[candidate_id].get("lbs")
            return None

        manifest["lbs"] = manifest["baseline_path"].apply(lookup_lbs)
        before = len(manifest)
        manifest = manifest.dropna(subset=["lbs"])
        print(f"  Rows with a Module 1 LBS cache hit: {before} -> {len(manifest)}")

        if len(manifest) == 0:
            raise RuntimeError(
                "No usable rows after dropping excluded/missing grades and cache misses -- "
                "Module 3 needs a real progression label and a real LBS value for every "
                "training sample, unlike Module 2's placeholder-tolerant conditioning."
            )

        strata, thresholds_by_grade = stratify_lbs_by_grade(
            manifest["lbs"].values, manifest["baseline_icdr"].values
        )
        manifest["lbs_stratum"] = strata
        self.lbs_thresholds_by_grade = thresholds_by_grade

        self.manifest = manifest.reset_index(drop=True)
        event_rate = self.manifest["event"].mean()
        print(f"✓ Found {len(self.manifest)} usable Module 3 pairs "
              f"({self.manifest['patient_id'].nunique()} patients, "
              f"{event_rate:.1%} progressed by {fixed_followup_years:.0f}yr)")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @staticmethod
    def _load_progression_labels(dataset_dir) -> pd.DataFrame:
        """
        Reads BOTH the Baseline and 2-Year Follow-up sheets of Organized_Data of
        Patients.xlsx and returns one (patient_id, baseline_grade_raw, followup_grade_raw)
        row per patient present in both. Sheet/column matching is case/space-insensitive with
        a loud failure on a miss -- see tianjin_dataset.py's identical convention and its
        module docstring's note on the xlsx schema being unverified against the real file.
        """
        xlsx_path = os.path.join(dataset_dir, "Organized_Data of Patients.xlsx")
        if not os.path.isfile(xlsx_path):
            raise FileNotFoundError(f"Could not find 'Organized_Data of Patients.xlsx' in {dataset_dir}")

        xl = pd.ExcelFile(xlsx_path)
        baseline_sheet = next((s for s in xl.sheet_names if "baseline" in _normalize_colname(s)), None)
        followup_sheet = next((s for s in xl.sheet_names if "follow" in _normalize_colname(s)), None)
        if baseline_sheet is None or followup_sheet is None:
            raise KeyError(
                f"Expected a sheet name containing 'baseline' and one containing 'follow' "
                f"(e.g. '2-Year Follow-up'). Actual sheets: {xl.sheet_names}."
            )

        base_df = xl.parse(baseline_sheet)
        fu_df = xl.parse(followup_sheet)

        base_id_col = _find_column(base_df.columns, ["patient", "id"], "baseline patient ID")
        base_grade_col = _find_column(base_df.columns, ["dr", "grade"], "baseline DR grade")
        fu_id_col = _find_column(fu_df.columns, ["patient", "id"], "follow-up patient ID")
        fu_grade_col = _find_column(fu_df.columns, ["dr", "grade"], "follow-up DR grade")
        print(f"  Baseline sheet '{baseline_sheet}': patient_id <- '{base_id_col}', grade <- '{base_grade_col}'")
        print(f"  Follow-up sheet '{followup_sheet}': patient_id <- '{fu_id_col}', grade <- '{fu_grade_col}'")

        base_df = base_df.rename(columns={base_id_col: "patient_id", base_grade_col: "baseline_grade_raw"})
        fu_df = fu_df.rename(columns={fu_id_col: "patient_id", fu_grade_col: "followup_grade_raw"})

        base_df["patient_id"] = base_df["patient_id"].astype(str).str.strip()
        fu_df["patient_id"] = fu_df["patient_id"].astype(str).str.strip()
        base_df["baseline_grade_raw"] = pd.to_numeric(base_df["baseline_grade_raw"], errors="coerce")
        fu_df["followup_grade_raw"] = pd.to_numeric(fu_df["followup_grade_raw"], errors="coerce")

        base_df = base_df[["patient_id", "baseline_grade_raw"]].drop_duplicates(subset="patient_id")
        fu_df = fu_df[["patient_id", "followup_grade_raw"]].drop_duplicates(subset="patient_id")
        # inner join: Module 3 needs BOTH grades to compute an event label, unlike Module 2
        # which only ever needs the baseline grade.
        return base_df.merge(fu_df, on="patient_id", how="inner")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        baseline_path = os.path.join(self.dataset_dir, row["baseline_path"])
        try:
            img = Image.open(baseline_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.manifest))
        img = self.transform(img)

        return {
            "baseline": img,
            "event": torch.tensor(float(row["event"])),
            "time_to_followup": torch.tensor(float(row["time_to_followup"])),
            "lbs": torch.tensor(float(row["lbs"])),
            "lbs_stratum_idx": torch.tensor(STRATUM_TO_IDX[row["lbs_stratum"]], dtype=torch.long),
            "baseline_grade_icdr": torch.tensor(int(row["baseline_icdr"]), dtype=torch.long),
            "patient_id": row["patient_id"],
        }


def patient_level_split(dataset: TianjinSurvivalDataset, val_fraction=0.2, seed=42):
    """
    Splits at the PATIENT level, not the row level -- a plain torch.utils.data.random_split
    could put the same patient's two eyes into both train and val, leaking the label (they
    share one progression event) across the split. Matches docs/DATASETS.md's recommended
    Tianjin split convention (patient-level, both eyes together).
    """
    patient_ids = dataset.manifest["patient_id"].tolist()
    unique_patients = sorted(set(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_patients)
    n_val_patients = max(1, int(len(unique_patients) * val_fraction))
    val_patients = set(unique_patients[:n_val_patients])

    train_idx = [i for i, pid in enumerate(patient_ids) if pid not in val_patients]
    val_idx = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./retinal-dr-longitudinal")
    parser.add_argument("--module1-cache", required=True)
    args = parser.parse_args()

    print("=" * 70)
    print("Testing Tianjin Survival Dataset (Module 3)")
    print("=" * 70)

    dataset = TianjinSurvivalDataset(args.dataset_dir, module1_cache_path=args.module1_cache, image_size=128)
    print(f"\nTotal usable pairs: {len(dataset)}")
    train_set, val_set = patient_level_split(dataset)
    print(f"Patient-level split: {len(train_set)} train / {len(val_set)} val")

    sample = dataset[0]
    print(f"  Baseline shape: {sample['baseline'].shape}")
    print(f"  Event: {sample['event'].item()}, time_to_followup: {sample['time_to_followup'].item()}")
    print(f"  LBS: {sample['lbs'].item():.5f}, stratum idx: {sample['lbs_stratum_idx'].item()}")
    print(f"  Patient ID: {sample['patient_id']}")
    print("\n✓ Module 3 survival dataset working!")
