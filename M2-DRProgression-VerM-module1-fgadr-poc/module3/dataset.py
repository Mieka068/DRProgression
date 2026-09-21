"""
Module 3 -- Tianjin survival dataset.

Builds (baseline_image, LBS stratum, progression event, follow-up time) samples from the same
Tianjin corrected_manifest.csv + Organized_Data of Patients.xlsx that Module 2's
tianjin_dataset.py reads, plus the module1_outputs_tianjin.pt cache built by
notebooks/05_tianjin_data_prep_colab.ipynb. This is a separate loader, not a reuse of
TianjinLongitudinalDataset: Module 3 needs the "2-Year Follow-up" sheet's grade too, to
compute the progression event label, which Module 2 never touches.

Module 3 takes the real baseline fundus image directly, independent of Module 2's synthesized
output.

Every Tianjin patient's follow-up interval is a fixed 2 years, and only whether progression
happened by that single checkpoint is observed, never the exact progression date within it.
This is current-status / case-1 interval-censored data, not exact-event-time survival data.
`time_to_followup` is threaded through as a per-sample field rather than hardcoded, so this
interface doesn't need to change if a future source supplies real variable inter-visit timing.

Unlike tianjin_dataset.py's placeholder-tolerant grade fallback, Module 3 needs a real
progression label to train on at all -- a row with an excluded/missing baseline or follow-up
grade is dropped here, not defaulted.

Grading is per-eye, not per-patient: Organized_Data of Patients.xlsx has separate OS/OD/
at-risk-eye columns per sheet, not one flat "DR grade" column, and corrected_manifest.csv
doesn't say which eye a row is. This loader requires the same laterality_resolved.csv that
tianjin_dataset.py requires (module1/resolve_eye_laterality.py), and applies the same
OD/OS/worse-eye fallback per row.
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
    _find_eye_grade_column,
    _module1_image_id_candidates,
    _normalize_colname,
    tianjin_grade_to_icdr,
)

# ImageNet normalization stats -- this feeds an ImageNet-pretrained EfficientNet-B4 backbone,
# not a GAN, so it uses different stats than Module 2's [-1,1] range.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STRATUM_TO_IDX = {"low": 0, "medium": 1, "high": 2}


class TianjinSurvivalDataset(Dataset):
    """
    One sample per corrected_manifest.csv row (i.e. per baseline<->follow-up eye pair) whose
    patient has both a usable baseline grade and a usable follow-up grade. Multiple rows can
    belong to the same patient (e.g. both eyes) -- use patient_level_split() below rather than
    a plain random split to avoid leaking a patient across train/val.
    """

    def __init__(self, dataset_dir, module1_cache_path, image_size=380, fixed_followup_years=2.0):
        """
        Args:
            dataset_dir: path to the extracted usama10/retinal-dr-longitudinal dataset (same
                layout tianjin_dataset.py expects).
            module1_cache_path: required -- path to the module1/apply_to_progression_data.py
                cache built over Tianjin's baseline images. Module 3's LBS input feature comes
                entirely from this cache's precomputed 'lbs' field; there is no placeholder
                fallback.
            image_size: resize images to this size. Default 380 matches EfficientNet-B4's
                standard ImageNet input resolution.
            fixed_followup_years: the observation time for every sample. Tianjin's follow-up
                interval is fixed at 2 years for the whole cohort, exposed as a parameter
                rather than hardcoded so a future variable-interval source can reuse this
                class/interface.
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
        print(f"  Total eye-pairs in corrected_manifest.csv: {len(manifest)}")

        # Resolve which eye each manifest row is via module1/resolve_eye_laterality.py's
        # precomputed output. The same resolved eye applies to both the baseline and
        # follow-up image of a row, since corrected_manifest.csv pairs same-eye images by
        # construction.
        laterality_path = os.path.join(dataset_dir, "laterality_resolved.csv")
        if not os.path.isfile(laterality_path):
            raise FileNotFoundError(
                f"Could not find laterality_resolved.csv in {dataset_dir}. Run "
                f"`python module1/resolve_eye_laterality.py --dataset-dir {dataset_dir}` first."
            )
        laterality_df = pd.read_csv(laterality_path)[["key", "eye"]]
        manifest = manifest.merge(laterality_df, on="key", how="left")
        manifest["eye"] = manifest["eye"].fillna("uncertain")

        labels = self._load_progression_labels(dataset_dir)
        manifest = manifest.merge(labels, on="patient_id", how="inner")
        print(f"  Patients with both a baseline and follow-up grade row: {manifest['patient_id'].nunique()}")

        def _pick_eye_grades(row):
            if row["eye"] == "OD":
                return row["baseline_od_grade_raw"], row["followup_od_grade_raw"], True
            if row["eye"] == "OS":
                return row["baseline_os_grade_raw"], row["followup_os_grade_raw"], True
            return row["baseline_worse_eye_grade_raw"], row["followup_worse_eye_grade_raw"], False

        picked = manifest.apply(_pick_eye_grades, axis=1, result_type="expand")
        manifest["baseline_grade_raw"] = picked[0]
        manifest["followup_grade_raw"] = picked[1]
        manifest["grade_is_eye_specific"] = picked[2]
        n_eye_specific = int(manifest["grade_is_eye_specific"].sum())
        print(f"  Eye-specific grade resolved for {n_eye_specific}/{len(manifest)} rows "
              f"({len(manifest) - n_eye_specific} fell back to the worse-eye summary)")

        before = len(manifest)
        manifest = manifest[~manifest["baseline_grade_raw"].isin(EXCLUDED_GRADES)]
        manifest = manifest[~manifest["followup_grade_raw"].isin(EXCLUDED_GRADES)]
        manifest = manifest.dropna(subset=["baseline_grade_raw", "followup_grade_raw"])
        print(f"  Excluded/missing baseline or follow-up grade: {before} -> {len(manifest)} pairs")

        manifest["baseline_icdr"] = manifest["baseline_grade_raw"].astype(int).map(tianjin_grade_to_icdr)
        manifest["followup_icdr"] = manifest["followup_grade_raw"].astype(int).map(tianjin_grade_to_icdr)
        # Event = progressed by at least one ICDR stage within the fixed follow-up window.
        manifest["event"] = (manifest["followup_icdr"] > manifest["baseline_icdr"]).astype(int)
        manifest["time_to_followup"] = fixed_followup_years

        # Cross-check against the dataset's own reported progression column (1=No, 2=Yes),
        # where present. Informational only -- does not redefine `event` above.
        has_dataset_flag = manifest["dataset_reported_progression_raw"].notna()
        if has_dataset_flag.any():
            dataset_event = (manifest.loc[has_dataset_flag, "dataset_reported_progression_raw"] == 2).astype(int)
            agreement = (dataset_event == manifest.loc[has_dataset_flag, "event"]).mean()
            print(
                f"  Cross-check vs. dataset's own 'Progression' column: {agreement:.1%} "
                f"agreement on {int(has_dataset_flag.sum())} rows with both labels"
            )

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
                "training sample."
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
        Reads both the Baseline and 2-Year Follow-up sheets of Organized_Data of
        Patients.xlsx and returns one row per patient present in both, with separate per-eye
        grade columns for each sheet (baseline_os/od/worse_eye_grade_raw, and the followup_*
        equivalents). Per-eye selection for a specific manifest row happens in __init__, using
        the same resolved 'eye' column tianjin_dataset.py uses.

        Also reads the follow-up sheet's own "Progression (1=No; 2=Yes)" column as
        dataset_reported_progression_raw, purely as a cross-check against this loader's own
        `event = followup_icdr > baseline_icdr` computation -- see __init__'s agreement-rate
        print. Missing entirely if that column can't be found.
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

        def _match_eye_cols(df, sheet_label):
            try:
                id_col = _find_column(df.columns, ["id"], f"{sheet_label} patient ID")
                os_col = _find_eye_grade_column(df.columns, "os", f"{sheet_label} OS grade")
                od_col = _find_eye_grade_column(df.columns, "od", f"{sheet_label} OD grade")
                worse_col = _find_column(df.columns, ["atrisk", "grade"], f"{sheet_label} worse-eye grade")
            except KeyError as e:
                raise KeyError(
                    f"{e}\nAll normalized column names in '{sheet_label}' sheet: "
                    f"{[(c, _normalize_colname(c)) for c in df.columns]}."
                ) from e
            return id_col, os_col, od_col, worse_col

        base_id, base_os, base_od, base_worse = _match_eye_cols(base_df, "baseline")
        fu_id, fu_os, fu_od, fu_worse = _match_eye_cols(fu_df, "follow-up")
        print(f"  Baseline sheet '{baseline_sheet}': patient_id <- '{base_id}', OS <- '{base_os}', "
              f"OD <- '{base_od}', worse-eye <- '{base_worse}'")
        print(f"  Follow-up sheet '{followup_sheet}': patient_id <- '{fu_id}', OS <- '{fu_os}', "
              f"OD <- '{fu_od}', worse-eye <- '{fu_worse}'")

        base_df = base_df.rename(columns={
            base_id: "patient_id", base_os: "baseline_os_grade_raw", base_od: "baseline_od_grade_raw",
            base_worse: "baseline_worse_eye_grade_raw",
        })
        fu_df = fu_df.rename(columns={
            fu_id: "patient_id", fu_os: "followup_os_grade_raw", fu_od: "followup_od_grade_raw",
            fu_worse: "followup_worse_eye_grade_raw",
        })

        base_df["patient_id"] = base_df["patient_id"].astype(str).str.strip()
        fu_df["patient_id"] = fu_df["patient_id"].astype(str).str.strip()
        for c in ("baseline_os_grade_raw", "baseline_od_grade_raw", "baseline_worse_eye_grade_raw"):
            base_df[c] = pd.to_numeric(base_df[c], errors="coerce")
        for c in ("followup_os_grade_raw", "followup_od_grade_raw", "followup_worse_eye_grade_raw"):
            fu_df[c] = pd.to_numeric(fu_df[c], errors="coerce")

        try:
            progression_col = _find_column(fu_df.columns, ["progression"], "dataset-reported progression")
            fu_df = fu_df.rename(columns={progression_col: "dataset_reported_progression_raw"})
            fu_df["dataset_reported_progression_raw"] = pd.to_numeric(
                fu_df["dataset_reported_progression_raw"], errors="coerce"
            )
            print(f"  Found dataset's own progression column: '{progression_col}' (cross-check only)")
        except KeyError:
            fu_df["dataset_reported_progression_raw"] = np.nan
            print("  No dataset-reported progression column found -- skipping the cross-check.")

        base_cols = ["patient_id", "baseline_os_grade_raw", "baseline_od_grade_raw", "baseline_worse_eye_grade_raw"]
        fu_cols = ["patient_id", "followup_os_grade_raw", "followup_od_grade_raw",
                   "followup_worse_eye_grade_raw", "dataset_reported_progression_raw"]
        base_df = base_df[base_cols].drop_duplicates(subset="patient_id")
        fu_df = fu_df[fu_cols].drop_duplicates(subset="patient_id")
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
            "grade_is_eye_specific": bool(row["grade_is_eye_specific"]),
        }


def patient_level_split(dataset: TianjinSurvivalDataset, val_fraction=0.2, seed=42):
    """
    Splits at the patient level, not the row level -- a plain torch.utils.data.random_split
    could put the same patient's two eyes into both train and val, leaking the label across
    the split.
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
