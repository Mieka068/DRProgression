"""
Tianjin Longitudinal Dataset Loader
Source: huggingface.co/datasets/usama10/retinal-dr-longitudinal
Paper: Usama et al., "Conditional Latent Diffusion for Predictive Retinal Fundus Image
Synthesis from Baseline Imaging and Clinical Metadata" (manuscript, 2026)

574 patients, baseline + 2-year follow-up fundus photographs, collected at Tianjin Medical
University Eye Hospital. Unlike FIRE and LongDRScreening (which have no DR grade of their
own and fall back to a placeholder Stage-2 one-hot in fire_dataset.py/longdr_dataset.py),
this dataset ships REAL clinical DR grades per patient in `Organized_Data of Patients.xlsx`.
That's new: this is the first source in the pipeline that can condition Module 2 on ground
truth rather than a placeholder or a Module 1 prediction.

What's ours vs. what the dataset provides:
  - Pairing: the dataset's own `corrected_manifest.csv` is used as-is for baseline<->follow-up
    correspondence. Do NOT pair by filename order -- the dataset card documents that naive
    filename-order pairing puts the wrong eye's follow-up against 45.1% of baselines (mixes
    up left/right eye per patient). corrected_manifest.csv already fixes this; we just read it.
  - Grade extraction, grade-scale mapping, exclusion filtering, and train/val/test split
    are ours (dataset ships raw grades in an Excel workbook, not directly usable).

Expected directory layout (as published at huggingface.co/datasets/usama10/
retinal-dr-longitudinal), rooted at `dataset_dir`:
    baseline fundus images/
        00194/
            00194-7256.jpg
        ...
    2 year follow-up fundus images/
        1036_2.jpg
        ...
    Organized_Data of Patients.xlsx
    corrected_manifest.csv

Grade scale mapping: the dataset's embedded column legend reads "1=No Apparent Retinopathy;
2=Mild NPDR; 3=Moderate NPDR; 4=Severe NPDR; 5=PDR; 6=After laser; 7=Missing" -- this matches
ICDR's 5-stage ordering exactly, just 1-indexed, so mapping by subtracting 1 (grade 1 -> ICDR
0, ..., grade 5 -> ICDR 4) is confirmed correct (see docs/IMPLEMENTATION_PLAN.md Task B.1).
Whether ETDRS and ICDR are clinically equivalent staging *criteria* (not just numerically
parallel labels) is still worth Dr. Atienza/the adviser's sign-off before this appears in the
methodology chapter -- the numeric offset itself is not in question, the clinical equivalence
claim is. Grades 6 (post-photocoagulation) and 7 (ungradable/missing) are dropped per the
dataset's own column legend, since neither reflects natural disease severity.

GRADING IS PER-EYE, NOT PER-PATIENT -- confirmed directly (this loader's own _load_grades()
was run against the real Organized_Data of Patients.xlsx and corrected_manifest.csv, not
inferred): Organized_Data of Patients.xlsx has no single "DR grade" column. Instead each
sheet has three columns: an "OS Grade" (left eye), an "OD Grade" (right eye), and an
"At-risk Eye Grade (Worse eye)" patient-level summary. corrected_manifest.csv has one row per
EYE (a patient can appear twice) but carries no OD/OS marker, and neither do the baseline
filenames. Per instruction, every eye-pair is kept as its own training sample (not collapsed
to one row per patient) to maximize data, especially for Module 3 -- this means we cannot
skip resolving which eye a given row actually is: of 613 real patients with both an OS and OD
grade present, 102 (16.6%) have a different OS grade than OD grade, so guessing wrong is a
real mislabeling risk, not a rounding error (both this exact count and the 476/473
OS/OD-at-Stage-0 counts referenced in docs/ROADMAP.md's Task G note were independently
reproduced against the real file, not just carried over from the plan that first reported
them). All 1,115 real corrected_manifest.csv rows matched a patient_id present in the xlsx --
no ID-format mismatch (e.g. zero-padding) actually occurs in this dataset, despite the
generic warning below being kept as a safety net for a differently-formatted future export.
Laterality is resolved by module1/resolve_eye_laterality.py (optic disc position heuristic)
into `laterality_resolved.csv`, which this loader requires -- see __init__. That heuristic
itself still needs validating against the real baseline photographs (not done as of this
note -- it requires the actual images, which metadata-only verification can't substitute
for). Rows whose laterality couldn't be confidently resolved ("uncertain") fall back to the
At-risk Eye Grade (Worse eye) column for that row only, tracked via the
`grade_is_eye_specific` output key, rather than being dropped.

Module 1 cache key scheme (see module1/apply_to_progression_data.py): that script's cache is
keyed by the image's path relative to whatever `--images-dir` it was pointed at, extension
stripped, os.sep replaced with "__". This loader assumes the recommended invocation points
`--images-dir` at "<dataset_dir>/baseline fundus images" (see
notebooks/05_tianjin_data_prep_colab.ipynb) -- that produces keys like "00194__00194-7256" for
this dataset's nested per-patient baseline layout, matching longdr_dataset.py's identical
"<eye>__<stem>" scheme. As a fallback (in case a cache was built pointing --images-dir
somewhere else), we also try the bare filename stem. If neither key is present, the mask falls
back to the same empty-mask placeholder FIRE/LongDR use.

License note: CC BY-NC-4.0, non-commercial research use only; do not redistribute the raw
images outside this dataset's own hosting.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EXCLUDED_GRADES = {6, 7}  # post-photocoagulation, ungradable -- dataset card's own instruction
NUM_STAGES = 5  # ICDR 0-4, matching target_grade elsewhere in this repo
BASELINE_SUBDIR = "baseline fundus images"
FOLLOWUP_SUBDIR = "2 year follow-up fundus images"


def tianjin_grade_to_icdr(raw_grade: int) -> int:
    """ETDRS-labeled 1-5 -> ICDR 0-4, by subtracting 1. See module docstring's grade-scale note."""
    return raw_grade - 1


def _normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_column(columns, must_contain, label):
    """
    Case/space-insensitive match: returns the first column whose normalized name contains
    every token in `must_contain`. Raises with the real column list on a miss, rather than
    silently mis-mapping a column. Verified directly against the real Organized_Data of
    Patients.xlsx: `["id"]` and `["atrisk","grade"]` both match their intended column
    correctly there (see _find_eye_grade_column for the one case -- "os"/"od" -- where a
    plain substring match like this one is NOT safe).
    """
    for col in columns:
        norm = _normalize_colname(col)
        if all(tok in norm for tok in must_contain):
            return col
    raise KeyError(
        f"Could not find a column for {label}: looked for a column name containing all of "
        f"{must_contain} (case/space-insensitive). Actual columns in the sheet: {list(columns)}. "
        f"Update the matching tokens in tianjin_dataset.py::_load_grades if the real header "
        f"wording differs."
    )


def _find_eye_grade_column(columns, eye_prefix, label):
    """
    Matches a column whose normalized name STARTS WITH `eye_prefix` ('os' or 'od') and also
    contains 'grade'. A plain substring-anywhere match (like _find_column) is UNSAFE for these
    two specifically: the real OS/OD Grade columns embed their full numeric legend inline
    (e.g. "...3=Moderate Non-Proliferative Diabetic Retinopathy...") and "Moderate" itself
    contains the literal substring "od" ("m-OD-erate"), which silently matches the OD lookup
    against the OS column instead. This was caught with a synthetic fixture before the real
    file was available, and this exact function has since been re-run directly against the
    real Organized_Data of Patients.xlsx (both the Baseline and 2-Year Follow-up sheets) and
    correctly resolves OS/OD/At-risk to three distinct columns -- not a hypothetical risk.
    Requiring the eye token as a PREFIX, not just present anywhere, avoids the collision since
    the real columns are always literally named "OS Grade(...)" / "OD Grade(...)".
    """
    for col in columns:
        norm = _normalize_colname(col)
        if norm.startswith(eye_prefix) and "grade" in norm:
            return col
    raise KeyError(
        f"Could not find a column for {label}: looked for a column name STARTING WITH "
        f"'{eye_prefix}' and containing 'grade'. Actual columns: {list(columns)}."
    )


def _module1_image_id_candidates(baseline_path: str) -> list:
    """
    Candidate cache keys to look up this baseline image's Module 1 output. See module
    docstring's "Module 1 cache key scheme" note -- we don't know for certain which
    --images-dir a given cache was built against, so try the recommended scheme first, then
    fall back to the bare filename stem.
    """
    norm_path = baseline_path.replace("\\", "/")
    stem_no_ext = os.path.splitext(norm_path)[0]
    candidates = []

    prefix = BASELINE_SUBDIR + "/"
    if norm_path.lower().startswith(prefix.lower()):
        rel = stem_no_ext[len(prefix):]
        candidates.append(rel.replace("/", "__"))

    candidates.append(Path(norm_path).stem)
    return candidates


class TianjinLongitudinalDataset(Dataset):
    """
    Loads Tianjin baseline -> 2-year-follow-up pairs via corrected_manifest.csv, with REAL
    DR grade conditioning from Organized_Data of Patients.xlsx wherever the manifest's
    grade filter keeps a row.

    Returns the same dict keys as FIREDataset/LongDRScreeningDataset so this drops straight
    into combined_dataset.py's ConcatDataset, plus four extra keys ('grade_is_real',
    'patient_id', 'pair_quality', 'grade_is_eye_specific') that those two never set --
    combined_dataset.py's AugmentedPairDataset fills in defaults for those keys on FIRE/LongDR
    samples so every sample in a mixed batch has the same dict shape (default_collate requires
    this).
    """

    SOURCE_NAME = "Tianjin"

    def __init__(
        self,
        dataset_dir,
        image_size=128,
        min_pair_quality=None,
        module1_cache_path=None,
        registration_cache_path=None,
    ):
        """
        Args:
            dataset_dir: path to the extracted usama10/retinal-dr-longitudinal dataset, i.e.
                the folder containing `corrected_manifest.csv`, `Organized_Data of
                Patients.xlsx`, `baseline fundus images/`, and `2 year follow-up fundus
                images/`.
            image_size: resize images to this size (matches FIRE/LongDR default of 128).
            min_pair_quality: optional float; drop corrected_manifest.csv rows whose
                `quality` (normalized cross-correlation of the recovered pairing) is below
                this threshold. None keeps every row the manifest provides regardless of
                registration confidence -- set e.g. 0.5 to be conservative.
            module1_cache_path: optional path to a module1/apply_to_progression_data.py
                cache. Same fallback contract as FIREDataset/LongDRScreeningDataset: if a
                real Module 1 mask exists for a given baseline image, it's used instead of
                the zero mask below. Grade conditioning, unlike the other two loaders, does
                NOT fall back to this cache first -- the real clinical grade from the xlsx
                always wins when present, since it's more reliable than a Module 1 prediction.
                The cache is only consulted for the lesion mask channel.
            registration_cache_path: Optional path to a cache produced by
                module1/train_registration.py ({(source, baseline_path): warped_followup_uint8}).
                Same fallback behavior as FIREDataset -- see its docstring and
                docs/IMPLEMENTATION_PLAN.md Task E/F.
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

        manifest_path = os.path.join(dataset_dir, "corrected_manifest.csv")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Could not find corrected_manifest.csv in {dataset_dir}. Do not fall back to "
                "filename-order pairing -- see module docstring."
            )
        manifest = pd.read_csv(manifest_path)
        required_manifest_cols = {"patient_id", "baseline_path", "followup_path"}
        missing = required_manifest_cols - set(manifest.columns)
        if missing:
            raise KeyError(
                f"corrected_manifest.csv is missing expected column(s) {missing}. Actual "
                f"columns: {list(manifest.columns)}."
            )

        if min_pair_quality is not None:
            before = len(manifest)
            manifest = manifest[manifest["quality"] >= min_pair_quality]
            print(f"  Quality filter (>= {min_pair_quality}): {before} -> {len(manifest)} pairs")

        # Grading is per-eye, not per-patient (see module docstring) -- resolve which eye each
        # manifest row is via module1/resolve_eye_laterality.py's precomputed output. Required,
        # not optional: without it there's no reliable way to pick OS vs. OD grade for a row.
        laterality_path = os.path.join(dataset_dir, "laterality_resolved.csv")
        if not os.path.isfile(laterality_path):
            raise FileNotFoundError(
                f"Could not find laterality_resolved.csv in {dataset_dir}. Tianjin's grade "
                "columns are per-eye (OS/OD), not per-patient, and corrected_manifest.csv "
                "doesn't say which eye a given row is -- run "
                "`python module1/resolve_eye_laterality.py --dataset-dir " + dataset_dir + "` "
                "first (see its module docstring and docs/IMPLEMENTATION_PLAN.md Task B.3)."
            )
        laterality_df = pd.read_csv(laterality_path)[["key", "eye"]]

        manifest["patient_id"] = manifest["patient_id"].astype(str).str.strip()
        manifest = manifest.merge(laterality_df, on="key", how="left")
        manifest["eye"] = manifest["eye"].fillna("uncertain")

        grades = self._load_grades(dataset_dir)
        manifest = manifest.merge(grades, on="patient_id", how="left")

        def _pick_grade(row):
            if row["eye"] == "OD":
                return row["od_grade_raw"], True
            if row["eye"] == "OS":
                return row["os_grade_raw"], True
            # "uncertain" laterality -> fall back to the patient-level worse-eye summary for
            # this row only (per docs/IMPLEMENTATION_PLAN.md's fallback policy), rather than
            # dropping the eye-pair -- keeps every sample usable, maximizing training data.
            return row["worse_eye_grade_raw"], False

        picked = manifest.apply(_pick_grade, axis=1, result_type="expand")
        manifest["dr_grade_raw"] = picked[0]
        manifest["grade_is_eye_specific"] = picked[1]
        n_eye_specific = int(manifest["grade_is_eye_specific"].sum())
        print(f"  Eye-specific grade resolved for {n_eye_specific}/{len(manifest)} rows "
              f"({len(manifest) - n_eye_specific} fell back to the worse-eye summary)")

        match_rate = manifest["dr_grade_raw"].notna().mean() if len(manifest) else 0.0
        if match_rate < 0.5:
            print(
                f"⚠ Only {match_rate:.0%} of pairs matched a real grade by patient_id -- check "
                "for an ID-format mismatch (e.g. zero-padding) between corrected_manifest.csv's "
                "patient_id and the xlsx's patient ID column before trusting grade_is_real "
                "counts downstream."
            )

        # Exclude grades 6/7 per the dataset card. Rows with no grade at all (patient not in
        # the Baseline sheet, or an unparseable value) are KEPT but flagged via grade_is_real
        # = False, same placeholder-Stage-2 treatment as FIRE/LongDR -- we don't drop a real
        # image pair just because its metadata row is missing.
        before = len(manifest)
        manifest = manifest[~manifest["dr_grade_raw"].isin(EXCLUDED_GRADES)]
        print(f"  Excluded grades {EXCLUDED_GRADES}: {before} -> {len(manifest)} pairs")

        self.manifest = manifest.reset_index(drop=True)
        print(f"✓ Found {len(self.manifest)} pairs in Tianjin dataset "
              f"({self.manifest['dr_grade_raw'].notna().sum()} with real grade)")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def _load_grades(dataset_dir) -> pd.DataFrame:
        """
        Reads the Baseline sheet of Organized_Data of Patients.xlsx and returns a
        (patient_id, os_grade_raw, od_grade_raw, worse_eye_grade_raw) frame -- one row per
        patient, since the real xlsx has separate per-eye grade columns rather than one flat
        "DR grade" column (confirmed against the real downloaded file -- see module docstring
        and docs/IMPLEMENTATION_PLAN.md Task B.1). Per-eye selection for a specific manifest
        row happens in __init__, using the resolved 'eye' column from
        module1/resolve_eye_laterality.py's laterality_resolved.csv.

        Sheet and column names are matched case/space-insensitively rather than by exact
        string. A miss raises with the actual sheet/column names found, rather than silently
        mis-mapping one.
        """
        xlsx_path = os.path.join(dataset_dir, "Organized_Data of Patients.xlsx")
        if not os.path.isfile(xlsx_path):
            raise FileNotFoundError(f"Could not find 'Organized_Data of Patients.xlsx' in {dataset_dir}")

        xl = pd.ExcelFile(xlsx_path)
        sheet_name = next(
            (s for s in xl.sheet_names if "baseline" in _normalize_colname(s)), None
        )
        if sheet_name is None:
            raise KeyError(
                f"No sheet name containing 'baseline' found in {xlsx_path}. Actual sheets: "
                f"{xl.sheet_names}."
            )
        df = xl.parse(sheet_name)

        try:
            id_col = _find_column(df.columns, ["id"], "patient ID")
            os_col = _find_eye_grade_column(df.columns, "os", "OS (left eye) grade")
            od_col = _find_eye_grade_column(df.columns, "od", "OD (right eye) grade")
            worse_col = _find_column(df.columns, ["atrisk", "grade"], "at-risk (worse eye) grade")
        except KeyError as e:
            raise KeyError(
                f"{e}\nAll normalized column names in sheet '{sheet_name}': "
                f"{[(c, _normalize_colname(c)) for c in df.columns]}. If the real header "
                "wording differs from what's expected here, fix the matching tokens in "
                "tianjin_dataset.py::_load_grades (or reference the column by its exact "
                "string directly, per docs/IMPLEMENTATION_PLAN.md Task B.4's note)."
            ) from e
        print(f"  Reading grades from sheet '{sheet_name}': patient_id <- '{id_col}', "
              f"OS <- '{os_col}', OD <- '{od_col}', worse-eye <- '{worse_col}'")

        df = df.rename(columns={
            id_col: "patient_id", os_col: "os_grade_raw", od_col: "od_grade_raw",
            worse_col: "worse_eye_grade_raw",
        })
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        for c in ("os_grade_raw", "od_grade_raw", "worse_eye_grade_raw"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["patient_id", "os_grade_raw", "od_grade_raw", "worse_eye_grade_raw"]].drop_duplicates(
            subset="patient_id"
        )

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        baseline_path = os.path.join(self.dataset_dir, row["baseline_path"])
        followup_path = os.path.join(self.dataset_dir, row["followup_path"])

        try:
            img1 = Image.open(baseline_path).convert("RGB")
            warped = (
                self.registration_cache.get((self.SOURCE_NAME, row["baseline_path"]))
                if self.registration_cache
                else None
            )
            if warped is not None:
                # Pre-registered (baseline-aligned) follow-up, per
                # docs/IMPLEMENTATION_PLAN.md Task F -- falls back to the raw follow-up
                # image otherwise.
                img2 = Image.fromarray(np.transpose(warped, (1, 2, 0)))  # CHW uint8 -> HWC for PIL
            else:
                img2 = Image.open(followup_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.manifest))

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        module1_output = None
        if self.module1_cache:
            for candidate_id in _module1_image_id_candidates(row["baseline_path"]):
                if candidate_id in self.module1_cache:
                    module1_output = self.module1_cache[candidate_id]
                    break

        if module1_output is not None:
            mask_np = module1_output["mask"]
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
            mask = torch.from_numpy(np.array(mask_img) > 0).float().unsqueeze(0)
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        raw_grade = row["dr_grade_raw"]
        target_grade = torch.zeros(NUM_STAGES)
        if pd.notna(raw_grade):
            icdr_grade = int(np.clip(tianjin_grade_to_icdr(int(raw_grade)), 0, NUM_STAGES - 1))
            target_grade[icdr_grade] = 1.0
            grade_is_real = True
        else:
            target_grade[2] = 1.0  # same Stage-2 placeholder convention as FIRE/LongDR
            grade_is_real = False

        return {
            "baseline": img1,
            "follow_up": img2,
            "mask": mask,
            "target_grade": target_grade,
            "source": "Tianjin",
            "eye_id": row["patient_id"],
            # Extra keys not present on FIRE/LongDR samples -- combined_dataset.py's
            # AugmentedPairDataset defaults these on other sources' samples so batches that
            # mix sources still collate (see class docstring).
            "grade_is_real": grade_is_real,
            "patient_id": row["patient_id"],
            "pair_quality": float(row["quality"]) if "quality" in row and pd.notna(row["quality"]) else -1.0,
            "grade_is_eye_specific": bool(row["grade_is_eye_specific"]) if "grade_is_eye_specific" in row else False,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./retinal-dr-longitudinal")
    parser.add_argument("--min-quality", type=float, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("Testing Tianjin Longitudinal Data Loader")
    print("=" * 70)

    dataset = TianjinLongitudinalDataset(args.dataset_dir, image_size=128, min_pair_quality=args.min_quality)
    print(f"\nTotal pairs: {len(dataset)}")
    sample = dataset[0]
    print(f"  Baseline shape: {sample['baseline'].shape}")
    print(f"  Follow-up shape: {sample['follow_up'].shape}")
    print(f"  Target grade: {sample['target_grade']} (real={sample['grade_is_real']})")
    print(f"  Patient ID: {sample['patient_id']}")
    print("\n✓ Tianjin loader working!")
