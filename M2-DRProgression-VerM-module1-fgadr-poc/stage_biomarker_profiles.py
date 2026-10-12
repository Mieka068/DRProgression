"""
Per-stage biomarker profiles from a cross-sectional graded dataset (EyePACS, APTOS, ...).

Idea: a non-longitudinal dataset can't show one eye progressing, but with thousands of images
per ICDR stage it shows what the *population* at each stage looks like. Running Module 1's
lesion segmenters over every image (module1/apply_to_progression_data.py, which now records
per-lesion `lesion_stats`) and grouping by the real CSV grade gives:

  1. stage profiles     -- per stage, the distribution of each biomarker (mean/std/quartiles in
                           a log-transformed space, plus raw medians and lesion prevalence).
                           These are what train_module2_nonlong_pretrain.py pulls synthesized
                           stage-k images towards (biomarker-profile loss).
  2. transition shifts  -- for each adjacent pair k -> k+1, which biomarkers change the most
                           (Cohen's d + Mann-Whitney U), i.e. what a progression step should
                           add. Ranked, so the top entries are the biomarkers that best
                           separate the two stages.
  3. shared biomarkers  -- for each adjacent pair, biomarkers whose distribution barely moves
                           (|d| < 0.2): present at both stages, so not what defines the step.

Biomarker vector (N_BIOMARKERS = 9), per image:
    for each lesion in EX, HE, MA, SE:
        <lesion>_area   log10(area_frac + 1e-5)   area fraction of the retinal FOV
        <lesion>_count  log1p(n connected components)
    lbs                 log10(LBS + 1e-5)          all lesions combined
Log transforms because lesion areas/counts are heavily right-skewed (most stage 0/1 eyes are
near zero, PDR eyes span orders of magnitude). A lesion whose segmenter wasn't trained when the
cache was built is NaN for that image and is masked out everywhere downstream.

Caveat to state in the write-up: biomarkers come from Module 1's *predicted* masks, so the
profiles inherit Module 1's segmentation error; they describe what Module 1 sees at each stage,
not manual annotations.

Usage:
    python stage_biomarker_profiles.py \
        --module1-cache /content/drive/MyDrive/.../module1_outputs_eyepacs.pt \
        --labels-csv /content/data/eyepacs/trainLabels.csv --image-col image --label-col level \
        --out ./stage_biomarker_profiles.json --fig ./stage_biomarker_profiles.png
"""
import argparse
import json

import numpy as np

LESIONS = ["EX", "HE", "MA", "SE"]
BIOMARKER_NAMES = [f"{l}_{k}" for l in LESIONS for k in ("area", "count")] + ["lbs"]
N_BIOMARKERS = len(BIOMARKER_NAMES)
AREA_EPS = 1e-5
SHARED_D_THRESHOLD = 0.2  # |Cohen's d| below this = "small" effect (Cohen 1988)


def biomarker_vector(entry):
    """
    Module 1 cache entry -> float32[N_BIOMARKERS] in the transformed space, NaN for lesions
    that weren't segmented. Returns None if the entry predates lesion_stats (old caches).
    """
    stats = entry.get("lesion_stats")
    if stats is None:
        return None
    vec = np.full(N_BIOMARKERS, np.nan, dtype=np.float32)
    for i, lesion in enumerate(LESIONS):
        s = stats.get(lesion)
        if s is not None:
            vec[2 * i] = np.log10(s["area_frac"] + AREA_EPS)
            vec[2 * i + 1] = np.log1p(s["count"])
    if stats:
        vec[-1] = np.log10(float(entry["lbs"]) + AREA_EPS)
    return vec


def _cohens_d(a, b):
    """Effect size of b relative to a (positive = higher at b), pooled SD."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


def build_profiles(cache, labels, num_stages=5):
    """
    Args:
        cache: Module 1 cache dict {image_id: entry}.
        labels: dict {image_id: icdr_grade} (real labels; Module 1's predicted grade is NOT
            used for grouping).
    Returns:
        JSON-serialisable dict with 'features', 'stages', 'transitions'.
    """
    from scipy.stats import mannwhitneyu

    per_stage = {g: [] for g in range(num_stages)}
    raw = {g: {"area": {l: [] for l in LESIONS}, "count": {l: [] for l in LESIONS}, "lbs": []}
           for g in range(num_stages)}
    n_no_stats = 0
    for image_id, grade in labels.items():
        entry = cache.get(image_id)
        if entry is None or grade not in per_stage:
            continue
        vec = biomarker_vector(entry)
        if vec is None:
            n_no_stats += 1
            continue
        per_stage[grade].append(vec)
        raw[grade]["lbs"].append(float(entry["lbs"]))
        for l, s in entry["lesion_stats"].items():
            if l in LESIONS:
                raw[grade]["area"][l].append(s["area_frac"])
                raw[grade]["count"][l].append(s["count"])
    if n_no_stats:
        print(f"⚠ {n_no_stats} cache entries have no lesion_stats (cache built before per-lesion "
              "stats were added?) -- re-run module1/apply_to_progression_data.py to include them")

    arrays = {g: np.stack(v) if v else np.zeros((0, N_BIOMARKERS), np.float32) for g, v in per_stage.items()}

    stages = {}
    for g in range(num_stages):
        X = arrays[g]
        feats = {}
        for j, name in enumerate(BIOMARKER_NAMES):
            col = X[:, j][np.isfinite(X[:, j])] if len(X) else np.array([])
            if len(col) == 0:
                feats[name] = None
                continue
            feats[name] = {
                "n": int(len(col)),
                "mean": float(col.mean()),
                "std": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
                "median": float(np.median(col)),
                "q25": float(np.percentile(col, 25)),
                "q75": float(np.percentile(col, 75)),
            }
        raw_summary = {
            "lbs_median": float(np.median(raw[g]["lbs"])) if raw[g]["lbs"] else None,
            "lesions": {
                l: {
                    "area_frac_median": float(np.median(raw[g]["area"][l])),
                    "count_median": float(np.median(raw[g]["count"][l])),
                    "prevalence": float(np.mean(np.asarray(raw[g]["count"][l]) > 0)),
                }
                for l in LESIONS if raw[g]["area"][l]
            },
        }
        # Characteristic biomarkers: lesions present in most eyes at this stage.
        characteristic = sorted(
            (l for l, v in raw_summary["lesions"].items() if v["prevalence"] >= 0.5),
            key=lambda l: -raw_summary["lesions"][l]["prevalence"],
        )
        stages[str(g)] = {"n_images": int(len(X)), "features": feats, "raw": raw_summary,
                          "characteristic_lesions": characteristic}

    transitions = {}
    for g in range(num_stages - 1):
        A, B = arrays[g], arrays[g + 1]
        rows = []
        for j, name in enumerate(BIOMARKER_NAMES):
            a = A[:, j][np.isfinite(A[:, j])] if len(A) else np.array([])
            b = B[:, j][np.isfinite(B[:, j])] if len(B) else np.array([])
            if len(a) < 2 or len(b) < 2:
                continue
            try:
                p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except ValueError:  # all values identical
                p = 1.0
            rows.append({"feature": name, "cohens_d": _cohens_d(a, b),
                         "delta_mean": float(b.mean() - a.mean()), "mannwhitney_p": p})
        rows.sort(key=lambda r: -abs(r["cohens_d"]) if np.isfinite(r["cohens_d"]) else 0)
        transitions[f"{g}->{g + 1}"] = {
            "ranked_shifts": rows,
            "shared_biomarkers": [r["feature"] for r in rows if abs(r["cohens_d"]) < SHARED_D_THRESHOLD],
        }

    return {"features": BIOMARKER_NAMES, "transform": "area: log10(x+1e-5); count: log1p; lbs: log10(x+1e-5)",
            "stages": stages, "transitions": transitions}


def profile_tensors(profiles, num_stages=5, min_std=0.05):
    """
    (mu, sigma) float32 arrays of shape [num_stages, N_BIOMARKERS] for the training loss.
    Stages/features with no data get NaN, which the loss masks out. sigma is floored at
    min_std so a near-constant feature (e.g. stage 0 SE area) doesn't blow the z-score up.
    """
    mu = np.full((num_stages, N_BIOMARKERS), np.nan, np.float32)
    sd = np.full((num_stages, N_BIOMARKERS), np.nan, np.float32)
    for g in range(num_stages):
        feats = profiles["stages"].get(str(g), {}).get("features", {})
        for j, name in enumerate(BIOMARKER_NAMES):
            f = feats.get(name)
            if f:
                mu[g, j] = f["mean"]
                sd[g, j] = max(f["std"], min_std)
    return mu, sd


def plot_profiles(cache, labels, out_path, num_stages=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_stage = {g: [] for g in range(num_stages)}
    for image_id, grade in labels.items():
        entry = cache.get(image_id)
        if entry is not None and grade in per_stage:
            vec = biomarker_vector(entry)
            if vec is not None:
                per_stage[grade].append(vec)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for j, (ax, name) in enumerate(zip(axes.flat, BIOMARKER_NAMES)):
        data = []
        for g in range(num_stages):
            col = np.array([v[j] for v in per_stage[g]])
            data.append(col[np.isfinite(col)] if len(col) else np.array([]))
        ax.boxplot([d if len(d) else [np.nan] for d in data], showfliers=False)
        ax.set_xticks(range(1, num_stages + 1))
        ax.set_xticklabels([str(g) for g in range(num_stages)])
        ax.set_title(name)
        ax.set_xlabel("ICDR stage")
    fig.suptitle("Module 1 biomarkers per DR stage (cross-sectional)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✓ Saved {out_path}")


def print_summary(profiles):
    for g, s in profiles["stages"].items():
        print(f"  stage {g}: n={s['n_images']}, characteristic lesions: {s['characteristic_lesions'] or '-'}")
    for t, v in profiles["transitions"].items():
        top = ", ".join(f"{r['feature']} (d={r['cohens_d']:+.2f})" for r in v["ranked_shifts"][:3])
        print(f"  {t}: biggest shifts {top or '-'}; shared: {v['shared_biomarkers'] or '-'}")


def main():
    import torch

    from crosssectional_dataset import load_labels

    parser = argparse.ArgumentParser()
    parser.add_argument("--module1-cache", required=True,
                        help="module1/apply_to_progression_data.py cache over the cross-sectional images")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--image-col", default="image", help="EyePACS: image, APTOS: id_code")
    parser.add_argument("--label-col", default="level", help="EyePACS: level, APTOS: diagnosis")
    parser.add_argument("--out", default="./stage_biomarker_profiles.json")
    parser.add_argument("--fig", default=None, help="Optional PNG of per-stage biomarker boxplots")
    args = parser.parse_args()

    cache = torch.load(args.module1_cache, weights_only=False)
    df = load_labels(args.labels_csv, args.image_col, args.label_col)
    # Cache keys have no extension; tolerate CSVs that include one.
    labels = {str(i).rsplit(".", 1)[0] if str(i) not in cache else str(i): int(g)
              for i, g in zip(df["image_id"], df["grade"])}
    matched = sum(1 for i in labels if i in cache)
    print(f"✓ {matched}/{len(labels)} labelled images found in the Module 1 cache")

    profiles = build_profiles(cache, labels)
    with open(args.out, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"✓ Wrote {args.out}")
    print_summary(profiles)
    if args.fig:
        plot_profiles(cache, labels, args.fig)


if __name__ == "__main__":
    main()
