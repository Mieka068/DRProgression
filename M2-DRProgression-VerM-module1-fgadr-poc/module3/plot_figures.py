"""
Module 3 figures (Task O): calibration plot, ROC curve, NLL-vs-epoch training curve, and a
few representative per-patient Weibull survival curves across LBS strata.

The calibration/ROC/survival-curve figures need per-patient predictions, which
train_module3_poc.py's own evaluate() only ever aggregates -- so this script rebuilds the
SAME held-out val split (same seed/val_fraction, read back from poc_results.json) and reruns
the saved checkpoint over it, the same "load a trained checkpoint and recompute" pattern
visualize_trajectory.py already uses for Module 3. This never re-trains anything.

Usage:
    python plot_figures.py \
        --tianjin-dir /content/data/retinal-dr-longitudinal \
        --tianjin-module1-cache /content/drive/.../module1_outputs_tianjin.pt \
        --checkpoint ./module3_runs_poc/final-model.ckpt \
        --poc-results-json ./module3_runs_poc/poc_results.json \
        --out-dir ./figures/
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import TianjinSurvivalDataset, patient_level_split  # noqa: E402
from model import EfficientNetWeibullSurvival  # noqa: E402

STRATUM_NAMES = ["low", "medium", "high"]  # matches module3/dataset.py::STRATUM_TO_IDX ordering


def load_val_predictions(tianjin_dir, tianjin_module1_cache, checkpoint_path, poc_results_json, device):
    with open(poc_results_json) as f:
        results = json.load(f)
    seed = results.get("seed", 42)
    val_fraction = results.get("val_fraction", 0.2)
    image_size = results.get("image_size", 380)
    batch_size = results.get("batch_size", 8)

    full_dataset = TianjinSurvivalDataset(
        tianjin_dir, module1_cache_path=tianjin_module1_cache, image_size=image_size
    )
    _, val_set = patient_level_split(full_dataset, val_fraction=val_fraction, seed=seed)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

    model = EfficientNetWeibullSurvival(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    records = []
    with torch.no_grad():
        for batch in val_loader:
            image = batch["baseline"].to(device)
            lbs_idx = batch["lbs_stratum_idx"].to(device)
            baseline_grade = batch["baseline_grade_icdr"].to(device)
            t = batch["time_to_followup"].to(device)
            event = batch["event"]

            shape, scale = model(image, lbs_idx, baseline_grade)
            s_t = model.survival_function(shape, scale, t)
            prob_progression = (1 - s_t).cpu()

            for i in range(image.size(0)):
                records.append({
                    "patient_id": batch["patient_id"][i],
                    "prob": float(prob_progression[i]),
                    "event": float(event[i]),
                    "baseline_grade": int(baseline_grade[i].item()),
                    "lbs_stratum_idx": int(lbs_idx[i].item()),
                    "shape": float(shape[i].item()),
                    "scale": float(scale[i].item()),
                })
    return records


def plot_calibration(records, out_path, n_bins=10):
    probs = np.array([r["prob"] for r in records])
    events = np.array([r["event"] for r in records])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)

    bin_mean_pred, bin_obs_rate, bin_n = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_mean_pred.append(probs[mask].mean())
        bin_obs_rate.append(events[mask].mean())
        bin_n.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(bin_mean_pred, bin_obs_rate, marker="o", color="tab:blue", label="Model")
    for x, y, n in zip(bin_mean_pred, bin_obs_rate, bin_n):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Predicted P(progression by 2yr)")
    ax.set_ylabel("Observed progression rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Module 3 calibration at 24-month horizon (n={len(records)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def plot_roc(records, out_path):
    probs = np.array([r["prob"] for r in records])
    events = np.array([r["event"] for r in records])
    if len(set(events.tolist())) < 2:
        raise ValueError("Only one class present in val events -- ROC curve is undefined.")
    fpr, tpr, _ = roc_curve(events, probs)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(events, probs)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color="tab:blue", label=f"Module 3 (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Module 3 ROC at 24-month horizon (n={len(records)})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def plot_nll_vs_epoch(poc_results_json, out_path):
    with open(poc_results_json) as f:
        results = json.load(f)
    history = results.get("loss_history")
    if not history:
        raise ValueError(f"{poc_results_json} has no 'loss_history' -- retrain with the "
                          "current train_module3_poc.py to populate it.")
    epochs = [h["epoch"] for h in history]
    nll = [h["nll_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, nll, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Current-status Weibull NLL")
    ax.set_title("Module 3 training loss vs. epoch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def plot_survival_curves(records, out_path, baseline_grade=None, t_max=4.0, n_points=200):
    """
    S(t) for one representative patient per LBS stratum (low/medium/high), all sharing the
    same baseline_grade (defaults to whichever grade has patients in every stratum, so the
    comparison isolates the LBS effect rather than mixing it with a grade effect) -- a
    concrete way to show the stratification is doing something, even without multi-horizon
    validation (see IMPLEMENTATION_PLAN_2.md Task O).
    """
    if baseline_grade is None:
        grades_with_all_strata = []
        for g in sorted(set(r["baseline_grade"] for r in records)):
            strata_present = set(r["lbs_stratum_idx"] for r in records if r["baseline_grade"] == g)
            if strata_present == {0, 1, 2}:
                grades_with_all_strata.append(g)
        if not grades_with_all_strata:
            raise ValueError("No single baseline grade has patients in all 3 LBS strata in "
                              "the val set -- pass --baseline-grade explicitly and accept a "
                              "partial comparison.")
        baseline_grade = grades_with_all_strata[0]

    t = np.linspace(0.01, t_max, n_points)
    fig, ax = plt.subplots(figsize=(7, 5))
    for stratum_idx, stratum_name in enumerate(STRATUM_NAMES):
        candidates = [r for r in records if r["baseline_grade"] == baseline_grade
                      and r["lbs_stratum_idx"] == stratum_idx]
        if not candidates:
            continue
        r = candidates[0]  # first match -- representative, not cherry-picked for effect size
        s_t = np.exp(-np.power(t / r["scale"], r["shape"]))
        ax.plot(t, s_t, label=f"{stratum_name} LBS (patient {r['patient_id']})")
    ax.axvline(2.0, color="gray", linestyle=":", linewidth=0.8, label="2yr (observed horizon)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("S(t) = P(no progression by t)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Module 3: example survival curves by LBS stratum (baseline ICDR grade {baseline_grade})\n"
                 "Note: only t=2yr has training signal -- other horizons are the fitted "
                 "Weibull's extrapolation.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tianjin-dir", required=True)
    parser.add_argument("--tianjin-module1-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--poc-results-json", required=True)
    parser.add_argument("--baseline-grade", type=int, default=None,
                         help="For the survival-curve figure; auto-picked if omitted")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Rebuilding val split and running the trained checkpoint over it...")
    records = load_val_predictions(
        args.tianjin_dir, args.tianjin_module1_cache, args.checkpoint, args.poc_results_json, args.device
    )
    print(f"✓ {len(records)} val predictions")

    plot_calibration(records, os.path.join(args.out_dir, "module3_calibration.png"))
    plot_roc(records, os.path.join(args.out_dir, "module3_roc.png"))
    plot_nll_vs_epoch(args.poc_results_json, os.path.join(args.out_dir, "module3_nll_vs_epoch.png"))
    plot_survival_curves(
        records, os.path.join(args.out_dir, "module3_survival_curves.png"), baseline_grade=args.baseline_grade
    )


if __name__ == "__main__":
    main()
