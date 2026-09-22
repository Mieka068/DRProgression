"""
Module 2 figures (Task O): per-cascade-step FID/PSNR/SSIM line chart, G/D/reconstruction loss
curves, and a Module 1 consistency AUC bar chart against DRForecastGAN's published reference.
Reads the JSON files train_module2_poc.py and evaluate_trajectory.py already write -- no new
training or inference here.

Usage:
    python plot_module2_figures.py per-step-quality \
        --trajectory-eval-json ./trajectory_eval_results.json \
        --out ./figures/module2_per_step_quality.png

    python plot_module2_figures.py loss-curves \
        --poc-results-json ./DRForestGAN-v2/stargan/models_poc/poc_results.json \
        --out ./figures/module2_loss_curves.png

    python plot_module2_figures.py consistency-auc \
        --trajectory-eval-json ./trajectory_eval_results.json \
        --out ./figures/module2_consistency_auc.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# DRForecastGAN's own published re-grading consistency AUC (see evaluate_trajectory.py's
# module docstring / Task L) -- drawn as a reference line so "are we in the same ballpark as
# prior work" is answerable at a glance.
DRFORECASTGAN_AUC_INTERNAL = 0.87
DRFORECASTGAN_AUC_EXTERNAL = 0.85


def _step_sort_key(key):
    return int(key.split("_")[-1])


def plot_per_step_quality(trajectory_eval_json: str, out_path: str):
    with open(trajectory_eval_json) as f:
        results = json.load(f)
    quality = results["quality"]
    steps = sorted(quality.keys(), key=_step_sort_key)
    step_nums = [_step_sort_key(s) for s in steps]
    psnr = [quality[s]["mean_psnr"] for s in steps]
    ssim = [quality[s]["mean_ssim"] for s in steps]
    fid = [quality[s]["fid"] for s in steps]  # may contain None below min_n_for_fid
    n_pairs = [quality[s]["n_pairs"] for s in steps]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(step_nums, psnr, marker="o", color="tab:blue", label="PSNR")
    ax1.plot(step_nums, ssim, marker="s", color="tab:green", label="SSIM")
    ax1.set_xlabel("Cascade step (stages ahead of baseline)")
    ax1.set_ylabel("PSNR / SSIM")
    ax1.set_xticks(step_nums)

    ax2 = ax1.twinx()
    fid_steps = [s for s, v in zip(step_nums, fid) if v is not None]
    fid_vals = [v for v in fid if v is not None]
    if fid_vals:
        ax2.plot(fid_steps, fid_vals, marker="^", color="tab:red", label="FID")
    ax2.set_ylabel("FID (lower is better)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("Module 2: quality vs. cascade step\n(n_pairs per step: "
                  + ", ".join(f"{n}={c}" for n, c in zip(step_nums, n_pairs)) + ")", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def plot_loss_curves(poc_results_json: str, out_path: str):
    with open(poc_results_json) as f:
        results = json.load(f)
    history = results.get("loss_history")
    if not history:
        raise ValueError(f"{poc_results_json} has no 'loss_history' -- retrain with the "
                          "current train_module2_poc.py to populate it.")
    epochs = [h["epoch"] for h in history]
    d_loss = [h["d_loss"] for h in history]
    g_loss = [h["g_loss"] for h in history]
    rec_loss = [h["rec_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, d_loss, label="D loss")
    ax.plot(epochs, g_loss, label="G loss (adv + rec)")
    ax.plot(epochs, rec_loss, label="G reconstruction (L1) loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Module 2 training loss (n_epochs={results.get('num_epochs')}, "
                 f"POC-scale -- see train_module2_poc.py docstring)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def plot_consistency_auc(trajectory_eval_json: str, out_path: str, primary="auc"):
    with open(trajectory_eval_json) as f:
        results = json.load(f)
    consistency = results["module1_consistency"]
    steps = sorted(consistency.keys(), key=_step_sort_key)
    step_nums = [_step_sort_key(s) for s in steps]

    if primary == "auc":
        heights = [consistency[s]["module1_consistency_auc"] for s in steps]
        ylabel = "Module 1 re-grading consistency AUC"
    else:
        heights = [consistency[s]["module1_agrees_with_target_stage_rate"] for s in steps]
        ylabel = "Module 1 raw agreement rate"

    plot_steps = [s for s, h in zip(step_nums, heights) if h is not None]
    plot_heights = [h for h in heights if h is not None]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(plot_steps, plot_heights, color="tab:purple", label="This run")
    if primary == "auc":
        ax.axhline(DRFORECASTGAN_AUC_INTERNAL, color="tab:orange", linestyle="--",
                   label=f"DRForecastGAN internal ({DRFORECASTGAN_AUC_INTERNAL})")
        ax.axhline(DRFORECASTGAN_AUC_EXTERNAL, color="tab:red", linestyle="--",
                   label=f"DRForecastGAN external ({DRFORECASTGAN_AUC_EXTERNAL})")
    ax.set_xlabel("Cascade step")
    ax.set_ylabel(ylabel)
    ax.set_xticks(step_nums)
    ax.set_ylim(0, 1)
    ax.set_title("Module 2: Module 1 re-grading consistency per cascade step")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_q = sub.add_parser("per-step-quality")
    p_q.add_argument("--trajectory-eval-json", required=True)
    p_q.add_argument("--out", required=True)

    p_l = sub.add_parser("loss-curves")
    p_l.add_argument("--poc-results-json", required=True)
    p_l.add_argument("--out", required=True)

    p_c = sub.add_parser("consistency-auc")
    p_c.add_argument("--trajectory-eval-json", required=True)
    p_c.add_argument("--primary", choices=["auc", "rate"], default="auc")
    p_c.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "per-step-quality":
        plot_per_step_quality(args.trajectory_eval_json, args.out)
    elif args.command == "loss-curves":
        plot_loss_curves(args.poc_results_json, args.out)
    elif args.command == "consistency-auc":
        plot_consistency_auc(args.trajectory_eval_json, args.out, primary=args.primary)


if __name__ == "__main__":
    main()
