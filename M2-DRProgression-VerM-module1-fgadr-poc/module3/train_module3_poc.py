"""
Module 3 proof-of-concept training run: EfficientNet-B4 + Weibull survival head, trained on
Tianjin's baseline -> 2-year-follow-up progression labels (see dataset.py).

Independently trained from Module 1's classifier -- no shared weights (docs/DECISIONS.md).

HONEST FRAMING -- read before presenting any result from this script (docs/DECISIONS.md,
docs/ROADMAP.md's "Consequence of the Tianjin timing structure" note):
  - Tianjin's follow-up interval is a FIXED 2 years for every patient, not a variable one, and
    we only ever observe whether progression happened by that single checkpoint -- not the
    exact progression date. This is current-status / case-1 interval-censored data, and the
    training loss (weibull_current_status_nll in model.py) is written for exactly that
    likelihood, not the exact-event-time Weibull likelihood appropriate for variable-timing
    data (e.g. a future DCCT/EDIC or Moorfields source).
  - Because every training example is observed at the SAME t=2yr, this can only ever calibrate
    the model's predicted P(progression by 2 years) = 1 - S(2) well. The predicted shape/scale
    parameters extrapolate a full survival curve S(t) to other horizons t != 2 WITHOUT any
    training signal at those horizons -- report S(2)/"probability of progression by 2 years"
    as the result, not curves at other time points, unless/until a variable-timing source is
    available to actually supervise them.
  - Report predictions as population-level probabilistic estimates with the AUROC/calibration
    numbers below, not as per-patient point predictions presented as certain (docs/DECISIONS.md's
    "two-visit data means honest framing" rule).
  - "Progression" here means "follow-up ICDR grade > baseline ICDR grade" (any upward stage
    move). That is not necessarily the same clinical claim as "progressed to sight-threatening
    disease" -- flag to the adviser/Dr. Atienza if the thesis needs a stricter definition
    (e.g. reaching PDR specifically).

POC scale: ImageNet-pretrained EfficientNet-B4 backbone (fine-tuned, not trained from scratch
-- Tianjin's post-exclusion N is too small for that), reduced epochs, patient-level train/val
split (see dataset.py::patient_level_split).

Usage (Colab, after notebook 05 has built module1_outputs_tianjin.pt):
    python train_module3_poc.py \
        --tianjin-dir /content/data/retinal-dr-longitudinal \
        --tianjin-module1-cache /content/drive/MyDrive/.../module1_outputs_tianjin.pt \
        --num-epochs 10
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import TianjinSurvivalDataset, patient_level_split  # noqa: E402
from model import EfficientNetWeibullSurvival, weibull_current_status_nll  # noqa: E402


class Config:
    tianjin_dir = "../retinal-dr-longitudinal"
    tianjin_module1_cache_path = None
    image_size = 380
    batch_size = 8
    val_fraction = 0.2
    num_epochs = 10
    lr = 1e-4
    save_dir = "./module3_runs_poc/"
    seed = 42


def evaluate(model, loader, device):
    """
    POC eval AT t=2 years (Tianjin's only observed horizon): mean predicted P(progression) vs.
    actual event rate, plus AUROC as a discrimination metric. This is NOT a concordance index
    -- concordance needs varying observation/event times to compare risk orderings across
    different horizons, and every Tianjin subject shares the same one (see module docstring).
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs, all_events = [], []
    with torch.no_grad():
        for batch in loader:
            image = batch["baseline"].to(device)
            lbs_idx = batch["lbs_stratum_idx"].to(device)
            baseline_grade = batch["baseline_grade_icdr"].to(device)
            t = batch["time_to_followup"].to(device)
            event = batch["event"]

            shape, scale = model(image, lbs_idx, baseline_grade)
            s_t = model.survival_function(shape, scale, t)
            prob_progression = (1 - s_t).cpu()

            all_probs.extend(prob_progression.tolist())
            all_events.extend(event.tolist())
    model.train()

    n = len(all_probs)
    mean_pred = sum(all_probs) / n if n else float("nan")
    actual_rate = sum(all_events) / n if n else float("nan")
    try:
        auroc = roc_auc_score(all_events, all_probs) if len(set(all_events)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")

    return {
        "n": n,
        "mean_predicted_progression_by_2yr": mean_pred,
        "actual_progression_rate": actual_rate,
        "auroc": auroc,
    }


def train(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Module 3 POC training (EfficientNet-B4 + Weibull survival head)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"tianjin_dir: {config.tianjin_dir}")
    print(f"tianjin_module1_cache_path: {config.tianjin_module1_cache_path}")

    os.makedirs(config.save_dir, exist_ok=True)

    print("\n[1/4] Loading Tianjin survival dataset...")
    full_dataset = TianjinSurvivalDataset(
        config.tianjin_dir,
        module1_cache_path=config.tianjin_module1_cache_path,
        image_size=config.image_size,
    )
    train_set, val_set = patient_level_split(full_dataset, val_fraction=config.val_fraction, seed=config.seed)
    print(f"✓ {len(full_dataset)} usable pairs -> {len(train_set)} train / {len(val_set)} val (patient-level split)")

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, drop_last=False)

    print("\n[2/4] Initializing model (ImageNet-pretrained EfficientNet-B4 backbone)...")
    model = EfficientNetWeibullSurvival(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    print(f"\n[3/4] Training for {config.num_epochs} epochs (POC scale -- see module docstring)...")
    start_time = time.time()
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            image = batch["baseline"].to(device)
            lbs_idx = batch["lbs_stratum_idx"].to(device)
            baseline_grade = batch["baseline_grade_icdr"].to(device)
            t = batch["time_to_followup"].to(device)
            event = batch["event"].to(device)

            shape, scale = model(image, lbs_idx, baseline_grade)
            loss = weibull_current_status_nll(shape, scale, t, event)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) else float("nan")
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] | NLL: {avg_loss:.4f} | Elapsed: {elapsed / 60:.1f}min")

    print("\n[4/4] Evaluating on held-out patients (see module docstring's honest-framing note)...")
    metrics = evaluate(model, val_loader, device)
    print(
        f"✓ n={metrics['n']} | mean predicted P(progression by 2yr)="
        f"{metrics['mean_predicted_progression_by_2yr']:.3f} | actual rate="
        f"{metrics['actual_progression_rate']:.3f} | AUROC={metrics['auroc']:.3f}"
    )

    ckpt_path = os.path.join(config.save_dir, "final-model.ckpt")
    torch.save(model.state_dict(), ckpt_path)
    results = {
        "num_epochs": config.num_epochs,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "tianjin_module1_cache_path": config.tianjin_module1_cache_path,
        "metrics": metrics,
        "total_time_min": (time.time() - start_time) / 60,
    }
    results_path = os.path.join(config.save_dir, "poc_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Wrote {results_path}")
    print(f"✓ Final model saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tianjin-dir", default=Config.tianjin_dir)
    parser.add_argument(
        "--tianjin-module1-cache",
        required=True,
        help="Required -- Module 3's LBS input comes from this cache, unlike Module 2's grade "
        "conditioning which tolerates a missing cache.",
    )
    parser.add_argument("--num-epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    args = parser.parse_args()

    cfg = Config()
    cfg.tianjin_dir = args.tianjin_dir
    cfg.tianjin_module1_cache_path = args.tianjin_module1_cache
    cfg.num_epochs = args.num_epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr

    train(cfg)
