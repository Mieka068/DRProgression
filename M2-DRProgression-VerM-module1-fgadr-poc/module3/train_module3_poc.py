"""
Module 3 proof-of-concept training run: EfficientNet-B4 + Weibull survival head, trained on
Tianjin's baseline -> 2-year-follow-up progression labels (see dataset.py).

Independently trained from Module 1's classifier -- no shared weights.

Tianjin's follow-up interval is fixed at 2 years for every patient, and only whether
progression happened by that single checkpoint is observed, not the exact progression date.
This is current-status / case-1 interval-censored data, and the training loss
(weibull_current_status_nll in model.py) is written for that likelihood, not the
exact-event-time Weibull likelihood used for variable-timing data. Since every training
example is observed at the same t=2yr, this calibrates the model's predicted P(progression by
2 years) = 1 - S(2); the predicted shape/scale parameters extrapolate a full survival curve
S(t) to other horizons t != 2 without training signal at those horizons.

"Progression" here means "follow-up ICDR grade > baseline ICDR grade" (any upward stage move).

Evaluation reports AUROC and Brier Score at the single fixed 2-year horizon, not a concordance
index or an integrated Brier score (the metrics DeepDR Plus reports). This isn't a downgrade
from DeepDR Plus's numbers -- concordance/integrated Brier need patients observed at *varying*
follow-up times, which is what makes ranking-by-observed-time meaningful; Tianjin gives every
patient the same fixed 24-month checkpoint, so there's no varying time axis to rank against.
AUROC + Brier Score at that one horizon is the statistically correct tool for this data
structure. DeepDR Plus's published numbers are context for what metadata-augmented models can
achieve with richer temporal data, not a target this number should match.

POC scale: ImageNet-pretrained EfficientNet-B4 backbone (fine-tuned, not trained from scratch),
reduced epochs, patient-level train/val split (see dataset.py::patient_level_split).

Best-epoch saving: after every epoch the model's mean NLL on the held-out val patients is
computed (the training loss, on unseen patients -- a proper scoring rule, steadier than AUROC on
a small val set). The lowest-NLL epoch is written to best-model.ckpt (use this one downstream);
final-model.ckpt stays the last epoch. The reported metrics are best-model's. Caveat: the val
patients both pick the epoch and give the reported metrics, so those metrics are slightly
optimistic -- there is no separate test split at this data size.

Changes on branch AllModuleEdits-PreFinals (CODE_FIXES section 6; each has a flag back to the
earlier behaviour):
  6.1 Patient-level train / val / test = 70 / 15 / 15 (--val-fraction 0.15 --test-fraction 0.15):
      val picks the epoch, the TEST patients (never used for selection) give the reported
      metrics; val metrics are reported alongside. --test-fraction 0 --val-fraction 0.2 = the
      earlier train/val split (and the optimism caveat above applies again).
  6.2 --freeze-blocks N (default 6): the stem and MBConv stages 1-5 keep their ImageNet
      weights; 0 = fine-tune the whole backbone.
  6.3 Train-time augmentation (h-flip, +/-15 deg rotation, mild brightness/contrast;
      --no-augment = off) and AdamW weight decay (--weight-decay, default 1e-2; 0 = Adam as before).
  7.2 --min-baseline-grade 1 keeps only eyes that start at >= mild NPDR (default 0: all).

Round 4 (items R4-6):
  * --report-min-baseline-grade 1 (default): metrics are reported on ALL held-out eyes and on the
    in-scope ones (baseline >= mild NPDR, the manuscript's scope) -- training keeps every eye
    (--min-baseline-grade 0), since the baseline grade is an input and the in-scope subset alone
    has few progression events.
  * 95 % bootstrap intervals (patients resampled) for AUROC and Brier -- with ~20 test events a
    single AUROC is not interpretable on its own.
  * Tabular baselines on the SAME patients (fit on train, scored on val/test): the train event rate
    per clinical baseline grade, and logistic regression on clinical baseline grade (one-hot) +
    log LBS + Module 1's predicted grade. The image model has to beat these to add anything.
  * The output is P(progression by 2 years) -- with every follow-up at t = 2 the Weibull shape is
    not identifiable, so report the 2-year probability, not survival curves at other horizons.

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from resume_utils import load_resume_state, save_resume_state, seed_everything  # noqa: E402


class Config:
    report_min_baseline_grade = 1  # round 4: also report metrics on eyes with baseline >= this
    n_bootstrap = 1000
    tianjin_dir = "../retinal-dr-longitudinal"
    tianjin_module1_cache_path = None
    image_size = 380
    batch_size = 8
    val_fraction = 0.15
    test_fraction = 0.15  # 0 = no test split (earlier behaviour, with val_fraction 0.2)
    freeze_blocks = 6  # stem + MBConv stages 1-5 (3.6M of 17.5M backbone params); 0 = all trained
    augment = True
    weight_decay = 1e-2
    min_baseline_grade = 0
    num_epochs = 10
    lr = 1e-4
    save_dir = "./module3_runs_poc/"
    seed = 42
    resume_state_path = None  # optional per-epoch resume file (see resume_utils.py)


def binary_metrics(probs, events, groups=None, n_bootstrap=0, seed=0):
    """AUROC / Brier / rates for P(progression by 2yr); with n_bootstrap, 95 % intervals from
    resampling patients (groups) with replacement."""
    import numpy as np
    from sklearn.metrics import brier_score_loss, roc_auc_score

    probs, events = np.asarray(probs, dtype=float), np.asarray(events, dtype=float)
    n = len(probs)
    out = {"n": n, "n_events": int(events.sum()) if n else 0,
           "mean_predicted_progression_by_2yr": float(probs.mean()) if n else float("nan"),
           "actual_progression_rate": float(events.mean()) if n else float("nan"),
           "auroc": float(roc_auc_score(events, probs)) if n and len(set(events)) > 1 else float("nan"),
           "brier_score": float(brier_score_loss(events, probs)) if n else float("nan")}
    out["brier_constant_rate"] = out["actual_progression_rate"] * (1 - out["actual_progression_rate"]) if n else float("nan")
    if n_bootstrap and n and len(set(events)) > 1:
        rng = np.random.default_rng(seed)
        groups = np.arange(n) if groups is None else np.asarray(groups)
        uniq = np.unique(groups)
        rows = {g: np.nonzero(groups == g)[0] for g in uniq}
        aucs, briers = [], []
        for _ in range(n_bootstrap):
            idx = np.concatenate([rows[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
            if len(set(events[idx])) > 1:
                aucs.append(roc_auc_score(events[idx], probs[idx]))
            briers.append(brier_score_loss(events[idx], probs[idx]))
        if aucs:
            out["auroc_95ci"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
        out["brier_95ci"] = [float(np.percentile(briers, 2.5)), float(np.percentile(briers, 97.5))]
    return out


def _fmt(m):
    if not m or not m.get("n"):
        return "n=0"
    ci = m.get("auroc_95ci")
    return (f"n={m['n']} ({m['n_events']} events) | AUROC={m['auroc']:.3f}"
            + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
            + f" | Brier={m['brier_score']:.4f} (constant-rate guess: {m['brier_constant_rate']:.4f})")


def tabular_baselines(full_dataset, train_idx, eval_sets, report_min_grade, n_bootstrap, seed):
    """Round 4: non-image baselines fit on the training patients, scored on each eval set.
    eval_sets: {name: indices}. Returns {baseline: {set name: metrics, set name + '_in_scope': metrics}}."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    m = full_dataset.manifest
    def feats(idx):
        rows = m.iloc[idx]
        g = rows["baseline_icdr"].astype(int).to_numpy()
        onehot = np.eye(5)[g]
        lbs = np.log10(rows["lbs"].astype(float).to_numpy() + 1e-5)[:, None]
        m1 = rows["module1_grade"].fillna(-1).astype(float).to_numpy()[:, None]
        return np.hstack([onehot, lbs, m1]), rows["event"].to_numpy(), g, rows["patient_id"].to_numpy()

    Xtr, ytr, gtr, _ = feats(train_idx)
    out = {}
    if len(set(ytr)) < 2:
        return out
    # 1) event rate per clinical baseline grade (train), overall rate for unseen grades
    rate = {k: float(ytr[gtr == k].mean()) for k in np.unique(gtr)}
    overall = float(ytr.mean())
    # 2) logistic regression: grade one-hot + log LBS + Module 1 grade
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
    for name, idx in eval_sets.items():
        if idx is None or not len(idx):
            continue
        X, y, g, pid = feats(idx)
        preds = {"grade_event_rate": np.array([rate.get(k, overall) for k in g]),
                 "logreg_grade_lbs_module1": lr.predict_proba(X)[:, 1]}
        for bname, p in preds.items():
            out.setdefault(bname, {})[name] = binary_metrics(p, y, pid, n_bootstrap, seed)
            keep = g >= report_min_grade
            if report_min_grade > 0 and keep.any():
                out[bname][name + "_in_scope"] = binary_metrics(p[keep], y[keep], pid[keep], n_bootstrap, seed)
    return out


def evaluate(model, loader, device, report_min_grade=0, n_bootstrap=0, seed=0):
    """
    Evaluation at t=2 years (Tianjin's only observed horizon): mean predicted P(progression)
    vs. actual event rate, plus AUROC as a discrimination metric. Not a concordance index --
    concordance needs varying observation/event times, and every Tianjin subject shares the
    same one.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    model.eval()
    all_probs, all_events, all_grades, all_pids = [], [], [], []
    nll_sum, nll_n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            image = batch["baseline"].to(device)
            lbs_idx = batch["lbs_stratum_idx"].to(device)
            baseline_grade = batch["baseline_grade_icdr"].to(device)
            t = batch["time_to_followup"].to(device)
            event = batch["event"]

            shape, scale = model(image, lbs_idx, baseline_grade)
            nll_sum += float(weibull_current_status_nll(shape, scale, t, event.to(device))) * image.size(0)
            nll_n += image.size(0)
            s_t = model.survival_function(shape, scale, t)
            prob_progression = (1 - s_t).cpu()

            all_probs.extend(prob_progression.tolist())
            all_events.extend(event.tolist())
            all_grades.extend(batch["baseline_grade_icdr"].tolist())
            all_pids.extend(batch["patient_id"])
    model.train()

    n = len(all_probs)
    mean_pred = sum(all_probs) / n if n else float("nan")
    actual_rate = sum(all_events) / n if n else float("nan")
    try:
        auroc = roc_auc_score(all_events, all_probs) if len(set(all_events)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
    brier = brier_score_loss(all_events, all_probs) if n else float("nan")

    result = {
        "n": n,
        "mean_predicted_progression_by_2yr": mean_pred,
        "actual_progression_rate": actual_rate,
        "auroc": auroc,
        "brier_score": brier,
        "nll": nll_sum / nll_n if nll_n else float("nan"),
    }
    if n_bootstrap:
        result.update({k: v for k, v in binary_metrics(all_probs, all_events, all_pids, n_bootstrap, seed).items()
                       if k not in result})
    if report_min_grade > 0 and n:
        keep = [g >= report_min_grade for g in all_grades]
        result["in_scope"] = binary_metrics([p for p, k in zip(all_probs, keep) if k],
                                            [e for e, k in zip(all_events, keep) if k],
                                            [q for q, k in zip(all_pids, keep) if k], n_bootstrap, seed)
        result["in_scope"]["min_baseline_grade"] = report_min_grade
    return result


def train(config: Config):
    seed_everything(config.seed)  # was only used for the patient split; now also init/shuffling
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
        min_baseline_grade=config.min_baseline_grade,
    )
    splits = patient_level_split(full_dataset, val_fraction=config.val_fraction, seed=config.seed,
                                 test_fraction=config.test_fraction, augment_train=config.augment)
    train_set, val_set = splits[0], splits[1]
    test_set = splits[2] if len(splits) == 3 else None
    print(f"✓ {len(full_dataset)} usable pairs -> {len(train_set)} train / {len(val_set)} val"
          + (f" / {len(test_set)} test" if test_set is not None else "") + " (patient-level split)"
          + (" | train augmentation on" if config.augment else ""))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_loader = (DataLoader(test_set, batch_size=config.batch_size, shuffle=False, drop_last=False)
                   if test_set is not None and len(test_set) else None)

    print("\n[2/4] Initializing model (ImageNet-pretrained EfficientNet-B4 backbone)...")
    model = EfficientNetWeibullSurvival(pretrained=True, freeze_blocks=config.freeze_blocks).to(device)
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    print(f"  freeze_blocks={config.freeze_blocks}: {sum(p.numel() for p in trainable):,} of "
          f"{sum(p.numel() for p in model.parameters()):,} parameters trained")
    # AdamW with weight_decay 0 is plain Adam (the earlier optimizer).
    optimizer = optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)

    print(f"\n[3/4] Training for {config.num_epochs} epochs (POC scale)...")
    start_time = time.time()
    loss_history = []  # per-epoch {epoch, nll_loss} -- for the NLL-vs-epoch figure
    start_epoch, extra = load_resume_state(config.resume_state_path, {"model": model}, {"opt": optimizer}, device)
    loss_history = extra.get("loss_history", loss_history)
    # Best-epoch tracking; best weights are kept in the resume state so a disconnect after the
    # best epoch doesn't lose them (save_dir is on the runtime disk).
    best = extra.get("best", {"epoch": None, "val_nll": float("inf"), "model": None})
    for epoch in range(start_epoch, config.num_epochs):
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
        val_nll = evaluate(model, val_loader, device)["nll"] if len(val_set) else float("nan")
        loss_history.append({"epoch": epoch + 1, "nll_loss": avg_loss, "val_nll": val_nll})
        is_best = val_nll < best["val_nll"]  # False for nan (empty val set)
        if is_best:
            best = {"epoch": epoch + 1, "val_nll": val_nll,
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            torch.save(best["model"], os.path.join(config.save_dir, "best-model.ckpt"))
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] | NLL: {avg_loss:.4f} | val NLL: {val_nll:.4f}"
              + (" ★ new best -> best-model.ckpt" if is_best else f" | best so far: epoch {best['epoch']}")
              + f" | Elapsed: {elapsed / 60:.1f}min")
        save_resume_state(config.resume_state_path, epoch, {"model": model}, {"opt": optimizer},
                          {"loss_history": loss_history, "best": best}, num_epochs=config.num_epochs)

    ckpt_path = os.path.join(config.save_dir, "final-model.ckpt")
    torch.save(model.state_dict(), ckpt_path)  # last epoch
    best_path = None
    if best["model"] is not None:
        best_path = os.path.join(config.save_dir, "best-model.ckpt")
        torch.save(best["model"], best_path)
        model.load_state_dict(best["model"])
        print(f"\n✓ Best epoch: {best['epoch']} (val NLL {best['val_nll']:.4f}) -> {best_path}; "
              f"final-model.ckpt = last epoch ({config.num_epochs})")

    print(f"\n[4/4] Evaluating {'best-model' if best_path else 'final-model'} on held-out patients...")
    rmg, nb = config.report_min_baseline_grade, config.n_bootstrap
    val_metrics = evaluate(model, val_loader, device, rmg, nb, config.seed)
    test_metrics = evaluate(model, test_loader, device, rmg, nb, config.seed) if test_loader is not None else None
    # The reported metrics: test patients when there is a test split (never used to pick the
    # epoch), else val (the earlier, slightly optimistic setup).
    metrics = test_metrics if test_metrics is not None else val_metrics
    for name, m in (("val (epoch selection)", val_metrics), ("TEST", test_metrics)):
        if m is None:
            continue
        print(
            f"✓ {name}: n={m['n']} | mean predicted P(progression by 2yr)="
            f"{m['mean_predicted_progression_by_2yr']:.3f} | actual rate="
            f"{m['actual_progression_rate']:.3f} | AUROC={m['auroc']:.3f}"
            + (f" [{m['auroc_95ci'][0]:.3f}, {m['auroc_95ci'][1]:.3f}]" if m.get("auroc_95ci") else "")
            + f" | Brier={m['brier_score']:.4f} (constant-rate guess: "
            f"{m['actual_progression_rate'] * (1 - m['actual_progression_rate']):.4f})"
        )
        if m.get("in_scope"):
            print(f"    in scope (baseline >= {rmg}): {_fmt(m['in_scope'])}")

    # Round 4: non-image baselines on the same patients (fit on train).
    _ix = lambda s: list(s.indices) if s is not None else None  # noqa: E731
    baselines = tabular_baselines(full_dataset, _ix(train_set), {"val": _ix(val_set), "test": _ix(test_set)},
                                  rmg, nb, config.seed)
    print("\nTabular baselines (fit on the training patients; the image model has to beat these):")
    for bname, sets in baselines.items():
        for sname, bm in sets.items():
            print(f"  {bname:<26} {sname:<14} {_fmt(bm)}")

    results = {
        "num_epochs": config.num_epochs,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set) if test_set is not None else 0,
        "tianjin_module1_cache_path": config.tianjin_module1_cache_path,
        "best_epoch": best["epoch"],
        "best_val_nll": best["val_nll"] if best["epoch"] else None,
        "best_checkpoint": best_path,
        "metrics_model": "best-model" if best_path else "final-model",
        "metrics_on": "test patients" if test_metrics is not None else "val patients (also used for epoch selection)",
        "metrics": metrics,
        "output_meaning": "P(progression by 2 years) -- every follow-up is at t = 2, so the Weibull shape is "
                          "not identifiable; do not report survival at other horizons",
        "tabular_baselines": baselines,
        "report_min_baseline_grade": config.report_min_baseline_grade,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "loss_history": loss_history,
        "total_time_min": (time.time() - start_time) / 60,
        "seed": config.seed,
        "val_fraction": config.val_fraction,
        "test_fraction": config.test_fraction,
        "freeze_blocks": config.freeze_blocks,
        "augment": config.augment,
        "weight_decay": config.weight_decay,
        "min_baseline_grade": config.min_baseline_grade,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
    }
    # Saved so a new image's LBS can be classified into low/medium/high after training ends,
    # via module1/compute_lbs.py::classify_lbs_stratum. Keys cast to plain int and values to
    # list since JSON can't serialize numpy scalars or tuples directly.
    results["lbs_thresholds_by_grade"] = {
        int(k): list(v) for k, v in full_dataset.lbs_thresholds_by_grade.items()
    }
    results_path = os.path.join(config.save_dir, "poc_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Wrote {results_path}")
    print(f"✓ Final (last-epoch) model saved to {ckpt_path}")
    if best_path:
        print(f"✓ Best-epoch model saved to {best_path} <- use this one downstream")


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
    parser.add_argument("--val-fraction", type=float, default=Config.val_fraction)
    parser.add_argument("--test-fraction", type=float, default=Config.test_fraction,
                        help="Patients held out for the reported metrics (0 = no test split, earlier)")
    parser.add_argument("--freeze-blocks", type=int, default=Config.freeze_blocks,
                        help="Frozen EfficientNet-B4 feature blocks (0 = fine-tune all, earlier)")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="No train-time augmentation (earlier behaviour)")
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay,
                        help="AdamW weight decay (0 = Adam, earlier)")
    parser.add_argument("--min-baseline-grade", type=int, default=Config.min_baseline_grade,
                        help="1 = only eyes starting at >= mild NPDR (item 7.2); 0 = all")
    parser.add_argument("--report-min-baseline-grade", type=int, default=Config.report_min_baseline_grade,
                        help="Also report metrics on eyes with baseline >= this (1 = the manuscript's scope); 0 = off")
    parser.add_argument("--n-bootstrap", type=int, default=Config.n_bootstrap,
                        help="Patient-level bootstrap resamples for the 95 %% intervals (0 = off)")
    parser.add_argument("--save-dir", default=Config.save_dir)
    parser.add_argument("--resume-state", default=None,
                        help="Per-epoch resume file (put it on Drive); re-running with the same path "
                             "continues after the last finished epoch")
    args = parser.parse_args()

    cfg = Config()
    cfg.tianjin_dir = args.tianjin_dir
    cfg.tianjin_module1_cache_path = args.tianjin_module1_cache
    cfg.num_epochs = args.num_epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.resume_state_path = args.resume_state
    cfg.val_fraction, cfg.test_fraction = args.val_fraction, args.test_fraction
    cfg.freeze_blocks, cfg.augment, cfg.weight_decay = args.freeze_blocks, args.augment, args.weight_decay
    cfg.min_baseline_grade = args.min_baseline_grade
    cfg.report_min_baseline_grade = args.report_min_baseline_grade
    cfg.n_bootstrap = args.n_bootstrap
    cfg.save_dir = args.save_dir

    train(cfg)
