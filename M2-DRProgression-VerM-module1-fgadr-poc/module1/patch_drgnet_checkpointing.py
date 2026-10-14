"""
Adds resumable checkpointing to DRG-Net's training scripts (pinned commit 0da1bbe), so a Colab
disconnect costs at most one epoch instead of the whole run. Training itself (model, losses,
optimiser, LR schedule, data, evaluation metric) is unchanged.

What DRG-Net does by default:
  * classification (dr_classification/train.py): saves weights only (best_validation_weights.pt,
    epoch_N.pt every save_interval, final_weights.pt) -- no optimiser/epoch, so no resume.
  * segmentation (train_fgadr.py for FGADR, train.py for IDRiD): saves only when eval AP improves
    (eval every 20 / 40 epochs), so up to 20-40 epochs are lost on a cut-off; its RESUME_MODEL
    restarts the LR schedule; and `range(start_epoch, start_epoch + num_epochs)` runs num_epochs
    MORE epochs after a resume instead of finishing the planned total.

After patching:
  * classification: <save_path>/last_checkpoint.pt every epoch (model + optimiser + epoch + best
    val score). Re-running the same config resumes from it automatically (DRG_CLS_RESUME=0 to
    start over). On resume the per-epoch sampler / loss-weight / warm-up / LR-scheduler steps are
    replayed, so the schedule continues exactly where it stopped. save_path is on Drive in our
    configs, so the checkpoint already survives the runtime.
  * segmentation: results/<run>/last.pth.tar every epoch (+ scheduler state and best AP), copied
    to $DRG_SEG_MIRROR_DIR (Drive) together with the best model and the AP log. Resume with
    DRG_SEG_RESUME=<last.pth.tar>; EPOCHES is the planned TOTAL; eval interval via
    $DRG_SEG_EVAL_EVERY (default = DRG-Net's 20 / 40) and always at the final epoch.
  * Not restored: Python/torch RNG state, so a resumed run is not bit-identical to an
    uninterrupted one (data order / augmentation differ after the resume point).

Usage (idempotent -- safe to run on an already-patched checkout):
    python module1/patch_drgnet_checkpointing.py --drg-root /content/dr-joint-learning
"""
import argparse
import os
import re

MARK = "# [DRProgression checkpoint patch]"


def _sub(text, old, new, label, count=1, regex=False):
    if regex:
        out, n = re.subn(old, new, text, count=count)
    else:
        n = text.count(old)
        out = text.replace(old, new, count)
    if n == 0:
        raise SystemExit(f"Patch anchor not found ({label}) -- is this DRG-Net commit 0da1bbe?")
    return out


def patch_classification(path):
    s = open(path).read()
    if MARK in s:
        return False
    s = _sub(s, "import os\n", "import os\n" + MARK + "\n", "cls header")
    s = _sub(s, """    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset, weighted_sampler)
""", """    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset, weighted_sampler)

    # [DRProgression] resume from last_checkpoint.pt if present
    start_epoch, max_indicator = 1, 0
    last_ckpt = os.path.join(cfg.base.save_path, 'last_checkpoint.pt')
    net = model.module if isinstance(model, nn.DataParallel) else model
    if os.environ.get('DRG_CLS_RESUME', '1') == '1' and os.path.isfile(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device)
        start_epoch, max_indicator = ck['epoch'] + 1, ck['max_indicator']
        # Replay the per-epoch schedule steps of the finished epochs on the fresh optimiser
        # (same order as the loop below), then restore the optimiser state on top.
        for _ in range(1, start_epoch):
            if weighted_sampler:
                weighted_sampler.step()
            if loss_weight_scheduler:
                loss_function.weight = loss_weight_scheduler.step().to(device)
            if warmup_scheduler and not warmup_scheduler.is_finish():
                warmup_scheduler.step()
            if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()) \\
                    and cfg.solver.lr_scheduler != 'reduce_on_plateau':
                lr_scheduler.step()
        net.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        print('[resume] continuing from epoch {} (best val so far {:.4f}) -- {}'.format(
            start_epoch, max_indicator, last_ckpt))
""", "cls resume")
    s = _sub(s, """    model.train()
    max_indicator = 0
""", """    model.train()
""", "cls max_indicator")
    s = _sub(s, "    for epoch in range(1, cfg.train.epochs + 1):\n",
             "    for epoch in range(start_epoch, cfg.train.epochs + 1):\n", "cls loop")
    s = _sub(s, """            logger.add_scalar('learning rate', curr_lr, epoch)
""", """            logger.add_scalar('learning rate', curr_lr, epoch)

        # [DRProgression] resumable checkpoint, written atomically every epoch
        torch.save({'epoch': epoch, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                    'max_indicator': max_indicator}, last_ckpt + '.tmp')
        os.replace(last_ckpt + '.tmp', last_ckpt)
""", "cls save")
    open(path, "w").write(s)
    return True


def patch_segmentation(path, default_eval_every):
    s = open(path).read()
    if MARK in s:
        return False
    s = _sub(s, "import argparse\n", f"""import argparse
{MARK}
import shutil
DRG_EVAL_EVERY = int(os.environ.get('DRG_SEG_EVAL_EVERY', {default_eval_every}))
DRG_MIRROR = os.environ.get('DRG_SEG_MIRROR_DIR') or None
RESUME_BEST_AP = 0.


def _drg_mirror(*paths):
    if DRG_MIRROR:
        os.makedirs(DRG_MIRROR, exist_ok=True)
        for p in paths:
            if os.path.isfile(p):
                shutil.copy(p, os.path.join(DRG_MIRROR, os.path.basename(p)))
""", "seg header")
    s = _sub(s, "    best_ap = 0.\n", "    best_ap = RESUME_BEST_AP\n", "seg best_ap")
    # Planned TOTAL epochs, also after a resume.
    s = _sub(s, r"for epoch in range\(start_epoch, start_epoch \+ num_epochs\):",
             "for epoch in range(start_epoch, num_epochs):", "seg loop", regex=True)
    s = _sub(s, r"\.format\(epoch \+ 1, start_epoch ?\+ ?num_epochs\)", ".format(epoch + 1, num_epochs)",
             "seg print", regex=True)
    # Eval cadence (also matches notebook 03's older `% 5` sed edit) + always at the end.
    s = _sub(s, r"if \(epoch \+ 1\) % \d+ == 0:",
             "if (epoch + 1) % DRG_EVAL_EVERY == 0 or epoch + 1 == num_epochs:", "seg eval", regex=True)
    s = _sub(s, r"'optimizer': g_optimizer\.state_dict\(\)\n(\s*)\}",
             r"'optimizer': g_optimizer.state_dict(),\n\1    'scheduler': g_scheduler.state_dict(),"
             r"\n\1    'best_ap': best_ap,\n\1}", "seg best state", regex=True)
    # Save point: end of each epoch. train.py ends its epoch with logger.flush(); train_fgadr.py
    # with the best-model save -- anchor on the best-model save and append after its block.
    save_last = """
        # [DRProgression] resumable checkpoint every epoch, mirrored to Drive with best + logs
        _last = os.path.join(dir_checkpoint, 'last.pth.tar')
        torch.save({'epoch': epoch, 'step': tot_step_count, 'state_dict': model.state_dict(),
                    'optimizer': g_optimizer.state_dict(), 'scheduler': g_scheduler.state_dict(),
                    'best_ap': best_ap}, _last + '.tmp')
        os.replace(_last + '.tmp', _last)
        _drg_mirror(_last, os.path.join(dir_checkpoint, 'model_' + str(preprocess) + '.pth.tar'),
                    *[f for f in os.listdir('.') if f.endswith('.txt') and 'ap_during_learning' in f],
                    *[os.path.join(dir_checkpoint, f) for f in os.listdir(dir_checkpoint) if f.endswith('.txt')])
"""
    if "        logger.flush()\n        logger.flush()\n" in s:  # train.py (IDRiD)
        s = _sub(s, "        logger.flush()\n        logger.flush()\n",
                 "        logger.flush()\n        logger.flush()\n" + save_last, "seg save (train.py)")
    else:  # train_fgadr.py
        s = _sub(s, r"(torch\.save\(state, \\\n\s*os\.path\.join\(dir_checkpoint, 'model_' \+ (?:str\()?preprocess\)? \+ '\.pth\.tar'\)\)\n)",
                 lambda m: m.group(1) + save_last, "seg save (train_fgadr.py)", regex=True)
    # Resume: restore best AP + scheduler too (DRG-Net restores model/optimiser/epoch/step).
    s = _sub(s, "            start_step = checkpoint['step']\n",
             "            start_step = checkpoint['step']\n"
             "            RESUME_BEST_AP = checkpoint.get('best_ap', 0.)\n", "seg resume best_ap")
    s = _sub(s, r"(\n(\s*)g_scheduler = lr_scheduler\.StepLR\(g_optimizer, step_size=200, gamma=0\.9\)\n)",
             lambda m: m.group(1)
             + f"{m.group(2)}if resume and os.path.isfile(resume) and 'scheduler' in checkpoint:\n"
             + f"{m.group(2)}    g_scheduler.load_state_dict(checkpoint['scheduler'])\n"
             + f"{m.group(2)}    print('[resume] epoch', start_epoch, 'best AP', RESUME_BEST_AP)\n",
             "seg resume scheduler", regex=True)
    s = s.replace("'model_' + preprocess + '.pth.tar'", "'model_' + str(preprocess) + '.pth.tar'")
    open(path, "w").write(s)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drg-root", default="/content/dr-joint-learning")
    args = ap.parse_args()
    for rel, fn in (("dr_classification/train.py", patch_classification),
                    ("dr_segmentation/train_fgadr.py", lambda p: patch_segmentation(p, 20)),
                    ("dr_segmentation/train.py", lambda p: patch_segmentation(p, 40))):
        path = os.path.join(args.drg_root, rel)
        print(("patched   " if fn(path) else "already ok ") + rel)
