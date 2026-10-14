"""
Adds checkpoint/resume to a cloned DRG-Net repo (pinned commit 0da1bbe) so long runs survive a
Colab disconnect. Applied by notebook 11; idempotent (re-running is a no-op) and it fails loudly
if DRG-Net's code doesn't look as expected, rather than silently patching the wrong place.

What DRG-Net does on its own, and what this adds:

dr_classification/train.py -- DRG-Net saves only model weights (best_validation_weights.pt,
  epoch_N.pt every 5 epochs), no optimizer / LR schedule / epoch, so a run cannot be resumed.
  Patch: after every epoch, write <save_path>/resume_state.pt (model, optimizer, epoch,
  best-so-far indicator, RNG states). With DRG_CLS_RESUME=1, train() loads it, replays the
  per-epoch schedulers (sampler re-weighting, loss weights, warmup, LR) for the completed
  epochs exactly as an uninterrupted run would have stepped them, restores the optimizer, and
  continues at the next epoch.

dr_segmentation/train_fgadr.py (FGADR) and train.py (IDRiD) -- DRG-Net saves model_2.pth.tar
  only when eval AP improves (every 20 / 40 epochs). train.py's RESUME_MODEL path restores
  model + optimizer + epoch, but (a) then trains EPOCHES *more* epochs instead of up to EPOCHES,
  (b) restarts the StepLR count and the best-AP tracker; train_fgadr.py hard-codes
  `resume = False`, so it can't resume at all. Patch:
    * train_fgadr.py honours config.RESUME_MODEL (DRG_SEG_RESUME), like train.py;
    * model_last.pth.tar every DRG_SEG_SAVE_EVERY epochs (default 5) and at the end, with the
      current best AP -- the file to resume from (model_2.pth.tar stays the best-AP model);
    * resumed runs train up to EPOCHES total;
    * StepLR continues from the resumed epoch; best AP restored from DRG_SEG_BEST_AP;
    * eval interval from DRG_SEG_EVAL_EVERY (default = DRG-Net's 20 for FGADR, 40 for IDRiD).
  The training step, loss, data, and augmentation are untouched.

Usage:  python module1/drgnet_resume_patch.py /content/dr-joint-learning
"""
import os
import sys

MARK = "# [resume patch]"


def _patch(path, edits):
    src = open(path).read()
    if MARK in src:
        print(f"= {os.path.relpath(path)} already patched")
        return
    for old, new in edits:
        if src.count(old) != 1:
            raise SystemExit(f"Resume patch: expected exactly one match in {path} for:\n{old}\n"
                             f"found {src.count(old)} -- DRG-Net code differs from the pinned commit.")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"✓ patched {os.path.relpath(path)}")


CLS_EDITS = [
    ("""    model.train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(1, cfg.train.epochs + 1):
""", """    model.train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    start_epoch = 1  # [resume patch]
    _resume_path = os.path.join(cfg.base.save_path, 'resume_state.pt')
    if os.environ.get('DRG_CLS_RESUME') == '1' and os.path.isfile(_resume_path):
        import random as _random
        import numpy as _np
        _st = torch.load(_resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(_st['model'])
        # Replay the per-epoch schedules for the completed epochs, in train()'s own order.
        for _ in range(_st['epoch']):
            if weighted_sampler:
                weighted_sampler.step()
            if loss_weight_scheduler:
                loss_function.weight = loss_weight_scheduler.step().to(device)
            if warmup_scheduler and not warmup_scheduler.is_finish():
                warmup_scheduler.step()
            if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()) \\
                    and cfg.solver.lr_scheduler != 'reduce_on_plateau':
                lr_scheduler.step()
        optimizer.load_state_dict(_st['optimizer'])
        if _st.get('lr_scheduler') is not None:
            lr_scheduler.load_state_dict(_st['lr_scheduler'])
        max_indicator = _st['max_indicator']
        torch.set_rng_state(_st['rng_torch'])
        if torch.cuda.is_available() and _st.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(_st['rng_cuda'])
        _np.random.set_state(_st['rng_numpy'])
        _random.setstate(_st['rng_python'])
        start_epoch = _st['epoch'] + 1
        print('[resume] continuing at epoch {} / {} (best indicator so far {:.4f})'.format(
            start_epoch, cfg.train.epochs, max_indicator))
    for epoch in range(start_epoch, cfg.train.epochs + 1):
"""),
    ("""        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)""",
     """        # [resume patch] full state after every epoch (atomic write)
        import random as _random
        import numpy as _np
        _tmp = _resume_path + '.tmp'
        torch.save({
            'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if cfg.solver.lr_scheduler == 'reduce_on_plateau' else None,
            'max_indicator': max_indicator, 'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'rng_numpy': _np.random.get_state(), 'rng_python': _random.getstate(),
        }, _tmp)
        os.replace(_tmp, _resume_path)

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)"""),
]


def seg_edits(eval_every, save_anchor):
    last_save = """
        if (epoch + 1) % int(os.environ.get('DRG_SEG_SAVE_EVERY', 5)) == 0 \\
                or epoch + 1 == start_epoch + num_epochs:  # [resume patch]
            _last = os.path.join(dir_checkpoint, 'model_last.pth.tar')
            torch.save({'epoch': epoch, 'step': tot_step_count, 'state_dict': model.state_dict(),
                        'optimizer': g_optimizer.state_dict(), 'best_ap': best_ap}, _last + '.tmp')
            os.replace(_last + '.tmp', _last)
"""
    return [
        ("    best_ap = 0.\n",
         "    best_ap = float(os.environ.get('DRG_SEG_BEST_AP', 0.))  # [resume patch]\n"),
        (f"        if (epoch + 1) % {eval_every} == 0:\n",
         f"        if (epoch + 1) % int(os.environ.get('DRG_SEG_EVAL_EVERY', {eval_every})) == 0:\n"),
        (save_anchor, save_anchor + last_save),
        ("    g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=200, gamma=0.9)\n",
         "    g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=200, gamma=0.9)\n"
         "    g_scheduler.last_epoch = start_epoch  # [resume patch] continue the LR schedule\n"),
        ("num_epochs=config.EPOCHES,", "num_epochs=max(config.EPOCHES - start_epoch, 0),"),
    ]


def apply(drg_root):
    _patch(os.path.join(drg_root, "dr_classification", "train.py"), CLS_EDITS)
    _patch(os.path.join(drg_root, "dr_segmentation", "train_fgadr.py"), seg_edits(
        20, "                torch.save(state, \\\n"
            "                           os.path.join(dir_checkpoint, 'model_' + preprocess + '.pth.tar'))\n") + [
        # train_fgadr.py hard-codes `resume = False`, ignoring config.RESUME_MODEL entirely.
        ("    resume = False\n", "    resume = config.RESUME_MODEL  # [resume patch] DRG-Net hard-coded False\n"),
    ])
    _patch(os.path.join(drg_root, "dr_segmentation", "train.py"), seg_edits(
        40, "        logger.flush()\n        logger.flush()\n"))


if __name__ == "__main__":
    apply(sys.argv[1] if len(sys.argv) > 1 else "/content/dr-joint-learning")
