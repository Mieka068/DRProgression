"""
Adds checkpoint/resume to a cloned DRG-Net repo (pinned commit 0da1bbe) so long runs survive a
Colab disconnect. Applied by notebook 11; idempotent (re-running is a no-op) and it fails loudly
if DRG-Net's code doesn't look as expected, rather than silently patching the wrong place.

What DRG-Net does on its own, and what this adds:

dr_classification/train.py -- DRG-Net saves only model weights (best_validation_weights.pt,
  epoch_N.pt every 5 epochs), no optimizer / LR schedule / epoch, so a run cannot be resumed.
  It also only writes best_validation_weights.pt once validation kappa exceeds 0, then loads it
  unconditionally -- a run whose kappa never exceeds 0 crashes at the end; the patch starts the
  best-so-far at -1 so the first epoch is always saved.
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

dr_segmentation/train.py + utils.py -- also reused for DDR (same layout as IDRiD, see
  module1/dataprep/prepare_ddr_segmentation.py). Separate patch marker, so it also applies to a
  clone that already has the resume patch. Env-driven, defaults = DRG-Net's behaviour:
    * DRG_SEG_RUN_PREFIX (default 'new_'): checkpoints in results/<prefix>models_<lesion>/ and the
      AP log <prefix>ap_during_learning_*.txt -- 'ddr_' keeps DDR runs apart from IDRiD's;
    * DRG_SEG_TRAIN_RATIO (default 0.8): train/eval cut of the sorted TrainingSet -- DDR's prep
      sorts its official train images before its val images and writes the exact ratio.
  dataset.py's IDRIDDataset holds every image and mask in RAM, masks as 3-channel RGB of which
  only channel 0 is ever read. The patch keeps that channel as a 1-channel image instead (same
  values; the group transforms treat it identically), cutting mask memory to a third -- needed
  for DDR (532 images) and the FGADR + DDR combined run (853) to fit in Colab's RAM.

Small-lesion loss (separate marker, applies to already-patched clones; CODE_FIXES item 1.4):
  DRG-Net's config has LESION_DICE_WEIGHT, but train.py / train_fgadr.py never read it -- the loss is
  plain class-weighted cross-entropy. With DRG_SEG_DICE_WEIGHT=w > 0 the patch adds w x soft Dice
  loss (1 - Dice of the lesion-class softmax vs. the mask, per batch) to it, which weighs a lesion
  by its area rather than its pixel count -- aimed at microaneurysms, whose few pixels barely move
  the CE loss. Unset / 0 = DRG-Net's loss.

Resume loading (separate marker): PyTorch >= 2.6 defaults torch.load to weights_only=True, which
  rejects model_last.pth.tar (numpy scalars inside), so a resumed segmentation run crashed before
  training -- the resume load now passes weights_only=False (and best_ap is saved as a plain float).

Drive mirroring (separate marker, applies to already-patched clones): every overwrite of a file on
a Colab-mounted Drive leaves the old copy behind, counted against the Drive quota -- rewriting
~250 MB checkpoints every few epochs straight onto Drive fills it within hours, after which model
files silently fail to upload. Notebook 11 therefore points DRG-Net's save folders at the VM's
local disk and sets
    * DRG_SEG_MIRROR_DIR / DRG_SEG_MIRROR_EVERY (segmentation): every N epochs (a multiple of
      DRG_SEG_SAVE_EVERY) and at the last epoch, copy the run folder's changed files to Drive;
    * DRG_CLS_MIRROR_DIR / DRG_CLS_MIRROR_EVERY (classification): same for save_path after the
      epoch's resume_state.pt (the epoch_N.pt snapshots stay local).

Usage:  python module1/drgnet_resume_patch.py /content/dr-joint-learning
"""
import os
import sys

MARK = "# [resume patch]"


def _patch(path, edits, mark=MARK):
    src = open(path).read()
    if mark in src:
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
    # [resume patch] DRG-Net starts at 0 and saves best_validation_weights.pt only when the val
    # indicator rises ABOVE it -- if kappa never gets above 0 (a weak/short run), the file is never
    # written and main.py crashes loading it. -1 (kappa's minimum) means epoch 1 is always saved.
    max_indicator = -1
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    start_epoch = 1
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
        if start_epoch > cfg.train.epochs:
            print('[resume] already trained {} / {} epochs -- skipping to evaluation'.format(
                _st['epoch'], cfg.train.epochs))
        else:
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
                        'optimizer': g_optimizer.state_dict(), 'best_ap': float(best_ap)}, _last + '.tmp')
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


DATASET_MARK = "# [dataset patch]"
SEG_DATASET_EDITS = [
    ("    dir_checkpoint = 'results/new_models_' + lesion.lower()\n",
     "    dir_checkpoint = 'results/' + os.environ.get('DRG_SEG_RUN_PREFIX', 'new_') + 'models_' + lesion.lower()"
     "  # [dataset patch]\n"),
    ('            with open("new_ap_during_learning_" + lesion + preprocess + ".txt", \'a\') as f:\n',
     "            with open(os.environ.get('DRG_SEG_RUN_PREFIX', 'new_') + 'ap_during_learning_' + lesion + preprocess"
     " + '.txt', 'a') as f:  # [dataset patch]\n"),
]
DATASET_PY_EDITS = [
    ("                    self.masks.append(self.pil_loader(mask_path))\n",
     "                    self.masks.append(Image.fromarray(np.array(self.pil_loader(mask_path))[:, :, 0]))"
     "  # [dataset patch] channel 0 only\n"),
    ("            mask = np.array(np.array(info[1]))[:, :, 0] / 255.0\n",
     "            mask = np.array(info[1]) / 255.0  # [dataset patch] masks are stored 1-channel\n"),
]
UTILS_DATASET_EDITS = [
    ("train_ratio = 0.8\neval_ratio = 0.2\n",
     "train_ratio = float(os.environ.get('DRG_SEG_TRAIN_RATIO', 0.8))  # [dataset patch]\n"
     "eval_ratio = 1 - train_ratio\n"),
]


LOSS_MARK = "# [loss patch]"
SEG_LOSS_ANCHOR = "            ce_weight = 1.\n            g_loss = loss_ce * ce_weight\n"
SEG_LOSS_EDITS = [(SEG_LOSS_ANCHOR, SEG_LOSS_ANCHOR + """\
            _dice_w = float(os.environ.get('DRG_SEG_DICE_WEIGHT', 0))  # [loss patch] soft Dice term
            if _dice_w > 0:
                _p = torch.softmax(masks_pred, dim=1)[:, 1]
                _t = true_masks_indices.float()
                g_loss = g_loss + _dice_w * (1 - (2 * (_p * _t).sum() + 1.) / (_p.sum() + _t.sum() + 1.))
""")]


# PyTorch >= 2.6 loads with weights_only=True by default, which rejects DRG-Net's resume checkpoints
# (they hold numpy scalars, e.g. best_ap): resuming crashed on Colab before training started. Our own
# checkpoint files, so load them fully. Separate marker, so already-patched clones get it too.
LOAD_MARK = "# [load patch]"
SEG_LOAD_EDITS = [("            checkpoint = torch.load(resume)\n",
                   "            checkpoint = torch.load(resume, weights_only=False)  # [load patch]\n")]


MIRROR_MARK = "# [mirror patch]"


def _mirror_code(indent, src_dir, cond):
    pad = " " * indent
    return (
        f"{pad}_mirror = os.environ.get('{cond[0]}')  {MIRROR_MARK} changed files -> Drive every N epochs + at the end\n"
        f"{pad}if _mirror and ({cond[1]}):\n"
        f"{pad}    import shutil as _sh\n"
        f"{pad}    _dst = {cond[2]}\n"
        f"{pad}    os.makedirs(_dst, exist_ok=True)\n"
        f"{pad}    for _f in os.listdir({src_dir}):\n"
        f"{pad}        _s, _d = os.path.join({src_dir}, _f), os.path.join(_dst, _f)\n"
        f"{pad}        if (os.path.isfile(_s) and not _f.endswith('.tmp') and not _f.startswith('epoch_')\n"
        f"{pad}                and (not os.path.isfile(_d) or os.path.getmtime(_s) > os.path.getmtime(_d))):\n"
        f"{pad}            _sh.copyfile(_s, _d)\n"
    )


SEG_MIRROR_ANCHOR = "            os.replace(_last + '.tmp', _last)\n"
SEG_MIRROR_EDITS = [(SEG_MIRROR_ANCHOR, SEG_MIRROR_ANCHOR + _mirror_code(
    12, "dir_checkpoint",
    ("DRG_SEG_MIRROR_DIR",
     "(epoch + 1) % int(os.environ.get('DRG_SEG_MIRROR_EVERY', 25)) == 0 or epoch + 1 == start_epoch + num_epochs",
     "os.path.join(_mirror, os.path.basename(dir_checkpoint))")))]
CLS_MIRROR_ANCHOR = "        os.replace(_tmp, _resume_path)\n"
CLS_MIRROR_EDITS = [(CLS_MIRROR_ANCHOR, CLS_MIRROR_ANCHOR + _mirror_code(
    8, "cfg.base.save_path",
    ("DRG_CLS_MIRROR_DIR",
     "epoch % int(os.environ.get('DRG_CLS_MIRROR_EVERY', 5)) == 0 or epoch == cfg.train.epochs",
     "_mirror")))]


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
    _patch(os.path.join(drg_root, "dr_segmentation", "train.py"), SEG_DATASET_EDITS, DATASET_MARK)
    _patch(os.path.join(drg_root, "dr_segmentation", "utils.py"), UTILS_DATASET_EDITS, DATASET_MARK)
    _patch(os.path.join(drg_root, "dr_segmentation", "dataset.py"), DATASET_PY_EDITS, DATASET_MARK)
    for script in ("train.py", "train_fgadr.py"):
        _patch(os.path.join(drg_root, "dr_segmentation", script), SEG_MIRROR_EDITS, MIRROR_MARK)
        _patch(os.path.join(drg_root, "dr_segmentation", script), SEG_LOSS_EDITS, LOSS_MARK)
        _patch(os.path.join(drg_root, "dr_segmentation", script), SEG_LOAD_EDITS, LOAD_MARK)
    _patch(os.path.join(drg_root, "dr_classification", "train.py"), CLS_MIRROR_EDITS, MIRROR_MARK)


if __name__ == "__main__":
    apply(sys.argv[1] if len(sys.argv) > 1 else "/content/dr-joint-learning")
