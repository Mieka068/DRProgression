"""
Per-epoch checkpoint/resume shared by the Module 2 and Module 3 training scripts
(train_module2_nonlong_pretrain.py, train_module2_poc.py, module3/train_module3_poc.py).

After every epoch the full training state -- model and optimizer state_dicts, the epoch just
finished, script-specific extras (loss history, ...) and the Python / NumPy / torch RNG states --
is written atomically to one file (point it at Google Drive in Colab). Re-running the same
command with the same --resume-state path continues at the next epoch. Because the RNG states
are restored, data shuffling, augmentation and sampling continue as in an uninterrupted run.

On Google Drive the per-epoch write goes to a local twin on the VM's disk and is copied to Drive
only every RESUME_DRIVE_EVERY epochs (default 10) and after the last epoch (see drive_io.py: every
Drive overwrite leaves a copy that counts against the quota). A disconnect therefore loses at most
RESUME_DRIVE_EVERY - 1 epochs; restarting the runtime without disconnecting loses none.
"""
import os
import random

import numpy as np
import torch

from drive_io import local_twin, push_to_drive, restore_from_drive

DRIVE_EVERY = int(os.environ.get("RESUME_DRIVE_EVERY", 10))


def seed_everything(seed):
    """Fixes Python / NumPy / torch RNGs at the start of training so a run is reproducible
    (same seed + same data -> same weights init, shuffling and augmentation)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_resume_state(path, epoch, models, optimizers, extra=None, num_epochs=None):
    """epoch: 0-based index of the epoch that just FINISHED. models/optimizers: {name: obj}.
    num_epochs: the run's total, so the last epoch is always copied to Drive."""
    if not path:
        return
    drive_path, path = path, local_twin(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {
        "epoch": epoch,
        "models": {k: m.state_dict() for k, m in models.items()},
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "extra": extra or {},
        "rng": _rng_state(),
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    if path != drive_path and ((epoch + 1) % DRIVE_EVERY == 0 or epoch + 1 == num_epochs):
        push_to_drive(drive_path)


def load_resume_state(path, models, optimizers, device):
    """Restores models/optimizers/RNG in place if `path` exists. Returns (start_epoch, extra):
    start_epoch is the 0-based epoch to run next (0 and {} when there is nothing to resume)."""
    if not path:
        return 0, {}
    path = restore_from_drive(path)  # the local twin (newer) if this runtime has one, else Drive's
    if not os.path.isfile(path):
        return 0, {}
    state = torch.load(path, map_location=device, weights_only=False)
    for k, m in models.items():
        m.load_state_dict(state["models"][k])
    for k, o in optimizers.items():
        o.load_state_dict(state["optimizers"][k])
    _set_rng_state(state["rng"])
    start = state["epoch"] + 1
    print(f"[resume] restored {path}: {start} epoch(s) already done -- training continues from "
          f"epoch {start + 1} if more epochs are requested, otherwise skips to evaluation")
    return start, state["extra"]
