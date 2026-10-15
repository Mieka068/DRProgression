"""
Per-epoch checkpoint/resume shared by the Module 2 and Module 3 training scripts
(train_module2_nonlong_pretrain.py, train_module2_poc.py, module3/train_module3_poc.py).

After every epoch the full training state -- model and optimizer state_dicts, the epoch just
finished, script-specific extras (loss history, ...) and the Python / NumPy / torch RNG states --
is written atomically to one file (point it at Google Drive in Colab). Re-running the same
command with the same --resume-state path continues at the next epoch. Because the RNG states
are restored, data shuffling, augmentation and sampling continue as in an uninterrupted run.
"""
import os
import random

import numpy as np
import torch


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


def save_resume_state(path, epoch, models, optimizers, extra=None):
    """epoch: 0-based index of the epoch that just FINISHED. models/optimizers: {name: obj}."""
    if not path:
        return
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


def load_resume_state(path, models, optimizers, device):
    """Restores models/optimizers/RNG in place if `path` exists. Returns (start_epoch, extra):
    start_epoch is the 0-based epoch to run next (0 and {} when there is nothing to resume)."""
    if not path or not os.path.isfile(path):
        return 0, {}
    state = torch.load(path, map_location=device, weights_only=False)
    for k, m in models.items():
        m.load_state_dict(state["models"][k])
    for k, o in optimizers.items():
        o.load_state_dict(state["optimizers"][k])
    _set_rng_state(state["rng"])
    start = state["epoch"] + 1
    print(f"[resume] restored {path}: {start} epoch(s) already done, continuing at epoch {start + 1}")
    return start, state["extra"]
