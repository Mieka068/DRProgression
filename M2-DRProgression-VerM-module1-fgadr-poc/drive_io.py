"""
Keep frequently rewritten files (resume states, checkpoints, growing caches) off Google Drive.

Every overwrite of a file on a Colab-mounted Drive leaves the old copy behind (Trash / an older
version), and those copies count against the Drive quota -- a ~250 MB checkpoint rewritten every
few epochs fills a 15 GB Drive within hours, after which large uploads fail silently (small logs
still save, model files don't). So: write every save to a local twin of the Drive path on the
Colab VM's own disk, and copy it to Drive only every N saves and at the end.

Paths outside /content/drive (local runs, tests) are used directly -- no twin, no copying.
"""
import os
import shutil

DRIVE_ROOT = "/content/drive"
LOCAL_ROOT = os.environ.get("DRIVE_LOCAL_TWIN_ROOT", "/content/local_drive_twin")


def on_drive(path):
    return bool(path) and os.path.abspath(path).startswith(DRIVE_ROOT + os.sep)


def local_twin(path):
    """Where to write `path` locally: the same relative path under LOCAL_ROOT, or `path` itself
    when it is not on Drive."""
    if not on_drive(path):
        return path
    rel = os.path.relpath(os.path.abspath(path), DRIVE_ROOT)
    return os.path.join(LOCAL_ROOT, rel)


def restore_from_drive(path):
    """Before resuming: copy the Drive file to its local twin if the twin is missing (new runtime).
    Returns the path to read/write from now on."""
    twin = local_twin(path)
    if twin != path and not os.path.isfile(twin) and os.path.isfile(path):
        os.makedirs(os.path.dirname(twin), exist_ok=True)
        shutil.copyfile(path, twin)
    return twin


def push_to_drive(path):
    """Copy the local twin over the Drive file (one overwrite) and check it arrived. No-op off Drive."""
    twin = local_twin(path)
    if twin == path or not os.path.isfile(twin):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copyfile(twin, path)
    if os.path.getsize(path) != os.path.getsize(twin):
        raise OSError(f"Copy to Drive incomplete: {path} -- is your Google Drive full? "
                      "Check drive.google.com/settings/storage and empty the Trash.")
