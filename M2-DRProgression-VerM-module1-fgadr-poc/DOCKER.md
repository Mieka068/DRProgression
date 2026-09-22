# Running this project locally with Docker (instead of Colab)

This lets you run the same scripts the notebooks call -- unmodified -- on a machine with its
own NVIDIA GPU, with no Google Drive mount and no per-session GitHub token prompt. All the
training/inference scripts already take their dataset paths as explicit CLI flags (they were
written that way for Colab, where the data doesn't live next to the code either), so nothing
about them had to change for this -- only the *environment* they run in is new.

**Not built or run yet.** This was written and reviewed carefully against the actual notebook
commands and script CLI flags, but there is no Docker daemon or GPU in the sandbox this was
authored in, so the image has never actually been built or run. Treat the first build on your
own machine as the real test, and send back the exact error if something breaks -- dependency
versions in particular are the most likely thing to need a tweak.

## Prerequisites

- An NVIDIA GPU with a recent driver installed.
- Docker Engine or Docker Desktop.
- The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) installed
  and configured -- this is what lets a container see the GPU at all. Without it, `--gpus all`
  fails outright, which is a useful way to confirm it's installed correctly.

Verify GPU access works before building anything project-specific:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
If that doesn't print your GPU, fix that first -- everything below assumes it works.

The Dockerfile's base image (`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`) targets CUDA
12.1. If `nvidia-smi` reports an older CUDA version than that, change the tag in `Dockerfile`
to a matching one from [PyTorch's image list](https://hub.docker.com/r/pytorch/pytorch/tags).

## Build

From inside this directory (`M2-DRProgression-VerM-module1-fgadr-poc/`):
```bash
docker build -t drprogression .
```
This bakes in this project's own code and Python dependencies, plus a pinned clone of
DRG-Net's reference implementation (`dr-joint-learning`, the same commit notebooks 02/03 use)
for Module 1's classifier/segmentation training. It does **not** include any dataset -- those
are mounted in at run time, same idea as Colab reading them off Drive instead of baking them
into the notebook.

Rebuild (`docker build -t drprogression .` again) whenever you pull code changes.

## Dataset layout

Extract your dataset zips into a `datasets/` folder next to this Dockerfile (any layout is
fine as long as the paths you pass to each script below match). For example:
```
datasets/
  FIRE_dataset/
  LongDRScreening_20150209/
  retinal-dr-longitudinal/        # Tianjin (usama10/retinal-dr-longitudinal)
  FGADR-Seg-set_Release/
  IDRiD/
    B. Disease Grading/
```
Also create an empty `runtime_outputs/` folder next to it -- checkpoints, caches, and figures
will land there instead of disappearing when the container exits.

## Run

Interactively (recommended -- lets you run one command at a time and see what fails first):
```bash
docker run --gpus all -it \
    -v "$(pwd)/datasets:/data" \
    -v "$(pwd)/runtime_outputs:/outputs" \
    drprogression bash
```
or with Compose:
```bash
docker compose run --rm drprogression bash
```

Everything below runs *inside* that container, from `/workspace` (this project's code).

## Module 1 -- data prep, then classification + segmentation training (DRG-Net)

First build the classification pkl indexes DRG-Net's `main.py` reads (same as notebook 01):
```bash
cd /workspace
mkdir -p /outputs/module1_pkl
python module1/dataprep/make_fgadr_classification_pkl.py \
    --fgadr-root /data/FGADR-Seg-set_Release/Seg-set \
    --out /outputs/module1_pkl/fgadr_classification_pkl.pkl
python module1/dataprep/make_idrid_classification_pkl.py \
    --idrid-root "/data/IDRiD/B. Disease Grading" \
    --out /outputs/module1_pkl/idrid_classification_pkl.pkl
```

Classification training is DRG-Net's own code, cloned to `/opt/dr-joint-learning` at build
time -- unmodified, same as notebook 02. `fgadr_local.yaml` / `idrid_local.yaml` are this
project's own `fgadr_poc.yaml` / `idrid_poc.yaml` with their `base:` paths pointed at `/data`
and `/outputs` instead of Colab's Drive paths (same epoch counts: 20 and 60, kept exactly as
published):
```bash
cp module1/configs/fgadr_local.yaml module1/configs/idrid_local.yaml /opt/dr-joint-learning/dr_classification/configs/
cd /opt/dr-joint-learning/dr_classification
python main.py -c configs/fgadr_local.yaml -overwrite   # 20 epochs, as published
python main.py -c configs/idrid_local.yaml -overwrite   # 60 epochs, as published
```

Segmentation (`train_fgadr.py` hardcodes `import config_fgadr as config`, so the config has to
replace that file, not get passed as a flag -- same as notebook 03).
`config_fgadr_seg_local.py` is `config_fgadr_seg_poc.py` with `IMAGE_DIR` pointed at `/data`
instead of `/content/data` (still the same reduced 25 epochs, EX + MA lesions):
```bash
cp module1/configs/config_fgadr_seg_local.py /opt/dr-joint-learning/dr_segmentation/config_fgadr.py
cd /opt/dr-joint-learning/dr_segmentation
python train_fgadr.py --seed 765                      # EX (hard exudates)
python train_fgadr.py --seed 765 --preprocess 2 --lesion MA   # MA (microaneurysms)
```
Checkpoints land under `/opt/dr-joint-learning/dr_segmentation/results/models_FGADR_NO_TATL_*`
-- copy them to `/outputs` afterward so they survive the container exiting (notebook 03 does
the same thing, just to Drive instead):
```bash
cp -r /opt/dr-joint-learning/dr_segmentation/results/models_FGADR_NO_TATL_* /outputs/module1_runs/segmentation/
```

## Module 1 -- Tianjin prep (laterality resolution + inference cache)

```bash
cd /workspace
python module1/resolve_eye_laterality.py --dataset-dir /data/retinal-dr-longitudinal

python module1/apply_to_progression_data.py \
    --images-dir /data/retinal-dr-longitudinal/"baseline fundus images" \
    --classifier-checkpoint /outputs/module1_runs/classification/fgadr/saves/final_weights.pt \
    --seg-checkpoint EX=/outputs/module1_runs/segmentation/models_FGADR_NO_TATL_ex/model_2.pth.tar \
    --seg-checkpoint MA=/outputs/module1_runs/segmentation/models_FGADR_NO_TATL_ma/model_2.pth.tar \
    --out /outputs/module1_cache/module1_outputs_tianjin.pt
```
Repeat with `--images-dir /data/FIRE_dataset/FIRE/Images` / `--images-dir
/data/LongDRScreening_20150209/FundusImagesNormalized` and matching `--out` paths for FIRE and
LongDR, exactly as notebook 04/05 do.

No trained checkpoints yet? Pass `--dry-run` instead of `--classifier-checkpoint` /
`--seg-checkpoint` to smoke-test the plumbing with random weights (see the module docstring).

## Module 2 -- registration network + GAN training

```bash
cd /workspace/module1
python train_registration.py \
    --fire-dir /data/FIRE_dataset \
    --longdr-dir /data/LongDRScreening_20150209 \
    --tianjin-dir /data/retinal-dr-longitudinal \
    --num-epochs 10 \
    --checkpoint-out /outputs/registration_net.ckpt \
    --cache-out /outputs/registered_followups.pt

mkdir -p /outputs/module2_run && cd /outputs/module2_run
python /workspace/train_module2_poc.py \
    --fire-dir /data/FIRE_dataset \
    --longdr-dir /data/LongDRScreening_20150209 \
    --tianjin-dir /data/retinal-dr-longitudinal \
    --fire-module1-cache /outputs/module1_cache/module1_outputs_fire.pt \
    --longdr-module1-cache /outputs/module1_cache/module1_outputs_longdr.pt \
    --tianjin-module1-cache /outputs/module1_cache/module1_outputs_tianjin.pt \
    --registration-cache /outputs/registered_followups.pt \
    --num-epochs 50 \
    --save-dir /outputs/models_poc_tianjin/ \
    --num-workers 4
```
`--save-dir` is an absolute path, so it lands under `/outputs` regardless of where this runs
from -- but the per-epoch sample images (`training_samples_poc/epoch_N.jpg`) have no CLI flag
and always write relative to the current directory. That's why the `cd /outputs/module2_run`
above matters: running this from `/workspace` instead would leave those samples inside the
container's own filesystem, gone as soon as it exits.

`--num-workers` is a real CPU/GPU tradeoff, not a Colab-specific knob -- set it to roughly your
machine's CPU core count (leaving a couple free), same reasoning as the notebook.

Evaluate the trained cascade:
```bash
cd /workspace
python evaluate_trajectory.py \
    --tianjin-dir /data/retinal-dr-longitudinal \
    --tianjin-module1-cache /outputs/module1_cache/module1_outputs_tianjin.pt \
    --generator-checkpoint /outputs/models_poc_tianjin/final-G.ckpt \
    --classifier-checkpoint /outputs/module1_runs/classification/fgadr/saves/final_weights.pt \
    --seg-checkpoint EX=/outputs/module1_runs/segmentation/models_FGADR_NO_TATL_ex/model_2.pth.tar \
    --seg-checkpoint MA=/outputs/module1_runs/segmentation/models_FGADR_NO_TATL_ma/model_2.pth.tar \
    --out /outputs/trajectory_eval_results.json
```

## Module 3 -- survival model training

`train_module3_poc.py` has no `--save-dir` flag at all -- it always writes
`final-model.ckpt`/`poc_results.json` to `./module3_runs_poc/`, relative to wherever it's run
from. `cd` into `/outputs` first (matching the `/outputs/module3/module3_runs_poc/...` paths
the visualization command below expects) rather than running it from `/workspace`:
```bash
mkdir -p /outputs/module3 && cd /outputs/module3
python /workspace/module3/train_module3_poc.py \
    --tianjin-dir /data/retinal-dr-longitudinal \
    --tianjin-module1-cache /outputs/module1_cache/module1_outputs_tianjin.pt \
    --num-epochs 10
```

## Trajectory visualization

```bash
cd /workspace
python visualize_trajectory.py \
    --tianjin-dir /data/retinal-dr-longitudinal \
    --tianjin-module1-cache /outputs/module1_cache/module1_outputs_tianjin.pt \
    --generator-checkpoint /outputs/models_poc_tianjin/final-G.ckpt \
    --classifier-checkpoint /outputs/module1_runs/classification/fgadr/saves/final_weights.pt \
    --seg-checkpoint EX=/outputs/module1_runs/segmentation/models_FGADR_NO_TATL_ex/model_2.pth.tar \
    --seg-checkpoint MA=/outputs/module1_runs/segmentation/models_FGADR_NO_TATL_ma/model_2.pth.tar \
    --module3-checkpoint /outputs/module3/module3_runs_poc/final-model.ckpt \
    --module3-thresholds-json /outputs/module3/module3_runs_poc/poc_results.json \
    --num-patients 4 \
    --out-dir /outputs/trajectory_figures/
```

## Why this can just reuse the existing scripts unmodified

Every script above already takes its dataset/checkpoint paths as explicit `argparse` flags --
that was a deliberate choice from when Tianjin support was added (see `combined_dataset.py` /
`train_module2_poc.py`'s own docstrings on this), specifically because Colab's data and code
live in different places. Docker has exactly the same shape of problem (data mounted from the
host, code baked into the image), so no script needed to change -- only the environment
(this Dockerfile) and the paths you pass in.
