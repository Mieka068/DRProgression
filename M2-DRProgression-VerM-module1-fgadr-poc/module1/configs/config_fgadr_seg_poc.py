#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copy of DRG-Net's dr_segmentation/config_fgadr.py (pinned commit
# 0da1bbe885f438390a0e94ec486283d9b21d5547). Only IMAGE_DIR, EPOCHES and RESUME_MODEL changed.
#
# Colab usage: this file replaces dr_segmentation/config_fgadr.py in the cloned repo
# (train_fgadr.py does `import config_fgadr as config`).
#
# EPOCHES / RESUME_MODEL come from environment variables so the same file serves both runs:
#   POC (notebook 03, default):     DRG_SEG_EPOCHS unset -> 25
#   DRG-Net replication (nb 11):    DRG_SEG_EPOCHS=1500, the paper's published setting
#   Resume after a Colab cut-off:   DRG_SEG_RESUME=/path/to/model_last.pth.tar. Needs
#     module1/drgnet_resume_patch.py applied first: DRG-Net's train_fgadr.py hard-codes
#     `resume = False` and only saves the best-AP model; the patch adds model_last.pth.tar
#     every DRG_SEG_SAVE_EVERY epochs and makes resume continue the epoch count / LR schedule.
import os

LESION_IDS = {'EX': 0, 'HE': 1, 'MA': 2, 'SE': 3, 'SG': 4}
DATASET_NAME = 'FGADR'
TATL = 'NO_TATL'

# Modify the general parameters.
IMAGE_DIR = '/content/data/FGADR-Seg-set_Release/Seg-set'  # local FGADR copy, not the authors' /mnt/sda path
NET_NAME = 'unet'
IMAGE_SIZE = 512

# Modify the parameters for training.
# 25 = same-day POC time-boxing (say so plainly; don't present it as matching published
# results). 1500 = DRG-Net's published setting.
EPOCHES = int(os.environ.get('DRG_SEG_EPOCHS', 25))
TRAIN_BATCH_SIZE = 4
G_LEARNING_RATE = 0.001
LESION_DICE_WEIGHT = 0.  # never read by DRG-Net's train scripts -- set DRG_SEG_DICE_WEIGHT instead (drgnet_resume_patch.py)
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = os.environ.get('DRG_SEG_RESUME') or None
LOAD_PRETRAINED = None
