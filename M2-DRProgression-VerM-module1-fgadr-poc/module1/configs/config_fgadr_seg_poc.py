#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copy of DRG-Net's dr_segmentation/config_fgadr.py (pinned commit
# 0da1bbe885f438390a0e94ec486283d9b21d5547). Only IMAGE_DIR and EPOCHES changed.
#
# Colab usage: this file replaces dr_segmentation/config_fgadr.py in the cloned repo (or is
# imported as `config` in place of it -- train_fgadr.py does `import config_fgadr as config`,
# so either overwrite that file or symlink this one over it before running).
LESION_IDS = {'EX': 0, 'HE': 1, 'MA': 2, 'SE': 3, 'SG': 4}
DATASET_NAME = 'FGADR'
TATL = 'NO_TATL'

# Modify the general parameters.
IMAGE_DIR = '/content/data/FGADR-Seg-set_Release/Seg-set'  # local FGADR copy, not the authors' /mnt/sda path
NET_NAME = 'unet'
IMAGE_SIZE = 512

# Modify the parameters for training.
# EPOCHES reduced from the paper's published 1500 -> 25 for a same-day, time-boxed POC run
# on a single Colab GPU session. This is the direct, honest "why not 1500 epochs" answer for
# the presentation (b.3) -- say so plainly, don't present this as matching published results.
EPOCHES = 25
TRAIN_BATCH_SIZE = 4
G_LEARNING_RATE = 0.001
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = None
LOAD_PRETRAINED = None
