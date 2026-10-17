#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copy of DRG-Net's dr_segmentation/config.py (pinned commit
# 0da1bbe885f438390a0e94ec486283d9b21d5547) -- the config of their IDRiD segmentation run
# (dr_segmentation/train.py and evaluate_model.py both `import config`). Only IMAGE_DIR,
# EPOCHES and RESUME_MODEL changed (IMAGE_DIR also overridable, for DDR).
#
# Colab usage: copy over dr_segmentation/config.py, after
# module1/dataprep/prepare_idrid_segmentation.py has laid IDRiD's segmentation set out at
# IMAGE_DIR in the folder structure DRG-Net's utils.get_images() reads.
#
# EPOCHES: DRG-Net's published value is 2500 (DRG_SEG_EPOCHS unset). Set DRG_SEG_EPOCHS lower
# for a time-boxed run; DRG_SEG_RESUME as in config_fgadr_seg_poc.py (needs
# module1/drgnet_resume_patch.py).
import os

LESION_IDS = {'EX': 0, 'HE': 1, 'MA': 2, 'SE': 3, 'SG': 4}

# Modify the general parameters.
# DRG_SEG_IMAGE_DIR: the same config serves DDR (notebook 11 points it at the DDR layout from
# module1/dataprep/prepare_ddr_segmentation.py); unset = IDRiD.
IMAGE_DIR = os.environ.get('DRG_SEG_IMAGE_DIR', '/content/data/IDRiD_seg_drgnet/')
NET_NAME = 'unet'
IMAGE_SIZE = 512

# Modify the parameters for training.
EPOCHES = int(os.environ.get('DRG_SEG_EPOCHS', 2500))
TRAIN_BATCH_SIZE = 4
G_LEARNING_RATE = 0.001
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = os.environ.get('DRG_SEG_RESUME') or None
LOAD_PRETRAINED = None
