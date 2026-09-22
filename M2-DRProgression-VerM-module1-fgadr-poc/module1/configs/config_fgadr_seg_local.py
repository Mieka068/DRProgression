#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Same as config_fgadr_seg_poc.py, with IMAGE_DIR pointed at the Docker volume mount
# (/data, see DOCKER.md) instead of Colab's /content/data.
LESION_IDS = {'EX': 0, 'HE': 1, 'MA': 2, 'SE': 3, 'SG': 4}
DATASET_NAME = 'FGADR'
TATL = 'NO_TATL'

# Modify the general parameters.
IMAGE_DIR = '/data/FGADR-Seg-set_Release/Seg-set'
NET_NAME = 'unet'
IMAGE_SIZE = 512

# Modify the parameters for training.
# EPOCHES reduced from the paper's published 1500 -> 25 for a same-day, time-boxed POC run
# (see config_fgadr_seg_poc.py) -- unchanged here, still a POC-scale run.
EPOCHES = 25
TRAIN_BATCH_SIZE = 4
G_LEARNING_RATE = 0.001
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = None
LOAD_PRETRAINED = None
