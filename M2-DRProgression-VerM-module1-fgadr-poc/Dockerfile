# GPU image for running this project's training/inference scripts locally (outside Colab) on
# a machine with an NVIDIA GPU + the NVIDIA Container Toolkit installed. See DOCKER.md for
# build/run instructions and the local equivalent of each notebook's commands.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# libgl1/libglib2.0-0: opencv-python imports these at runtime even in "headless" mode.
# git: needed below to clone DRG-Net's reference implementation.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# This project's own dependencies.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# DRG-Net's pinned reference implementation (Module 1's dr_classification/main.py and
# dr_segmentation/train_fgadr.py) -- external code, not vendored into this repo. Pinned to the
# same commit notebooks 02/03 use, so results match.
#
# Deliberately NOT installing dr-joint-learning's own requirements.txt: it pins 2022-era
# torch==1.7.1 / numpy==1.21.6 / Pillow==6.2.2, which have no prebuilt wheels for this image's
# Python and would try to build from source, breaking the base image's CUDA-matched torch in
# the process (see notebooks 02/03's own comments on this exact problem in Colab). Install
# only the extra packages its imports need beyond what's already here.
RUN git clone https://github.com/DFKI-Interactive-Machine-Learning/dr-joint-learning.git /opt/dr-joint-learning \
    && cd /opt/dr-joint-learning \
    && git checkout 0da1bbe885f438390a0e94ec486283d9b21d5547
RUN pip install --no-cache-dir munch self-attention-cv pretrainedmodels cnn_finetune

# This project's own code.
COPY . /workspace

CMD ["/bin/bash"]
