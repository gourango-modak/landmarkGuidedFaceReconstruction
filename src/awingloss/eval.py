from __future__ import print_function, division
import torch
import argparse
import numpy as np
import torch.nn as nn
import time
import os
from .core.evaler import eval_model
from .core import models


def test_model(image_path, pretrained_weights, gray_scale, hg_blocks, end_relu, num_landmarks):

    PRETRAINED_WEIGHTS = pretrained_weights
    GRAY_SCALE = False if gray_scale == 'False' else True
    HG_BLOCKS = hg_blocks
    END_RELU = False if end_relu == 'False' else True
    NUM_LANDMARKS = num_landmarks

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_gpu = torch.cuda.is_available()
    model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)

    if PRETRAINED_WEIGHTS != "None":
        checkpoint = torch.load(PRETRAINED_WEIGHTS)
        if 'state_dict' not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model_ft.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                if k in model_weights}
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights)

    model_ft = model_ft.to(device)

    img, landmark = eval_model(model_ft, image_path, use_gpu, NUM_LANDMARKS)

    return img, landmark
