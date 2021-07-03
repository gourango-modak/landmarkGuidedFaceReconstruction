import matplotlib
matplotlib.use('Agg')
import math
import torch
import copy
import time
from torch.autograd import Variable
import shutil
from skimage import io
import numpy as np
from .utils.utils import fan_NME, show_landmarks, get_preds_fromhm
from PIL import Image, ImageDraw
import os
import sys
import cv2
import matplotlib.pyplot as plt
from .dataloader import get_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_model(model, image_path, use_gpu=True, num_landmarks=68):

    input = get_image(image_path)['image'].type(torch.FloatTensor)
    
    if use_gpu:
        input = input.to(device)
    else:
        input = Variable(input)
        
    outputs, boundary_channels = model(input)
    
    img = input
  
    img = img.cpu().numpy()
    img = np.squeeze(img, axis=0)
    img = img.transpose((1, 2, 0))*255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    
    pred_heatmap = outputs[-1][:, :-1, :, :][0].detach().cpu()
    pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
    pred_landmarks = pred_landmarks.squeeze().numpy()

    pred_landmarks = pred_landmarks.astype(np.int32)
    pred_landmarks *= 4
    
    return input, pred_landmarks
