#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:03:07 2021

@author: sophierossi
"""
from PIL import Image
import os
import torch
import cv2
# os.chdir("home/rossis/yolov5/utils")
# from general import xywh2xyxy
# os.chdir("home/rossis/")

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def crop_and_send2_jetson(frame_to_crop, coords, destination = "PHOTOS/", name = "test.jpg"):
    s = frame_to_crop.shape
    det = coords[0]
    for i in range(len(det)):
        det[i] = int(det[i].item())
        if i == 0:
            det[i] = max(0, det[i] - 50)
        if i == 1:
            det[i] = max(0, det[i] - 50)
        if i == 2:
            det[i] = min(det[i] + 50, s[1])
        if i == 3:
            det[i] = min(det[i] + 50, s[0])
    x = det[0]
    w = det[2] - det[0]
    y = det[1]
    h = det[3] - det[1]
    frame_to_crop_ = frame_to_crop[y: y + h, x: x+w]
    cv2.imwrite(destination+name, frame_to_crop_)
    

def crop_and_send2(frame_to_crop, coords, destination = "PHOTOS/", name = "test.jpg"):
    s = frame_to_crop.shape
    det = coords[0]
    for i in range(len(det)):
        det[i] = int(det[i].item())
        if i == 0:
            det[i] = max(0, det[i] - 50)
        if i == 1:
            det[i] = max(0, det[i] - 50)
        if i == 2:
            det[i] = min(det[i] + 50, s[1])
        if i == 3:
            det[i] = min(det[i] + 50, s[0])
    x = det[0]
    w = det[2] - det[0]
    y = det[1]
    h = det[3] - det[1]
    frame_to_crop_ = frame_to_crop[y: y + h, x: x+w]
    cv2.imwrite(destination+name, frame_to_crop_)