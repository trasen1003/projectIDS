#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:50:59 2021

@author: sophierossi
"""
import sys
sys.path.append("/home/rossis/yolov5")
sys.path.append("/home/rossis/sort")
# import os
# os.chdir("/home/rossis/yolov5")

from yolov5 import *
from yolov5.detect_bis import *

from sort import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

os.chdir("/home/rossis/sort")
from models import *
from utils import *
os.chdir("..")


videopath = '/home/rossis/yolov5/data/videos/black_insect.mp4'

 
import cv2
print(cv2.__version__)
from IPython.display import clear_output
from math import floor
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# initialize Sort object and video capture
os.chdir("/home/rossis/sort")
from sort import *

vid = cv2.VideoCapture(videopath)
fps = vid.get(cv2.CAP_PROP_FPS)
nb_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
print(f"Total number of frames {nb_frames}")
os.chdir("..")

# print(vid)
mot_tracker = Sort() 
"""
aha = Opt(source = "./data/images")
print("aha", aha)
load_model(aha)
infer_pics()
"""    
    
    
    
while vid.isOpened():
    aha = Opt()
    load_model(aha)
    img_size = get_img_size()
    classes = get_names()
    # print(classes)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/rossis/sort/results/test.mp4',fourcc, floor(fps), (frame_width,frame_height))
    for ii in range(nb_frames):
        print(ii)
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        img = transform_frame(frame)
        pred = return_pred(img)[0]
        
        detections = shape_pred(pred)
        # Rescale boxes from img_size to im0 size
        detections[:, :4] = scale_coords(img.shape[1:], pred[:, :4], frame.shape).round()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = np.array(pilimg)
        #pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        #pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        #unpad_h = img_size - pad_y
        #unpad_w = img_size - pad_x
        #print(detections)
        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                #print(x1)
                #print(y1)
                #print(x2)
                #print(y2)
                #box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                #box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                #y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                #x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                color = colors[int(obj_id) % len(colors)]
                # color = [i * 255 for i in color]
                cls = classes[int(cls_pred)]
                make_box(detections, frame, cls, color)
                #cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                #cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                #cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        out.write(frame)
        
        fig= plt.figure(figsize=(12, 8))
        plt.title("Video Stream")
        plt.imshow(frame)
        plt.show()
        clear_output(wait=True)
    out and out.release()
    vid.release()




"""
aha = Opt()
print("aha", aha)
load_model(aha)
test_function()
"""


# infer_pics()