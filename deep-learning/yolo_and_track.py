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
from send_data import *

from sort import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import glob

os.chdir("/home/rossis/sort")
from models import *
from utils import *
os.chdir("..")


videopath = "/home/rossis/yolov5/data/videos/short_butterfly.mp4"

 
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
    
    
    
while vid.isOpened():
    aha = Opt(weights = "insect.pt", conf_thres =  0.3) # weights = 'insect.pt'
    load_model(aha)
    img_size = get_img_size()
    classes = get_names()
    destination = "PHOTOS/"       
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/rossis/sort/results/RESULT.mp4',fourcc, floor(fps), (frame_width,frame_height))
    for ii in range(nb_frames - 1):
        print(ii)
        ret, frame = vid.read()
        try:
            frame_to_crop = frame.copy()
            try: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Color error")
            pilimg = Image.fromarray(frame)
            img = transform_frame(frame)
            pred = return_pred(img)[0]

            detections = shape_pred(pred)
            # Rescale boxes from img_size to im0 size
            detections[:, :4] = scale_coords(img.shape[1:], pred[:, :4], frame.shape).round()

            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except:
                print("Color error")
            img = np.array(pilimg)

            if detections is not None:
                tracked_objects, ages = mot_tracker.update(detections.cpu())
                i = 0
                ages = ages[::-1]
                for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                    color = colors[int(obj_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cls = classes[int(cls_pred)] + "-" + str(int(obj_id))
                    # ahaha = torch.unsqueeze(detections[i,:], 0)
                    ahaha = torch.Tensor([x1, y1, x2, y2, 0.0, 0.0, 0.0])
                    ahaha = torch.unsqueeze(ahaha, 0)
                    # print(frame)
                    # print(coords)
                    coords = make_box(ahaha, frame, cls, color)
                    repo_name = f"{cls}"
                    repo_files = os.listdir("/home/rossis/PHOTOS")
                    if ages[i] == 1:
                        name = f"{cls}_0.jpg"
                        if repo_name in repo_files:
                            crop_and_send2(frame_to_crop, coords, destination = destination+repo_name+"/", name = name)
                        else:
                            os.mkdir(destination+repo_name)
                            crop_and_send2(frame_to_crop, coords, destination = destination+repo_name+"/", name = name)
                    
                    if ages[i] % 20 == 0:
                        j = np.floor(ages[i]//20)
                        name = f"{cls}_{int(j)}.jpg"
                        if repo_name in repo_files:
                            crop_and_send2(frame_to_crop, coords, destination = destination+repo_name+"/", name = name)
                        else:
                            os.mkdir(destination+repo_name)
                            crop_and_send2(frame_to_crop, coords, destination = destination+repo_name+"/", name = name)
                    i += 1

            out.write(frame)
            fig= plt.figure(figsize=(12, 8))
            plt.title("Video Stream")
            plt.imshow(frame)
            plt.show()
            clear_output(wait=True)
        except:
            print("none type copy error")
    out and out.release()
    vid.release() 