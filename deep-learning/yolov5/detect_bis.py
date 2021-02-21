#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:50:59 2021

@author: sophierossi
"""
import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy

from models.experimental import attempt_load
from utils.datasets import *
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class Opt:
    def __init__(self, weights = 'yolov5s.pt', source = "./data/videos", img_size = 640, conf_thres =  0.25,
                 iou_thres = 0.45, device = "", view_img = False, save_txt = False, save_conf = False, classes = None,
                 agnostic_nms = False, augment = False, project = './runs/detect', name = "exp", exist_ok = False):
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.project = project
        self.name = name
        self.exist_ok = exist_ok

def get_img_size():
    return imgsz

def get_classes():
    return classes

def get_names():
    return names

def get_colors():
    return colors

def shape_pred(pred):
    """
    yolov5 gives predictions of the shape: (x1, y1, x2, y2, conf, cls)
    we want a prediction of shape: (x1, y1, x2, y2, object_conf, class_score, class_pred)
    for the moment we replce class_score by 1
    """
    # print(pred)
    nb_lines = list(pred.size())[0]
    c = [1 for i in range(nb_lines)]
    c = numpy.array(c)
    c = numpy.reshape(c, (nb_lines, 1))
    c = torch.from_numpy(c).float().to(device)
    # print(c)
    
    b = torch.cat((pred[:, :5], c), 1)
    return torch.cat((b, torch.reshape(pred[:,5], (nb_lines, 1))),1)
    
def load_model(aha):
    # loads model until 1st inference
    ini_path = os.getcwd()
    
    os.chdir("./yolov5") # pour yolo_and_track
    # os.chdir("./rossis/yolov5") # pour detect_jetson
    
    
    # definition of global variables
    global source, weights, view_img, save_txt, imgsz, webcam, save_dir, device, model, dataset, vid_path, vid_writer, save_img, _
    global conf_thres, iou_thres, save_conf, augment, project, name, exist_ok, agnostic_nms
    global half, classes, classify
    global names, colors, t0
    
    
    opt = aha
    save_img=False # previously was a parameter
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    conf_thres, iou_thres, save_conf, augment, project = opt.conf_thres, opt.iou_thres, opt.save_conf, opt.augment, opt.project
    name, exist_ok, agnostic_nms = opt.name, opt.exist_ok, opt.agnostic_nms
    classes = opt.classes
    
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once    
    print("bye")
    os.chdir(ini_path)
    
def return_pred(img):
    ini_path = os.getcwd()
    # os.chdir("./yolov5")
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic= agnostic_nms)
    t2 = time_synchronized()
    
    return pred
    
def test_function():
    for path, img, im0s, vid_cap in dataset:
        print(return_pred(img))

def transform_frame(frame):
    # Padded resize
    img = letterbox(frame, new_shape = imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img

def make_box(det, frame, label, color):
    coords = []
    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, score, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            # print(xywh)
            coords.append(xyxy)
            plot_one_box(xyxy, frame, label=label, color= color, line_thickness=3)
    return coords

def infer_pics():
    """
    returns detections of the shape: 
    (x1, y1, x2, y2, object_conf, class_score, class_pred)
    eg: tensor([[123.3672, 361.0618, 137.2583, 401.2100,   0.8687,   0.9987,   0.0000],
        [ 88.8490, 366.1522, 118.7245, 410.6961,   0.9725,   0.9898,   2.0000]],
       device='cuda:0')
    """
    ini_path = os.getcwd()
    os.chdir("./yolov5")
    # definition of global variables
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic= agnostic_nms)
        print(pred)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    os.chdir(ini_path)
    
"""
# main d'exemple: fonctionne
aha = Opt()
print("aha", aha)
load_model(aha)
infer_pics()
"""
