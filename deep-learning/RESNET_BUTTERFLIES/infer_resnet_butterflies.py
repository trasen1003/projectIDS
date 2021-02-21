#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 20:49:04 2021

@author: sophierossi
"""

import os
import argparse
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import json
from models.resnet import Resnet
from settings import CLASSES_FILE, MODEL_WEIGHTS_FOLDER
import pandas as pd
import numpy as np
import random

global device, ALLOWED_EXTENSIONS, classes, model

def set_up():
    ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.jpeg'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open(CLASSES_FILE) as file:
        json_file = json.load(file)
    # List of classes
    classes = json_file['classes']
    return device, ALLOWED_EXTENSIONS, classes

def load_state_dict():
    checkpoint_path = os.path.join(MODEL_WEIGHTS_FOLDER, '29.pth')
    state_dict = torch.load(checkpoint_path, map_location=device)
    return state_dict

def load_resnet(device):
    model = Resnet(629)

    model_state_dict = load_state_dict()
    model.load_state_dict(model_state_dict)

    model.to(device)
    model.eval()
    return model

def allowed_file(filename):
    file_p, ext = os.path.splitext(filename)
    return ext in ALLOWED_EXTENSIONS

class Opt_resnet():
    def __init__(self, img_path, description='Classification'):
        self.description = description
        self.path = img_path

def process(model, image_path):
    if os.path.exists(image_path) and allowed_file(image_path):

        # Results Dictionary
        results_dict_top3 = {}
        results_dict_top1 = {}
        # Reading image
        uploaded_image = Image.open(image_path).convert('RGB')

        resized_image = uploaded_image.resize((224, 224), Image.ANTIALIAS)
        resized_image = ToTensor()(resized_image)

        # Passing image to model
        data = resized_image.to(device)
        output = model(data[None, ...])

        # Getting Results from model
        sm = torch.nn.Softmax()
        probabilities = sm(output)
        confidence_score_top3, class_pred_top3 = torch.topk(probabilities, 3)

        # Parsing confidences and predicted classes
        predicted_classes_top3 = class_pred_top3.cpu().numpy()[0]
        confidences_top3 = confidence_score_top3.cpu().detach().numpy()[0]
        predicted_classes_list_top3 = predicted_classes_top3.tolist()
        confidence_list_top3 = confidences_top3.tolist()

        confidence_score_top1, class_pred_top1 = torch.topk(probabilities, 1)

        # Parsing confidences and predicted classes
        predicted_classes_top1 = class_pred_top1.cpu().numpy()[0]
        confidences_top1 = confidence_score_top1.cpu().detach().numpy()[0]
        predicted_classes_list_top1 = predicted_classes_top1.tolist()
        confidence_list_top1 = confidences_top1.tolist()

        # Adding results to dictionary
        for i, each_class in enumerate(zip(predicted_classes_list_top3, confidence_list_top3)):
            results_dict_top3[i] = {"Confidence": str(float("{:.2f}".format(each_class[1] * 100))) + "%",
                                    "Class": str(classes[each_class[0]])}

        for i, each_class in enumerate(zip(predicted_classes_list_top1, confidence_list_top1)):
            results_dict_top1[i] = {"Confidence": str(float("{:.2f}".format(each_class[1] * 100))) + "%",
                                    "Class": str(classes[each_class[0]])}

        # print("\n\nTop3: ", results_dict_top3, "\n\nTop1: ", results_dict_top1)
        return str(classes[each_class[0]])
    else:
        print("Allowed image types are -> png, jpg, jpeg")
        return None

# loading model 
device, ALLOWED_EXTENSIONS, classes = set_up()
model = load_resnet(device)

# initializing lists
photos_path = "/home/rossis/PHOTOS"
already_done = []
column_names = ["dossier", "img", "date", "espece", "temperature", "humidite", "pression", "luminosite"]
df = pd.DataFrame(columns = column_names)
print("Model loaded - ready for inference")

while True:
    repos = os.listdir(photos_path)
    for repo in repos:
        imgs = os.listdir(photos_path+"/"+repo)
        for img in imgs:
            if (img in already_done) == False:
                try:
                    specie = process(model, photos_path+"/"+repo+"/"+img)
                    t = random.uniform(16.0, 19.5)
                    h = random.uniform(50.0, 55.0)
                    p = random.uniform(0.95, 1.05)
                    h = random.uniform(50.0, 55.0)
                    l = random.uniform(60.0, 65.0)
                    df = df.append({"dossier": repo, "img": img, "espece": specie, "temperature": t, "humidite": h, "pression": p, "luminosite": l}, ignore_index=True)
                    df.to_json("/home/rossis/species.json")
                    df.to_csv("/home/rossis/species.csv")
                    already_done.append(img)
                    print(f"butterfly detected - specie: {specie}")
                except:
                    pass
                    #print("error")
                    
                
                
                
                
                