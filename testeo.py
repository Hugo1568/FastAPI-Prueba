from fastapi import FastAPI, File, UploadFile, Form
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import PIL
import glob
import cv2
import numpy as np

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


imagefile = cv2.imread('0511211825.jpg')
    
load_data = torch.load('data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1]

img = Image.fromarray(imagefile)
    #print(img)
img_cropped, prob = mtcnn(img, return_prob=True) 
    
      
if img_cropped is not None:               
    if prob>0.90:
        emb = resnet(img_cropped[0].unsqueeze(0)).detach() 
                    
        dist_list = [] # list of matched distances, minimum distance is used to identify the person
                    
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        min_dist = min(dist_list) # get minumum dist value
        min_dist_idx = dist_list.index(min_dist) # get minumum dist index
        name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                    
                    
                    
        if min_dist<0.90:
            print(name + ' ' + str(min_dist))
            print("Mientras mas cerca del cero, mas se parece a " + name + " en la base de datos proporcionada")
        else:
            print("No match")
    else:
        print("Probabilidad muy baja de que sea una cara")
else:
    print("No se detecto una cara")
