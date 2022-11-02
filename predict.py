import numpy as np
import pandas as pd

from collections import OrderedDict
import json
from PIL import Image
import os
import argparse
import timeit

import torch
from torch import nn
import torchvision
from torchvision import models


top_k=5
image_path='flowers/test/10/image_07090.jpg'
checkpoint_path='checkpoint.pth'
json_path = 'cat_to_name.json'    
device='cuda'

parser = argparse.ArgumentParser(description="Enter a valid Image path,Checkpoint path and Top K calasses to view ")
parser.add_argument('-img','--image_path',action='store',type=str, help='Path of your image')
parser.add_argument('-Ch','--checkpoint_path',action='store',type=str, help="Checpoint's path that contain your model")
parser.add_argument('-tk','--top_k',action='store',type=int, help='top K most probably classes for image')
parser.add_argument('-d','--device',action='store',type=str, help='Choose GPU or CPU for predictions',choices=['cuda','cpu'])
parser.add_argument('-f','--json',action='store',type=str, help='path of the file holding plants names')
args = parser.parse_args()

if type(args.image_path) != type(None):
    if os.path.isfile(args.image_path):
        image_path=args.image_path
    else:
        print("Wrong path for img")
        
if type(args.checkpoint_path) != type(None):
    if os.path.isfile(args.checkpoint_path):
        checkpoint_path=args.checkpoint_path
    else:
        print("Wrong path for checkpoint")

if type(args.json) != type(None):
    if os.path.isfile(args.json):
        json_path=args.json
    else:
        print("Wrong path for the file")
        
if type(args.top_k) != type(None):
    top_k=args.top_k
    
if type(args.device) != type(None):
    device=args.device    
if torch.cuda.is_available()==False:
    device='cpu'

    
def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
                       ('fc1',nn.Linear(checkpoint['input_size'],checkpoint['hidden_layer'])),
                       ('ReLu1',nn.ReLU()),
                       ('Dropout1',nn.Dropout(p=checkpoint['drop_p'])),
                       ('fc2',nn.Linear(checkpoint['hidden_layer'],512)),
                       ('ReLu2',nn.ReLU()),
                       ('Dropout2',nn.Dropout(p=checkpoint['drop_p'])),
                       ('fc3',nn.Linear(512,checkpoint['output_size'])),
                       ('output',nn.LogSoftmax(dim=1))
                       ]))
    
    model.classifier=classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

model=load_checkpoint(checkpoint_path)

with open(json_path, 'r') as f:
        cat_to_name = json.load(f)

def process_image(image_path):
    
    im = Image.open(image_path)
    
    im = im.resize((224, 224), Image.ANTIALIAS)
    
    np_image = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk, device):
    
    image = torch.from_numpy(np.expand_dims(process_image(image_path),axis=0)).type(torch.FloatTensor).to(device)
    
    model.eval()
    model.to(device)
    ps=torch.exp(model.forward(image))
    top_kps, top_klabels = ps.topk(topk)
    
    top_kps = top_kps.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_klabels = top_klabels.detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_klabels]
    
    return top_kps,top_classes 

start = timeit.default_timer()

pred = predict(image_path,model,top_k,device)
print('Hieghest K probabilities: ',pred[0])
print('Corresponding labels: ',pred[1])

names_of_pred=[cat_to_name[pred[1][x]] for x in range(top_k)]
print('Names of labels: ',names_of_pred)

stop = timeit.default_timer()
print('Time: ', stop - start)  