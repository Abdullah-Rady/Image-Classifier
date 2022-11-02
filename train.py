import numpy as np
import pandas as pd

from collections import OrderedDict
import time
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models



data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=255),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

data = {
    'train': datasets.ImageFolder(train_dir, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=image_transforms['test'])
}

trainloader = torch.utils.data.DataLoader(data['train'], batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(data['valid'], batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(data['test'], batch_size=64, shuffle=True)


arch='vgg16'
hidden_units=4096
learning_rate=0.01
epochs=5
device='cuda'

parser = argparse.ArgumentParser(description="Choose model's architecture, num of hidden layes,learning rate,num of training epochs,CPU or GPU ")
parser.add_argument('-a','--arch',action='store',type=str, help='Choose among 3 pretrained networks',choices=['vgg16','densenet121','alexnet'])
parser.add_argument('-H','--hidden_units',action='store',type=int, help='Select number of hidden units for 1st layer')
parser.add_argument('-l','--learning_rate',action='store',type=float, help='Set the learning rate for the model')
parser.add_argument('-e','--epochs',action='store',type=int, help='Set the number of epochs')
parser.add_argument('-d','--device',action='store',type=str,help='Choose GPU or CPU for training', choices=['cuda','cpu'])
args = parser.parse_args()

if type(args.hidden_units) != type(None):
    hidden_units=args.hidden_units
if type(args.learning_rate) != type(None):
    learning_rate=args.learning_rate
if type(args.arch) != type(None):
    arch=args.arch
if type(args.epochs) != type(None):
    epochs=args.epochs
if type(args.device) != type(None):
    device=args.device    
if torch.cuda.is_available()==False:
    device='cpu'

exec("model = models.{}(pretrained=True)".format(arch))


for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                       ('fc1',nn.Linear(model.classifier[0].in_features,hidden_units)),
                       ('ReLu1',nn.ReLU()),
                       ('Dropout1',nn.Dropout(p=0.2)),
                       ('fc2',nn.Linear(hidden_units,512)),
                       ('ReLu2',nn.ReLU()),
                       ('Dropout2',nn.Dropout(p=0.2)),
                       ('fc3',nn.Linear(512,102)),
                       ('output',nn.LogSoftmax(dim=1))
                       ]))

model.classifier = classifier

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)

def validation(model,validloader,criterion,device=device):
    
    val_loss=0
    accuracy=0
    model.to(device)
    
    for images,labels in validloader:
        
        images,labels= images.to(device),labels.to(device)
        output=model.forward(images)
        loss=criterion(output,labels)
        ps=torch.exp(output)
        
        equality=(ps.max(dim=1)[1]==labels.data)
        
        val_loss += loss.item()
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return val_loss,accuracy

def train(model=model,trainloader=trainloader,epochs=epochs,device=device):
    print_every=40

    model.to(device)

    running_loss = 0

    print('Training started')

    for e in range(epochs):
        
            
        pass_count = 0
        
        for images,labels in iter(trainloader):
        
            model.train()
        
            pass_count += 1
            images, labels = images.to(device),labels.to(device)
        
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()                

            running_loss += loss.item()
        
            if pass_count % print_every == 0:
            
                model.eval()
            
                with torch.no_grad():
                    val_loss,accuracy=validation(model,validloader,criterion,"cuda")
            
                print("\nEpoch: {}/{} ".format(e+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/print_every))
                print("Validation Loss: {:.4f}  ".format(val_loss/len(validloader)),
                "Accuracy: {:.4f}".format(accuracy/len(validloader)))
            
                model.train()

            running_loss = 0

train()

def save_checkpoint(model):
    
    model.class_to_idx = data['train'].class_to_idx

    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'hidden_layer':hidden_units ,
                  'arch': arch,
                  'learning_rate':learning_rate,
                  'drop_p':0.2,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
save_checkpoint(model)

