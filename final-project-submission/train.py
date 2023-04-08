#!/usr/local/bin/python3
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from collections import OrderedDict
# utilities file
import utilities
import os

parser = argparse.ArgumentParser()
parser.add_argument('directory')
parser.add_argument('--arch', default='vgg16', help='Choose an architecture: alexnet, vgg16')
parser.add_argument('--save_dir')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int, default=4096)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

# set image directories
dirs = utilities.set_directories(args.directory)

# load datasets with ImageFolder
data = utilities.set_datasets(dirs)

# define dataloaders from image datasets and transforms
dataloaders = utilities.set_dataloaders(data)

model_list = {'alexnet' : models.alexnet(pretrained=True), 'vgg16' : models.vgg16(pretrained=True)}

# Build and train the network

# define the model
model = model_list[args.arch]

# freeze the features
for param in model.parameters():
    param.requires_grad = False

# set classifier for model
if args.arch == 'alexnet':
    input_size = 9216
else:
    input_size = 25088
classifier = nn.Sequential(OrderedDict([
                                        ('fc1', nn.Linear(input_size, args.hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('fc2', nn.Linear(args.hidden_units, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))

model.classifier = classifier

# set criterion
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# set scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, epochs=args.epochs, device = 'cuda'):
    
    # change to device
    if args.gpu:
        model.to(device)
    
    for e in range(epochs):
        print("Epoch: {}/{}... ".format(e+1, epochs))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0
            running_corrects = 0
            
            for ii, (inputs, labels) in enumerate(dataloaders[phase]):
                
                # Move input and label tensors to the GPU
                if args.gpu:
                    inputs, labels = inputs.to(device), labels.to(device)
                
                # zero out gradient
                optimizer.zero_grad()
                
                # Forward and backward passes
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
        
                # stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / utilities.dataset_sizes(dirs)[phase]
            epoch_acc = running_corrects.double() / utilities.dataset_sizes(dirs)[phase]
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
        print()
    return model

def save_checkpoint(model):
    model.class_to_idx = data['train'].class_to_idx
    checkpoint = {'input_size': input_size,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_dict': optimizer.state_dict(),
                  'hidden_layers': args.hidden_units,
                  'arch': args.arch
                  }
    if args.save_dir:
        os.chdir(args.save_dir)
        torch.save(checkpoint, 'checkpoint.pth')
    else:
        torch.save(checkpoint, 'checkpoint.pth')

train_model(model, criterion, optimizer, scheduler)
save_checkpoint(model)