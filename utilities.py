#!/usr/local/bin/python3
import numpy as np
import torch
from torchvision import datasets, transforms, models
from PIL import Image

    # Define the transforms for the training, validation, and testing sets
data_transforms = {
    'train':transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                                'valid':transforms.Compose([transforms.Resize((224,224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                                    'test':transforms.Compose([transforms.Resize((224,224)),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
                    }

def set_directories(input):
    data_dir = input
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    dirs = {'train' : train_dir, 'valid' : valid_dir, 'test' : test_dir}
    return dirs

def set_datasets(input):
    # TODO: Load the datasets with ImageFolder
    dataset = {x : datasets.ImageFolder(input[x],transform = data_transforms[x]) for x in ['train', 'valid', 'test']}
    return dataset

def set_dataloaders(input):
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x : torch.utils.data.DataLoader(input[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}
    return dataloaders

def dataset_sizes(input):
    # dataset sizes
    new = {x : datasets.ImageFolder(input[x],transform = data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(new[x])
        for x in ['train', 'valid', 'test']}
    return dataset_sizes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)

    if im.size[0] > im.size[1]:
        im.thumbnail((1000, 256))
    else:
        im.thumbnail((256, 1000))
        
    width,height = im.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)/255.0
    np_image = (np_image - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    np_image = np_image.transpose(2,0,1)
    return np_image