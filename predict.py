#!/usr/local/bin/python3
import argparse
import numpy as np
import torch
from torchvision import models
from torch import nn
from collections import OrderedDict
import json
# utilities file
import utilities
import matplotlib.pyplot as plt

predict_parser = argparse.ArgumentParser()
predict_parser.add_argument('image_path')
predict_parser.add_argument('checkpoint')
predict_parser.add_argument('--top_k', type=int, default=1)
predict_parser.add_argument('--category_names', default='')
predict_parser.add_argument('--gpu', action='store_true', default=False)

args = predict_parser.parse_args()

model_list = {'alexnet' : models.alexnet(pretrained=True), 'vgg16' : models.vgg16(pretrained=True)}

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = model_list[checkpoint['arch']]
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    if args.gpu:
        return model.cuda()
    else:
        return model

def predict(image_path, model, topk=args.top_k, device ='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    processed = utilities.process_image(image_path)
    processed = torch.from_numpy(processed).type(torch.FloatTensor)
    processed.unsqueeze_(0)
    
    # send to cuda if selected
    if args.gpu:
        processed = processed.to(device)
    log_output = model.forward(processed)
    probs = torch.exp(log_output)
    top_probs, top_labs = probs.topk(topk)[0].detach().cpu().numpy().tolist()[0], probs.topk(topk)[1].detach().cpu().numpy().tolist()[0]
    inv_map = {v: k for k, v in model.class_to_idx.items()}
    top_labs = [inv_map[top_labs[x]] for x in range(len(top_labs))]
    print("Image Class: {}".format(top_labs))
    print("Probability: {}".format(top_probs))
    return top_probs, top_labs

def labs_to_cats(labs,json_file):
    if json_file:
        with open(json_file, 'r') as f:
            cat_to_name = json.load(f)
        flowers = [cat_to_name[labs[i]] for i in range(len(labs))]
        print("Category: {}".format(flowers))

model = load_checkpoint(args.checkpoint)
probs, labs = predict(args.image_path, model)
labs_to_cats(labs,args.category_names)