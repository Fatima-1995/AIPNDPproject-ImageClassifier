import numpy as np
import argparse
from collections import OrderedDict
from PIL import Image
import os, random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import json
import time
import workspace_utils


def read_flowers_names(file_path = 'cat_to_name.json'):
    with open(file_path) as f:
        cat_to_name = json.load(f)
        return cat_to_name
    
def get_current_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def load_model(path):

    checkpoint = torch.load(path)
    model, criterion, optimizer = setup_network(checkpoint['arch'], checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    print('The model has been loaded successfully')
    return model

def setup_network(arch = 'vgg13', lr = 0.01, hidden_layers = 512):

    archs = {"vgg13":25088,
            "alexnet":9216}
    
    inputs_layers = archs[arch]
    
    if arch == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Choose one of the available architecture please")
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(25088, 1024)),
                           ('relu', nn.ReLU()),
                           ('drop',nn.Dropout(p=0.5)),
                           ('fc2', nn.Linear(1024, 102)),
                           ('output', nn.LogSoftmax(dim = 1))]))
    
    model.classifier = classifier
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
    
    return model, criterion, optimizer

def check_accuracy_on_test(testloader, model): 
    correct = 0
    total = 0
    device = get_current_device()
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Network Accuracy: %d %%' % (100 * correct / total))
    