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
from utility import load_model, setup_network, get_current_device, check_accuracy_on_test
import workspace_utils

def main():
    
    #Get arguments
    in_args = setup_train_args()
    
    train_loader, test_loader, valid_loader, train_data = setup_data(in_args.data_dir)
    model, criterion, optimizer = setup_network(in_args.arch,in_args.learn_rate,in_args.hidden_layers)
    train_model(model, criterion, optimizer, train_loader, valid_loader, in_args.epochs, in_args.enable_gpu)
    save_model(in_args.save_dir,train_data, model, in_args.arch, in_args.learn_rate, in_args.hidden_layers)
    ret_model = load_model(in_args.save_dir)
    
    check_accuracy_on_test(test_loader, ret_model)
    

def setup_train_args():

    #Define arguments
    parseArgs = argparse.ArgumentParser(description = "Train model")
    parseArgs.add_argument('--data_dir', default = 'flowers', type = str, help = "Data location'")
    parseArgs.add_argument('--save_dir', default = 'checkpoint.pth', type = str, help = "The path to chackpoint")
    parseArgs.add_argument('--arch', default = 'vgg13', type = str, help = "Choose Network Model - Vgg13, alexnet'")
    parseArgs.add_argument('--learn_rate', default = 0.01, type = float, help = "The learning rate during training")
    parseArgs.add_argument('--hidden_layers', default = 512, type = int, help = "The Hiddien layers of network")
    parseArgs.add_argument('--epochs', default = 5, type = int , help = "Number of epochs in the training")
    parseArgs.add_argument('--enable_gpu', default = 'cuda', help = "use GPU if available")
    
    return parseArgs.parse_args()

def setup_data(data_dir='flowers'):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(35),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size = 32, shuffle = True)
    
    return trainloader, testloader, validloader, train_data

def train_model(model, criterion, optimizer, train_data_loader, valid_data_loader, epochs = 5, gpu = False):

    epochs = 5
    print_every = 30
    steps = 0

    device = get_current_device()
    if gpu and device.type == 'cuda':
        model.to(device)
    else:
        device = 'cpu'
        model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_data_loader):
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            # Forward
            output = model.forward(inputs)
            loss = criterion(output, labels)
            # Backward
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                
                valid_loss, accuracy = validation(model, valid_data_loader, criterion, optimizer)
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.3f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss),
                      'Validation Accuracy: %d %%' % (100 * accuracy))
            
                running_loss = 0
                model.train()
                           
def validation(model, validloader, criterion, optimizer):    
    model.eval()
    valid_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(validloader):
        optimizer.zero_grad()
        
        device = get_current_device()
        inputs, labels = inputs.to(device), labels.to(device)
        model.to(device)
        
        with torch.no_grad():
            output = model.forward(inputs)
            valid_loss = criterion(output, labels)
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()
            
    valid_loss = valid_loss / len(validloader)
    accuracy = accuracy / len(validloader)
    return valid_loss, accuracy
             
def save_model(path, train_data, model, arch = 'vgg13', lr = 0.01, hidden_layers = 512):
    model.class_to_idx = train_data.class_to_idx
    
    model_checkpoint = {'input_size': 25088,
                        'output_size': 102,
                        'arch': 'vgg13',
                        'learning_rate': 0.01,
                        'batch_size': 64,
                        'epochs': 5,
                        'state_dict': model.state_dict(),
                        'class_to_idx': model.class_to_idx}

    torch.save(model_checkpoint, path)
    
    print("The model has been saved successfully")
    
    
    
if __name__== '__main__':
        main()