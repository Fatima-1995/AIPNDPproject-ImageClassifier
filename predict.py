import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from PIL import Image
import numpy as np
from torch.autograd import Variable
import argparse
from utility import get_current_device, load_model, read_flowers_names


def main():
    in_args = setup_predict_args()
    model = load_model(in_args.model_path)
    predict(in_args.image_path, model, in_args.topk, in_args.enable_gpu, in_args.names_path)

def setup_predict_args():

    parseArgs = argparse.ArgumentParser(description = "Images prediction Arguments")
    parseArgs.add_argument('--image_path', default = 'flowers/test/50/image_06541.jpg', type = str, help = "Images test directory")
    parseArgs.add_argument('--model_path', default = 'checkpoint.pth', type = str, help = "Saved model directory")
    parseArgs.add_argument('--enable_gpu', default = 'cuda', help = "Enable gpu if available")
    parseArgs.add_argument('--topk', default = 5, type = int, help = "top probs for the image")
    parseArgs.add_argument('--names_path', default = 'cat_to_name.json', type = str, help = "json file directory")
    return parseArgs.parse_args()

def process_image(image):

    pil_image = Image.open(image)
   
    apply_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    img_to_tensor = apply_transformations(pil_image)
    return img_to_tensor.numpy()

def predict(image_path, model, topk=5, gpu = False, names_path = 'cat_to_name.json'):

    device = get_current_device()
    if gpu and device.type == 'cuda':
        model.to(device)
    else:
        device = 'cpu'
        model.to(device)
    img_torch = process_image(image_path)
    image_converted = Variable(torch.from_numpy(img_torch))
    image_converted = image_converted.unsqueeze(0)
    
    with torch.no_grad():
        image_converted = image_converted.type_as(torch.FloatTensor()).to(device)
        output = model.forward(image_converted)
        output = torch.exp(output.cpu())
        
        probs, classes = output.topk(topk)
        
        probs = probs.data.numpy().tolist()[0]
        classes = classes.data.numpy().tolist()[0]
        cat_to_name = read_flowers_names(names_path)
       
        classes_to_idx ={idx:oid for oid,idx in model.class_to_idx.items()}
        classes_name = [cat_to_name[classes_to_idx[i]] for i in classes]
        
        print('These are the top 5 predictions')
        for c, p in zip(classes_name, probs):
            print("{}  : {:.2%}".format(c,p))
   


if __name__== '__main__':
        main()