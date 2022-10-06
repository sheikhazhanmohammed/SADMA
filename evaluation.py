import os
import sys
import random
import logging
import rasterio
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname as up

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import *
from dataloader import GenDEBRIS, bands_mean, bands_std

from metrics import Evaluation, confusion_matrix
from assets import labels

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def main(options):
    # Transformations
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)
    
    # Construct Data loader

    dataset_test = GenDEBRIS('test', transform=transform_test, standardization = standardization, agg_to_water = options['agg_to_water'])

    test_loader = DataLoader(   dataset_test, 
                                batch_size = options['batch'], 
                                shuffle = False)
    
    global labels
    # Aggregate Distribution Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water
    if options['agg_to_water']:
        labels = labels[:-4] # Drop Mixed Water, Wakes, Cloud Shadows, Waves

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = AttentionUNet(11, 11)
    model.load_state_dict(torch.load("/content/drive/MyDrive/marida/savedModels/att-unet/bestMacroF1Model.pth", map_location = device))
    model.to(device)

    # Load model from specific epoch to continue the training or start the evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()

    y_true = []
    y_predicted = []
    
    with torch.no_grad():
        for (image, target) in tqdm(test_loader, desc="testing"):
            image = image.to(device)
            target = target.to(device)
            logits = model(image)
            # Accuracy metrics only on annotated pixels
            logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
            logits = logits.reshape((-1,11))
            target = target.reshape(-1)
            mask = target != -1
            logits = logits[mask]
            target = target[mask]
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            target = target.cpu().numpy()
            y_predicted += probs.argmax(1).tolist()
            y_true += target.tolist()
        acc = Evaluation(y_predicted, y_true)
        print("Evaluation: " + str(acc))
        conf_mat = confusion_matrix(y_true, y_predicted, labels)
        print("Confusion Matrix:  \n" + str(conf_mat.to_string()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Options
    parser.add_argument('--agg_to_water', default=True, type=bool,  help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water')
    parser.add_argument('--batch', default=5, type=int, help='Number of images to run in a batch')
    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    main(options)