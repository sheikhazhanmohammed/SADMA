'''
Author: Azhan Mohammed
Email: azhanmohammed1999@gmail.com
Python: 3.7.10
Description: Trains the model, currently can train normal Unet model or Attention based UNet model
'''

import os
import ast
import sys
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname as up

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import *
from dataloader import GenDEBRIS, bands_mean, bands_std, RandomRotationTransform , class_distr, gen_weights
from metrics import Evaluation
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--trainBatchSize', type=int, default=8, help="batch size to use to train the model")
parser.add_argument('--testBatchSize', type=int, default=4, help="batch size to use to test the model")
parser.add_argument('--trainOnGPU', type=bool, default=False, help="True if train on GPU, else false")
parser.add_argument('--totalNumberOfEpochs', type=int, default=50, help="total number of training epochs")
parser.add_argument('--experimentName', type=str, required=True, help="path where to save the experiment logs")
parser.add_argument('--initialLearningRate', type=int, default=1e-3, help="initial learning rate to train the model")
parser.add_argument('--decayLearningRate', type=int, default=0, help="learning rate decay, helps stablizing model")
parser.add_argument('--learningRateScheduler', type=str, default="ms", help="learning rate scheduler, can be either rop or ms")
parser.add_argument('--trainOnMac', type=bool, default=False, help="True if training on a Mac Device with Metal GPU support")
parser.add_argument('--modelName', type=str, default="resattunet", help="Model architecture to train, currently supports: unet, attunet, resattunet")


args = parser.parse_args()
batchSizeTrain = args.trainBatchSize
batchSizeTest = args.testBatchSize
trainOnGPU = args.trainOnGPU
totalEpochs = args.totalNumberOfEpochs
logPath = args.experimentName
initialLR = args.initialLearningRate
decaryLR = args.decayLearningRate
schedulerLR = args.learningRateScheduler
bestValidationAccuracy = 0.0
macTrain = args.trainOnMac
modelName = args.modelName

logPath = "./"+logPath
os.makedirs(logPath)
writer = SummaryWriter(logPath)

def seedAll(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seedWorker(worker_id):
    # DataLoader Workers Reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seedAll(0)
g = torch.Generator()
g.manual_seed(0)
agg_to_water = True

transformTrain = transforms.Compose([transforms.ToTensor(),
                                    RandomRotationTransform([-90, 0, 90, 180]),
                                    transforms.RandomHorizontalFlip()])
    
transformTest = transforms.Compose([transforms.ToTensor()])
    
standardization = transforms.Normalize(bands_mean, bands_std)

datasetTrain = GenDEBRIS('train', transform=transformTrain, standardization = standardization, agg_to_water = agg_to_water)
datasetTest = GenDEBRIS('val', transform=transformTest, standardization = standardization, agg_to_water = agg_to_water)
        
trainLoader = DataLoader(datasetTrain, 
                        batch_size = batchSizeTrain, 
                        shuffle = True,
                        worker_init_fn=seedWorker,
                        generator=g)
        
testLoader = DataLoader(datasetTest, 
                        batch_size = batchSizeTest, 
                        shuffle = False,
                        worker_init_fn=seedWorker,
                        generator=g)

if trainOnMac:
    device = torch.device("mps")
else:
    if trainOnGPU:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

if modelName == "resattunet":
    model = UNet(11, 11)
elif modelName == "attunet":
    model = 
elif modelName == "unet":
    model = 
else:
    print("Enter correct choice of architecture")
    exit
