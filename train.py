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
decayLR = args.decayLearningRate
schedulerLR = args.learningRateScheduler
bestValidationAccuracy = 0.0
trainOnMac = args.trainOnMac
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
    model = ResidualAttentionUNet(11, 11)
elif modelName == "attunet":
    model = AttentionUNet(11, 11)
elif modelName == "unet":
    model = UNet(11, 11)
else:
    print("Enter correct choice of architecture")
    exit()

model.to(device)

if agg_to_water:
    agg_distr = sum(class_distr[-4:])
    class_distr[6] += agg_distr
    class_distr = class_distr[:-4]

weight = gen_weights(class_distr, c = 1.03)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction= 'mean', weight=weight.to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=initialLR, weight_decay=decayLR)

# Learning Rate scheduler
if schedulerLR=="rop":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40,80,120,160], gamma=0.5, verbose=True)

bestMacroF1 = 0.0
bestMicroF1 = 0.0
bestWeightF1 = 0.0

i = 0
for epoch in range(1, totalEpochs+1):
    trainingBatches = 0
    model.train()
    print("Training for epoch:",epoch)
    for (image, target) in tqdm(trainLoader):
        image = image.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        logits = model(image)
        loss = criterion(logits, target)
        loss.backward()
        trainingBatches+=target.shape[0]
        writer.add_scalar('Training Loss', loss, i)
        i = i + 1
        optimizer.step()
    print("Completed epoch:",epoch)
    print("Validating model")
    model.eval()
    testBatches = 0
    yTrue = []
    yPredicted = []
    testLossF = []
    with torch.no_grad():
        for (image, target) in testLoader:
            image = image.to(device)
            target = target.to(device)
            logits = model(image)
            loss = criterion(logits, target)
            logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
            logits = logits.reshape((-1,11))
            target = target.reshape(-1)
            mask = target != -1
            logits = logits[mask]
            target = target[mask]
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            target = target.cpu().numpy()
            testBatches += target.shape[0]
            testLossF.append((loss.data*target.shape[0]).tolist())
            yPredicted += probs.argmax(1).tolist()
            yTrue += target.tolist()
        writer.add_scalar('Testing Loss', sum(testLossF)/testBatches, epoch)
        yPredicted = np.asarray(yPredicted)
        yTrue = np.asarray(yTrue)
        acc = Evaluation(yPredicted, yTrue)
        modelname = "savedModels/att-unet/intermediateModel.pth"
        torch.save(model.state_dict(), modelname)
        print("Test Macro Precision",acc["macroPrec"])
        writer.add_scalar('Test Macro Precision', acc["macroPrec"], epoch)
        writer.add_scalar('Test Micro Precision', acc["microPrec"], epoch)
        writer.add_scalar('Test Weight Precision', acc["weightPrec"], epoch)

        print("Test Macro Recall",acc["macroPrec"])
        writer.add_scalar('Test Macro Recall', acc["macroRec"], epoch)
        writer.add_scalar('Test Micro Recall', acc["microRec"], epoch)
        writer.add_scalar('Test Weight Recall', acc["weightRec"], epoch)

        print("Test Macro F1",acc["macroF1"])
        writer.add_scalar('Test Macro F1', acc["macroF1"], epoch)
        if acc["macroF1"]>bestMacroF1:
          bestMacroF1 = acc["macroF1"]
          modelname = "savedModels/att-unet/bestMacroF1Model.pth"
          torch.save(model.state_dict(), modelname)
        writer.add_scalar('Test Micro F1', acc["microF1"], epoch)
        if acc["microF1"]>bestMicroF1:
          bestMicroF1 = acc["microF1"]
          modelname = "savedModels/att-unet/bestMicroF1Model.pth"
          torch.save(model.state_dict(), modelname)
        writer.add_scalar('Test Weight F1', acc["weightF1"], epoch)
        if acc["weightF1"]>bestWeightF1:
          bestWeightF1 = acc["microF1"]
          modelname = "savedModels/att-unet/bestWeightF1Model.pth"
          torch.save(model.state_dict(), modelname)

        writer.add_scalar('Test Macro IoU', acc["IoU"], epoch)
        print("Test Macro IoU",acc["IoU"])
    if schedulerLR=="rop":
        scheduler.step(sum(testLossF) / testBatches)
    else:
        scheduler.step()
