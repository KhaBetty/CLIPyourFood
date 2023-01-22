import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.models
import torchvision
from torch.utils.data import Dataset, DataLoader
#import transforms
from torchvision import transforms
from Data.IngredientsLoader import IngredientsDataset
import matplotlib.pyplot as plt
from CLIPyourFood.Data.utils import vec2lables
from CLIPyourFood.model.ResNet import ResNet, model_urls
from CLIPyourFood.model.utils import predict, accuracy
import time
import argparse
import os


def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, ingredients_vec ,labels in dataloader:
        inputs = inputs.to(device)
        #convert tuple to list
        labels = list(labels)
        #labels = labels.to(device)
        ingredients_vec = ingredients_vec.to(device)
        outputs = model((inputs,labels)) #TODO debug because not the correct way for batch
        preds = predict(outputs, threshold=0.8)
        loss = criterion(outputs, ingredients_vec)
        running_loss += loss.item() * inputs.size(0)
        running_corrects +=  accuracy(torch.sum(preds == ingredients_vec.data), batch_size, num_classes)
        print('loss', loss)
    epoch_loss = running_loss /  len(dataloader.dataset)  #preds.nelement()
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc


if __name__ == '__main__':

    evaluate_model()