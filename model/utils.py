import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from CLIPyourFood.Data.IngredientsLoader import IngredientsDataset
from CLIPyourFood.model import ResNet
from CLIPyourFood.model.ResNet import NUM_CATRGORIES


def predict(outputs, threshold=0.5):
    """
    :param outputs: output of the model
    :return: the predicted labels
    """
    predicted = torch.sigmoid(outputs)
    predicted[predicted >= threshold] = 1
    predicted[predicted < threshold] = 0
    return predicted


def accuracy(torch_sum, batch_size, categories_num):
    return torch_sum / (batch_size * categories_num)


def plot_statistics(train_results, val_results, output_path):
    '''
    Plot statistics and save at output path.
    '''
    x_axes = np.arange(1, len(train_results['loss']) + 1)
    # plot accuracy
    fig = plt.figure(figsize=(20, 10))
    plt.title("Train-Validation Accuracy")
    plt.plot(x_axes, train_results['accuracy'], label='train')
    plt.plot(x_axes, val_results['accuracy'], label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    fig.savefig(output_path + '/train_val_acc.png')
    plt.close()
    # plot loss
    fig = plt.figure(figsize=(20, 10))
    plt.title("Train-Validation Loss")
    plt.plot(x_axes, train_results['loss'], label='train')
    plt.plot(x_axes, val_results['loss'], label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    fig.savefig(output_path + '/train_val_loss.png')
    plt.close()


def load_data_in_sections(dataset_dir_path, json_dict, transforms, batch_size):
    '''
    Load the data from paths and return dataloaders.
    '''
    train_dataset = IngredientsDataset(json_dict['train'], dataset_dir_path, transform=transforms)
    val_dataset = IngredientsDataset(json_dict['val'], dataset_dir_path, transform=transforms)
    test_dataset = IngredientsDataset(json_dict['test'], dataset_dir_path, transform=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


def eval_mode_net(model, num_classes=NUM_CATRGORIES):
    '''
    Eval model as Resnet Original architecture (remove the additional layers)
    '''
    new_model = ResNet(depth=18, clip_flag=False)
    num_ftrs = new_model.fc.in_features
    new_model.fc = nn.Linear(num_ftrs, num_classes)
    new_model.load_state_dict(model.state_dict(), strict=False)
    new_model = new_model.cuda()
    return new_model.eval()
