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


# def load_model_data(dataset_dir_path, json_path, transforms):
