import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from CLIPyourFood.Data.IngredientsLoader import IngredientsDataset, TRANSFORMS
from CLIPyourFood.model.ResNet import NUM_CATRGORIES, model_urls,ResNet
from CLIPyourFood.model.ResNet_w_concat_connection import ResNet as concatResnet

THRESHOLD = 0.8


def predict(outputs, threshold=THRESHOLD):
    """
    Predict function for the ingredients.
    """
    predicted = torch.sigmoid(outputs)
    predicted[predicted >= threshold] = 1
    predicted[predicted < threshold] = 0
    return predicted


def accuracy(torch_sum, batch_size, categories_num=NUM_CATRGORIES):
    '''
    Accuracy calculated of the current batch.
    '''
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


def load_data_in_sections(dataset_dir_path, json_dict, transforms=TRANSFORMS, batch_size=32):
    '''
    Load the data from paths and return dataloaders.
    '''
    train_dataloader , val_dataloader, test_dataloader = None, None, None
    if 'train' in json_dict:
        train_dataset = IngredientsDataset(json_dict['train'], dataset_dir_path, transform=transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if 'val' in json_dict:
        val_dataset = IngredientsDataset(json_dict['val'], dataset_dir_path, transform=transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if 'test' in json_dict:
        test_dataset = IngredientsDataset(json_dict['test'], dataset_dir_path, transform=transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


def load_model(w_clip=False, model_path=None,other_connection_method=False):
    '''
    Load Resnet18 from local checkpoint or download pretrained.
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if other_connection_method:
        net=concatResnet(depth=18, clip_flag=True)
    else:
        net = ResNet(depth=18, clip_flag=w_clip)
    if model_path:
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, NUM_CATRGORIES)
        net.load_state_dict(torch.load(model_path), strict=False)
    else:
        net.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet18']), strict=False)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, NUM_CATRGORIES)
    net = net.to(device)
    return net


def freeze_original_resnet(model):
    '''
    Freeze Resnet layers for training only the additions.
    '''
    for name, param in model.named_parameters():
        if 'fc_clip_addition' not in name:
            param.requires_grad = False
    return model

# def eval_mode_net(model, num_classes=NUM_CATRGORIES):
#     '''
#     Eval model as Resnet Original architecture (remove the additional layers)
#     '''
#     new_model = ResNet(depth=18, clip_flag=False)
#     num_ftrs = new_model.fc.in_features
#     new_model.fc = nn.Linear(num_ftrs, num_classes)
#     new_model.load_state_dict(model.state_dict(), strict=False)
#     new_model = new_model.cuda()
#     return new_model.eval()
