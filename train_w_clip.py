import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.models
import torchvision
from torch.utils.data import Dataset, DataLoader
# import transforms
from torchvision import transforms
from Data.IngredientsLoader import IngredientsDataset
import matplotlib.pyplot as plt
from CLIPyourFood.Data.utils import vec2lables
from CLIPyourFood.model.ResNet import ResNet, model_urls
from CLIPyourFood.model.utils import predict, accuracy, plot_statistics
from CLIPyourFood.model.utils import predict, accuracy, load_data_in_sections, load_model, freeze_original_resnet
import time
import copy
import os

# choose random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# depends on our vector of ingredients
num_classes = 227

# hyperparameters
batch_size = 32
learning_rate = 1e-2
momentum = 0.9
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224))
        # add normilization
    ])
# criterion of loss for multilabel classification
criterion = nn.BCEWithLogitsLoss()
use_cuda = False

# load the dataset

dataset_path = 'food101/train/food-101/images'
json_path = dataset_path + '/ing_with_dish_jsn.json'
json_dict = {'train': 'food101/train/food-101/images/ing_with_dish_jsn_train.json',
             'val': 'food101/train/food-101/images/ing_with_dish_jsn_val.json',
             'test': 'food101/train/food-101/images/ing_with_dish_jsn_test.json'}

output_path = '/home/maya/proj_deep/CLIPyourFood/results/adding_fc_text_encode/resnet18_w_clip'

train_dataloader, val_dataloader, test_dataloader = load_data_in_sections(dataset_path, json_dict, transforms,
                                                                          batch_size)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model
net = load_model(w_clip=True, model_path=None)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# net = freeze_original_resnet(net)


# evaluation of model on test set
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, ingredients_vec, labels in dataloader:
        inputs = inputs.to(device)
        # convert tuple to list
        labels = list(labels)
        # labels = labels.to(device)
        ingredients_vec = ingredients_vec.to(device)
        outputs = model((inputs, labels))  # TODO debug because not the correct way for batch
        preds = predict(outputs, threshold=0.8)
        loss = criterion(outputs, ingredients_vec)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += accuracy(torch.sum(preds == ingredients_vec.data), batch_size, num_classes)
        print('loss', loss)
    epoch_loss = running_loss / len(dataloader.dataset)  # preds.nelement()
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc


# epoch_loss, epoch_acc = evaluate_model(net, test_dataloader, criterion)

n_epochs = 5
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step_train = len(train_dataloader)
total_step_val = len(val_dataloader)
net.train()
for epoch in range(1, n_epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_, labels_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        # data_.requires_grad_()
        optimizer.zero_grad()

        outputs = net((data_, labels_))
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # check loss
        pred = predict(outputs, threshold=0.8)
        correct += accuracy(torch.sum(pred == target_).item(), batch_size,
                            num_classes)  # TODO change to 1 when it all correct , divide by 227
        total += target_.size(0)
        if batch_idx % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, n_epochs, batch_idx, total_step_train, loss.item()))
    train_acc.append(100 * correct / total_step_train)
    train_loss.append(running_loss / total_step_train)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total_step_train):.4f}')
    batch_loss = 0
    total_t = 0
    correct_t = 0
    with torch.no_grad():
        net.eval()
        # eval the model
        # val_net = eval_mode_net(net)
        for data_t, target_t, labels_ in val_dataloader:
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net((data_t, labels_))  # val_net(data_t) #
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            # _,pred_t = torch.max(outputs_t, dim=1)
            pred_t = predict(outputs_t, threshold=0.8)
            correct_t += accuracy(torch.sum(pred_t == target_t).item(), batch_size, num_classes)
            # total_t += target_t.size(0)
        val_acc.append(100 * correct_t / total_step_val)
        val_loss.append(batch_loss / len(val_dataloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_step_val):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), output_path + '/resnet_w_clip.pt')
            print('Improvement-Detected, save-model')
    net.train()
train_results = {'accuracy': train_acc, 'loss': train_loss}
val_results = {'accuracy': val_acc, 'loss': val_loss}
plot_statistics(train_results, val_results, output_path)

# #remove fc of clip
# epoch_loss, epoch_acc = evaluate_model(net, test_dataloader, criterion)
# print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_step_val):.4f}\n')
