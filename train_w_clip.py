import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models
import torchvision
from torch.utils.data import Dataset, DataLoader
#import transforms
from torchvision import transforms
from Data.IngredientsLoader import IngredientsDataset
import matplotlib.pyplot as plt
from CLIPyourFood.Data.utils import vec2lables
from CLIPyourFood.model.ResNet import ResNet, model_urls
import time
import copy
import os

#choose random seed
seed = 42
np.random.seed(seed)

#depends on our vector of ingredients
num_classes = 227


#hyperparameters
batch_size = 2
learning_rate = 1e-2
momentum=0.9
transforms = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Resize((224,224))
    #add normilization
])
#criterion of loss for multilabel classification
criterion = nn.BCEWithLogitsLoss()
use_cuda = False

# load the dataset

dataset_path = 'food101/train/food-101/images'
json_path = dataset_path + '/ing_with_dish_jsn.json'

dataset = IngredientsDataset(json_path, dataset_path, transforms)

# split train and test
train_size = int(0.8 * dataset.__len__())
test_size = dataset.__len__() - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#model
# net = models.resnet18(pretrained=True)
net = ResNet(depth=18, clip_flag=False)#,num_classes=num_classes)
# net = torchvision.models.get_model_weights(torchvision.models.resnet18)
net.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet18']), strict =False)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
#net = net.cuda() if device =='cuda:0'  else net
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_classes)
net = net.cuda() if device =='cuda:0'  else net

def predict(outputs, threshold=0.5):
    """
    :param outputs: output of the model
    :return: the predicted labels
    """
    predicted = torch.sigmoid(outputs)
    predicted[predicted >= threshold] = 1
    predicted[predicted < threshold] = 0
    return predicted

#evaluation of model on test set
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
        outputs = model(inputs) #((inputs,labels)) #TODO debug because not the correct way for batch
        preds = predict(outputs, threshold=0.8)
        loss = criterion(outputs, ingredients_vec)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == ingredients_vec.data)
        print('loss', loss)
    epoch_loss = running_loss /  (len(dataloader.dataset) * preds.shape[1]) #preds.nelement()
    epoch_acc = running_corrects.double() /  (len(dataloader.dataset) * preds.shape[1])
    return epoch_loss, epoch_acc

epoch_loss, epoch_acc = evaluate_model(net, test_dataloader, criterion)

n_epochs = 5
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
net.train()
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_, labels_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        #data_.requires_grad_()
        optimizer.zero_grad()

        outputs = net((data_, labels_))
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = predict(outputs, threshold=0.8)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if batch_idx % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()
        for data_t, target_t, labels_ in test_dataloader:
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net((data_t,labels_))
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            #_,pred_t = torch.max(outputs_t, dim=1)
            pred_t = predict(outputs_t, threshold=0.8)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/(len(test_dataloader) * pred_t.shape[1]))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')


        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')
    net.train()