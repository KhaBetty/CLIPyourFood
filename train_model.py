import torch
import os
import torch.optim as optim
import numpy as np
from torchvision import transforms
from CLIPyourFood.model.utils import plot_statistics
from CLIPyourFood.model.utils import predict, accuracy, load_data_in_sections, load_model, freeze_original_resnet
from CLIPyourFood.model.ResNet import CRITERION

# choose random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def train(config, dataset_path, json_dict):
    # model
    net = load_model(w_clip=config['clip_addition'], model_path=config['model_checkpoint'],
                     clip_modification=config['clip_modification'])

    optimizer = optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    train_dataloader, val_dataloader, test_dataloader = load_data_in_sections(dataset_path, json_dict,
                                                                              config['transforms'],
                                                                              config['batch_size'])

    n_epochs = config['epochs_num']
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
            data_, target_ = data_.to(config['device']), target_.to(config['device'])
            # data_.requires_grad_()
            optimizer.zero_grad()

            outputs = net((data_, labels_))
            loss = CRITERION(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # check loss
            pred = predict(outputs, threshold=0.8)
            correct += accuracy(torch.sum(pred == target_).item(), config['batch_size'])
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
            for data_t, target_t, labels_ in val_dataloader:
                data_t, target_t = data_t.to(config['device']), target_t.to(config['device'])
                outputs_t = net((data_t, labels_))  # val_net(data_t) #
                loss_t = CRITERION(outputs_t, target_t)
                batch_loss += loss_t.item()
                pred_t = predict(outputs_t, threshold=0.8)
                correct_t += accuracy(torch.sum(pred_t == target_t).item(), config['batch_size'])
            val_acc.append(100 * correct_t / total_step_val)
            val_loss.append(batch_loss / len(val_dataloader))
            network_learned = batch_loss < valid_loss_min
            print(
                f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_step_val):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), config['output_path'] + '/best_network.pt')
                print('Improvement-Detected, save-model')
        net.train()
    train_results = {'accuracy': train_acc, 'loss': train_loss}
    val_results = {'accuracy': val_acc, 'loss': val_loss}
    plot_statistics(train_results, val_results, config['output_path'])


if __name__ == '__main__':
    current_dir = os.path.abspath(os.getcwd())
    #################################################
    # parameters modification in this part

    dataset_path = current_dir + '/Data/food-101/'
    json_dict = {'train': dataset_path + 'meta/ing_with_dish_jsn_train.json',
                 'val': dataset_path + 'meta/ing_with_dish_jsn_val.json',
                 'test': dataset_path + 'meta/ing_with_dish_jsn_test.json'}

    clip_modification = {'clip_image_features': True,
                         'clip_text_features': False,
                         'freeze_original_resnet': False,
                         # for running the second method of connection with image features only
                         'other_connection_method': False}

    config = {
        'batch_size': 32,
        'epochs_num': 5,
        'learning_rate': 1e-2,
        'momentum': 0.9,
        'transforms': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ]),
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'clip_addition': True,
        'clip_modification': clip_modification,  # can be edited above this dict in clip_modification
        'model_checkpoint': None,
        'output_path': ''
    }

    ###############################
    train(config, dataset_path, json_dict)
