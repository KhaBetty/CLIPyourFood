import torch
from CLIPyourFood.model.ResNet import CRITERION
from CLIPyourFood.model.utils import predict, accuracy, load_model, load_data_in_sections
import tqdm


def calc_scores(predictions, targets):
    """
    Calculate presicion, recall and f1 score.
    :param predictions: tensor of sample with predictions
    :param targets: tensor of the sample correct labels
    """

    def _opposite_binary_tensor(tensor):
        return -1 * tensor + 1

    tp = torch.sum(torch.mul(predictions, targets)).item()
    fn = torch.sum(torch.mul(_opposite_binary_tensor(predictions), targets)).item()
    tn = torch.sum(torch.mul(_opposite_binary_tensor(predictions), _opposite_binary_tensor(targets))).item()
    fp = torch.sum(torch.mul(predictions, _opposite_binary_tensor(targets))).item()
    if tp + fp == 0:
        presicion = 0
    else:
        presicion = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if recall + presicion == 0:
        f1_score = 0
    else:
        f1_score = 2 * presicion * recall / (recall + presicion)
    return presicion, recall, f1_score


def evaluate_model(model, dataloader, criterion=CRITERION, batch_size=32, clip_flag=False):
    '''
    Evaluate the trained model with different params as accuracy, loss, precision, recall, f1 score.
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    running_loss = 0.0
    running_accuracy = 0
    running_presicion = 0
    running_recall = 0
    running_f1 = 0
    for inputs, ingredients_vec, labels in tqdm.tqdm(dataloader):
        inputs = inputs.to(device)
        ingredients_vec = ingredients_vec.to(device)
        if clip_flag:
            outputs = model((inputs, labels))  # TODO debug because not the correct way for batch
        else:
            outputs = model(inputs)
        preds = predict(outputs)  # , threshold=0.5)
        loss = criterion(outputs, ingredients_vec)
        running_loss += loss.item()
        running_accuracy += accuracy(torch.sum(preds == ingredients_vec.data), batch_size).item()
        presicion, recall, f1_score = calc_scores(preds, ingredients_vec.data)
        running_presicion += presicion
        running_recall += recall
        running_f1 += f1_score

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100 * running_accuracy / len(dataloader.dataset)
    epoch_presicion = running_presicion / len(dataloader.dataset)
    epoch_recall = running_recall / len(dataloader.dataset)
    epoch_f1 = running_f1 / len(dataloader.dataset)
    eval_dict = {'loss': epoch_loss,
                 'accuracy': epoch_acc,
                 'precision': epoch_presicion,
                 'recall': epoch_recall,
                 'f1_score': epoch_f1}

    return eval_dict


if __name__ == '__main__':
    ####################################
    # parameters that can be modified
    model_path = ''
    clip_addition = True
    dataset_path = 'food101/train/food-101'
    json_dict = {'test': 'food101/train/food-101/images/ing_with_dish_jsn_test.json'}
    clip_modification = {'clip_image_features': True,
                         'clip_text_features': False,
                         'freeze_original_resnet': False,
                         'other_connection_method': False}
    ####################################
    # testing the model with the loaded data from json
    #always batch 1 for accurate calculation
    batch_size = 1
    model = load_model(w_clip=clip_addition, model_path=model_path, clip_modification=clip_modification)
    _, _, test_dataloader = load_data_in_sections(dataset_path, json_dict=json_dict, batch_size=batch_size)
    eval_dict = evaluate_model(model, test_dataloader, clip_flag=clip_addition, batch_size=batch_size)
    for score in list(eval_dict.keys()):
        print(score, ':', eval_dict[score])
