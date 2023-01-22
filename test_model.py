import torch
from CLIPyourFood.model.ResNet import ResNet, model_urls, NUM_CATRGORIES, CRITERION
from CLIPyourFood.model.utils import predict, accuracy, THRESHOLD, load_model, load_data_in_sections
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score


def precision_recall_f1_scores(predictions, targets):
    precision_model = MulticlassPrecision(num_classes=NUM_CATRGORIES)
    recall_model = MulticlassRecall(num_classes=NUM_CATRGORIES)
    f1_score_model = MulticlassF1Score(num_classes=NUM_CATRGORIES)
    scores_dict = {'precision': precision_model(predictions, targets),
                   'recall': recall_model(predictions, targets),
                   'f1_score': f1_score_model(predictions, targets)}
    return scores_dict


def evaluate_model(model, dataloader, criterion=CRITERION, batch_size=32, clip_flag=False):
    '''
    Evaluate the trained model with different params as accuracy, loss, precision, recall, f1 score.
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    running_loss = 0.0
    running_accuracy = 0
    all_preds = None
    all_tragets = None
    for inputs, ingredients_vec, labels in dataloader:
        inputs = inputs.to(device)
        # convert tuple to list
        labels = list(labels)
        # labels = labels.to(device)
        ingredients_vec = ingredients_vec.to(device)
        if clip_flag:
            outputs = model((inputs, labels))  # TODO debug because not the correct way for batch
        else:
            outputs = model(inputs)
        preds = predict(outputs)
        loss = criterion(outputs, ingredients_vec)
        running_loss += loss.item() * inputs.size(0)
        running_accuracy += accuracy(torch.sum(preds == ingredients_vec.data), batch_size)
        if all_preds:
            all_preds = torch.concat((all_preds, preds))
        else:
            all_preds = preds
        if all_tragets:
            all_tragets = torch.concat((all_tragets, ingredients_vec.data))
        else:
            all_tragets = ingredients_vec.data
    epoch_loss = running_loss / len(dataloader.dataset)  # preds.nelement()
    epoch_acc = running_accuracy.double() / len(dataloader.dataset)
    eval_dict = {'loss': epoch_loss,
                 'accuracy': epoch_acc} + precision_recall_f1_scores(all_preds, all_tragets)
    return eval_dict


if __name__ == '__main__':
    # load model
    model_path = ''
    w_clip = False
    dataset_path = ''
    json_dict = ''
    model = load_model(w_clip=w_clip, model_path=model_path)
    _, _, test_dataloader = load_data_in_sections(dataset_path, json_dict=json_dict)
    eval_dict = evaluate_model(model, test_dataloader)
    for score in list(eval_dict.keys()):
        print(score, ':', eval_dict[score])
