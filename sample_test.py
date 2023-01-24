import torch
from CLIPyourFood.model.ResNet import CRITERION
from CLIPyourFood.model.utils import predict, accuracy, load_model, load_data_in_sections,TRANSFORMS
from CLIPyourFood.Data.utils import vec2lables, imshow
from PIL import Image


def predict_sample(model, input_img_path, dish_name='irrelevant'):
    '''
    Get ingredients list for the image inserted with the model.
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    input_img = Image.open(input_img_path)
    input_img = TRANSFORMS(input_img.convert('RGB'))
    input_img = input_img.expand(1, input_img.size(0), input_img.size(1), input_img.size(2))
    input_img = input_img.to(device)
    label = ([input_img_path],dish_name)
    if model.clip_model:
        outputs = model((input_img, label))
    else:
        outputs = model(input_img)
    preds = predict(outputs)
    preds = preds.squeeze(0)
    preds_vec = vec2lables(preds, 'Data/food-101/meta/ingredients_dict.txt')
    return preds_vec


if __name__ == '__main__':
    # parameters that can be modified
    model_path = ''
    clip_addition = True
    clip_modification = {'clip_image_features': True,
                         'clip_text_features': False,
                         'freeze_original_resnet': False,
                         'other_connection_method': False}
    input_img_path = ''
    ####################################
    # testing the model with image
    model = load_model(w_clip=clip_addition, model_path=model_path, clip_modification=clip_modification)
    ing_vec = predict_sample(model, input_img_path)
    imshow(input_img_path, ing_vec)
    print('The ingredients are')
    for ing in ing_vec:
        print(ing)

