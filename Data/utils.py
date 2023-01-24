import matplotlib.pyplot as plt
import numpy as np
import torchvision
import os
import shutil
import glob
import tqdm
from torchvision import transforms
from PIL import Image

TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])


def download_dataset(path=None):
    """
	Download the Dataset food101 and mov the relevant json files to the metadata of it.
	Arguments:
		path: path for downloading the dataset
	"""
    if not path:
        path = os.path.abspath(os.getcwd())
    torchvision.datasets.Food101(path, download=True, transform=TRANSFORMS)
    # move the relevant json files to meta of dataset
    json_files = glob.glob('ingredients_json/*')
    for json_file in tqdm.tqdm(json_files):
        shutil.copy(json_file, path + '/food-101/meta/')


def lables2vec(lables, ing_vec_file_path):
    '''
	Recreating the ingredient labels vector to numerical representation of the vector
	'''
    with open(ing_vec_file_path) as f:
        ing_list = f.read().splitlines()
    N = len(ing_list)

    # define a mapping of ing to integers
    ing_list_to_int = dict((ing, i) for i, ing in enumerate(ing_list))
    # int_to_ing_list = dict((i, ing) for i, ing in enumerate(ing_list))

    # creating the vector
    multi_one_hot_vec = np.zeros(N)

    # integer encode input data
    integer_encoded = [ing_list_to_int[ing] for ing in lables]

    # multi-one hot encode
    for value in integer_encoded:
        multi_one_hot_vec[value] = 1

    return multi_one_hot_vec

def imshow(img_path, ingredients):
    """
    Display the image with it ingredients above as title
    """
    image = Image.open(img_path)
    plt.imshow(image)
    plt.title(str(ingredients))
    plt.axis('off')
    plt.show()
    plt.waitKey(0)

def vec2lables(vec, ing_vec_file_path):
    '''
	Recreating the numerical representation of the vector to ingredient labels vector
	'''
    with open(ing_vec_file_path) as f:
        ing_list = f.read().splitlines()
    N = len(ing_list)

    # define a mapping of ing to integers
    # ing_list_to_int = dict((ing, i) for i, ing in enumerate(ing_list))
    int_to_ing_list = dict((i, ing) for i, ing in enumerate(ing_list))

    integer_encoded = []
    for idx, element in enumerate(vec):
        if element == 1:
            integer_encoded.append(idx)

    lables = []
    for i in integer_encoded:
        lables.append(int_to_ing_list[i])

    return lables

if __name__ == '__main__':
    download_dataset()
