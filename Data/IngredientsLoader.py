from __future__ import print_function, division
import os
import torch
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from CLIPyourFood.Data.utils import lables2vec
from PIL import Image

import warnings

warnings.filterwarnings("ignore")

plt.ion()

TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])


class IngredientsDataset(Dataset):
    """Ingredients dataset."""

    def __init__(self, json_file, root_dir, metadata_path=None, transform=None, img_ext='.jpg'):
        """
		Args:
			json_file (string): Path to the json file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
        with open(json_file) as json_data:
            self.ingredients_frame = json.load(json_data)
        self.root_dir = root_dir
        self.transform = transform
        self.images = list(self.ingredients_frame.keys())
        if metadata_path:
            with open(metadata_path) as meta_data:
                self.images = meta_data.read().splitlines()
        self.img_ext = img_ext

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name + self.img_ext)
        # image = io.imread(img_path)
        image = Image.open(img_path)
        image = image.convert('RGB')
        # TODO change after full keys dict to tensor of labels
        ingredients_names = self.ingredients_frame[img_name][0]
        dish_name = self.ingredients_frame[img_name][1]
        dish_info = (img_path, dish_name)
        ingredients_vec = lables2vec(ingredients_names, os.path.join(self.root_dir,
                                                                     'ing_vector_one_in_line.txt'))

        if self.transform:
            image = self.transform(image)

        return image, ingredients_vec, dish_info
