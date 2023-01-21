from __future__ import print_function, division
import os
import torch
import pandas as pd
import json
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from CLIPyourFood.Data.utils import lables2vec
from PIL import Image

import warnings

warnings.filterwarnings("ignore")

plt.ion()


class IngredientsDataset(Dataset):
	"""Ingredients dataset."""

	def __init__(self, json_file, root_dir, transform=None, img_ext='.jpg'):
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
		self.img_ext = img_ext

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = self.images[idx]
		img_path = os.path.join(self.root_dir, img_name + self.img_ext)
		#image = io.imread(img_path)
		image = Image.open(img_path)
		image = image.convert('RGB')
		# TODO change after full keys dict to tensor of labels
		ingredients_names = self.ingredients_frame[img_name][0]
		dish_name = self.ingredients_frame[img_name][1]
		ingredients_vec = lables2vec(ingredients_names, os.path.join(self.root_dir,
		                                                             'ing_vector_one_in_line.txt'))

		if self.transform:
			image = self.transform(image)

		return image, ingredients_vec, dish_name


# dataset_path = '../food101/train/food-101/images'
# json_path = dataset_path + '/ing_jsn.json'
#
# dataset = IngredientsDataset(json_path, dataset_path, transforms)
#
# # split train and test
# train_size = int(0.8 * dataset.__len__())
# test_size = dataset.__len__() - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#
# images, ing_list = next(iter(train_dataloader))

#convert tuple to list
# ing_list = list(ing_list)