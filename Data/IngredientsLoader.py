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
		self.ingredients_frame = pd.read_json(json_file)
		self.root_dir = root_dir
		self.transform = transform
		self.images = list(self.ingredients_frame.columns)
		self.img_ext = img_ext

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = self.images[idx]
		img_path = os.path.join(self.root_dir, img_name + self.img_ext)
		image = io.imread(img_path)
		# TODO change after full keys dict to tensor of labels
		ingredients = list(self.ingredients_frame[img_name])

		if self.transform:
			image = self.transform(image)

		return image, ingredients

