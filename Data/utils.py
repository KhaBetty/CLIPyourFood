import numpy as np
import json5 as json

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


# Prep ing data -  Python script.


def create_ing_dict(ingredients_file, food_img_file, slash):
	'''
	Create file text of ingredients representation
	'''
	# open the text file of the ingredients
	with open(ingredients_file, 'r') as f_ing:
		# read the ingredients text file into a list of lines
		ing_lines = f_ing.readlines()

	# open the text file of the ingredients
	with open(food_img_file, 'r') as f_food:
		# read the ingredients text file into a list of lines
		food_lines = f_food.readlines()

	# create an food dictionary
	food_dict = {}

	for img in food_lines:
		# remove the \n from the end of the line:
		img = img[:-1]
		img_file_name = img.split("/")
		# split into food class and id:
		food_class, id = img_file_name
		food_dict.setdefault(food_class, [])
		food_dict[food_class].append(id)

	# create an empty dictionary for the ingredients
	ing_dict = {}

	# loop through the lines in the text file
	food_class = 'apple_pie'
	food_classes_list = list(food_dict)
	class_i = 1
	for line in ing_lines:
		# remove the \n from the end of the line:
		line = line[:-1]
		# split the line on ':'
		value = line.split(',')
		# for food_class in food_dict:
		for id in food_dict[food_class]:
			key = str(food_class) + slash + str(id)
			# print(key) # for debug
			ing_dict[key] = value
		class_i = class_i + 1
		if class_i <= 101:
			food_class = food_classes_list[food_classes_list.index(food_class) + 1]

	with open("../food101/train/food-101/images/ing_jsn.json", "w") as outfile:
		json.dump(ing_dict, outfile)

	ing_vector = []
	for line in ing_lines:
		# remove the \n from the end of the line:
		line = line[:-1]
		# split the line on ':'
		dish_ing = line.split(',')
		for single_ing in dish_ing:
			ing_vector.append(single_ing)

	ing_vector_wo_dup = list(dict.fromkeys(ing_vector))

	np.savetxt("ing_vector.csv",
	           ing_vector_wo_dup,
	           delimiter=", ",
	           fmt='% s')

	with open("ing_vector.txt", mode="w") as file:
		file.write(str(ing_vector_wo_dup))

	np.savetxt("ingredients_dict.txt",
	           ing_vector_wo_dup,
	           delimiter=", ",
	           fmt='% s')

	print("done!")


