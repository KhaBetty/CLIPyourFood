# This is a sample Python script.
import numpy as np


def lables2vec(lables, ing_vec_file_path):
	# recreating the ing label vec
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
	# recreating the ing label vec
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
	ing_vec_file_path = "C:\DL_local\Data_prep\ing_vector_one_in_line.txt"
	lable_list = ["salt", "apple", "flour"]
	multi_one_hot_vec = lables2vec(lable_list, ing_vec_file_path)
	print(multi_one_hot_vec)
	recover_labels = vec2lables(multi_one_hot_vec, ing_vec_file_path)
	print(recover_labels)
	print("done!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
