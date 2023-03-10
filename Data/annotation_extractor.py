import json5 as json
import numpy as np


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

if __name__ == '__main__':

    #parameters:
    ingredients_file = '/Annotations/ingredients_simplified.txt' #modify to the path where the annotations downloaded
    food_img_file = "/Annotations/img_train.txt" #modify to the path where the annotations downloaded
    slash = "/"


    # open the text file of the ingredients
    with open(ingredients_file, 'r') as f_ing:
        # read the ingredients text file into a list of lines
        ing_lines = f_ing.readlines()

    # open the text file of the food
    with open(food_img_file, 'r') as f_food:
        # read the ingredients text file into a list of lines
        food_lines = f_food.readlines()

    # create annotation food dictionary
    food_dict = {}

    for img in food_lines:
        # remove the \n from the end of the line:
        img = img[:-1]
        img_file_name = img.split("/")
        #split into food class and id:
        food_class, id = img_file_name
        food_dict.setdefault(food_class, [])
        food_dict[food_class].append(id)

    # create an empty dictionary for the ingredients
    ing_dict = {}

    # create an empty dictionary for the ingredients and dish
    ing_dish_dict = {}


    # loop through the lines in the text file
    food_class = 'apple_pie'
    food_classes_list = list(food_dict)
    class_i = 1
    for line in ing_lines:
        #remove the \n from the end of the line:
        line = line[:-1]
        # split the line on ':'
        value = line.split(',')
        value_with_dish = [value, food_class]
        #for food_class in food_dict:
        for id in food_dict[food_class]:
            key = str(food_class) + slash + str(id)
            ing_dict[key] = value
            ing_dish_dict[key] = value_with_dish
        class_i = class_i+1
        if class_i<=101 :
            food_class = food_classes_list[food_classes_list.index(food_class) + 1]

    with open("food-101/images/ing_jsn.json", "w") as outfile: # modify path to the downloaded dataset
        json.dump(ing_dict, outfile)

    with open("food-101/meta/ing_with_dish_jsn.json", "w") as outfile2: # modify path to the downloaded dataset
        json.dump(ing_dish_dict, outfile2)


