# This is a sample Python script.
import numpy as np
import json5 as json
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def create_jsn(food_img_file,jsn_name):
    # parameters:
    ingredients_file = "C:\DL_local\Annotations\ingredients_simplified.txt"

    slash = "\\"

    # open the text file of the ingredients
    with open(ingredients_file, 'r') as f_ing:
        # read the ingredients text file into a list of lines
        ing_lines = f_ing.readlines()

    # open the text file of the food
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

    # create an empty dictionary for the ingredients and dish
    ing_dish_dict = {}

    # loop through the lines in the text file
    food_class = 'apple_pie'
    food_classes_list = list(food_dict)
    class_i = 1
    for line in ing_lines:
        # remove the \n from the end of the line:
        line = line[:-1]
        # split the line on ':'
        value = line.split(',')
        value_with_dish = [value, food_class]
        # for food_class in food_dict:
        for id in food_dict[food_class]:
            key = str(food_class) + slash + str(id)
            ing_dict[key] = value
            ing_dish_dict[key] = value_with_dish
        class_i = class_i + 1
        if class_i <= 101:
            food_class = food_classes_list[food_classes_list.index(food_class) + 1]

    with open("../food101/train/food-101/images/ing_jsn_" + str(jsn_name) + ".json", "w") as outfile:
        json.dump(ing_dict, outfile)

    with open("../food101/train/food-101/images/ing_with_dish_jsn_"  + str(jsn_name) + ".json", "w") as outfile2:
        json.dump(ing_dish_dict, outfile2)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    food_img_file_train = "C:\DL_local\Annotations\img_train.txt"
    jsn_name_train = "train"
    food_img_file_val = "C:\DL_local\Annotations\img_val.txt"
    jsn_name_val = "val"
    food_img_file_test = "C:\DL_local\Annotations\img_test.txt"
    jsn_name_test = "test"

    create_jsn(food_img_file_train,jsn_name_train)
    create_jsn(food_img_file_val, jsn_name_val)
    create_jsn(food_img_file_test, jsn_name_test)

    print("done")

