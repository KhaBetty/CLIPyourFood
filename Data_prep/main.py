# Prep ing data -  Python script.
import numpy as np
import json5 as json



if __name__ == '__main__':

    #parameters:
    ingredients_file = "C:\DL_local\Annotations\ingredients_simplified.txt"
    food_img_file = "C:\DL_local\Annotations\img_train.txt"
    slash = "\\"


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
        #split into food class and id:
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
        #remove the \n from the end of the line:
        line = line[:-1]
        # split the line on ':'
        value = line.split(',')
        #for food_class in food_dict:
        for id in food_dict[food_class]:
            key = str(food_class) + slash + str(id)
            #print(key) # for debug
            ing_dict[key] = value
        class_i = class_i+1
        if class_i<=101 :
            food_class = food_classes_list[food_classes_list.index(food_class) + 1]


    
    with open("ing_jsn.json", "w") as outfile:
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

    np.savetxt("ing_vector_one_in_line.txt",
               ing_vector_wo_dup,
               delimiter=", ",
               fmt='% s')

    print("done!")
