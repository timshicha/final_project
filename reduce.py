
import os
from PIL import Image
import numpy as np
import pandas as pd


# Starts with folders of 100x100 images
# Results with two csv files (train and test) with each row
# representing a separate image (30x30)


# returns a list of lists of all object names and whether
# they're a fruit or a vegetable
# ie, [['apple','fruit'],['banana','fruit']]
def get_fruits_vegetables_list():
    # Read textfile with names of all fruits and vegetables
    fruits_vegetables = None
    with open('fruits_vegetables.txt') as reader:
        fruits_vegetables = reader.read().splitlines()

    for i in range(len(fruits_vegetables)):
        fruits_vegetables[i] = fruits_vegetables[i].split(',')
    
    return fruits_vegetables

def reduce():

    # get list of fruits/vegetables
    fruits_vegetables = get_fruits_vegetables_list()
    
    # for both the training and the test sets
    for datatype in ['Test','Training']:
        object_array = np.empty((100000,1876))
        counter = 0

        print(f"Converting {datatype} set...")
        # for each folder
        for i in range(len(fruits_vegetables)):
            object = fruits_vegetables[i]

            # for each image in the folder
            for filename in os.listdir(f"fruits-360/{datatype}/{fruits_vegetables[0][0]}"):
                image = Image.open(f"fruits-360/{datatype}/{fruits_vegetables[0][0]}/{filename}")
                resized_image = image.reduce((4,4))

                # convert image to array
                image_as_array = np.insert(np.asarray(resized_image).flatten(), 0, object[1])
                object_array[counter] = image_as_array
                counter += 1
                #print(counter)

            print(f"{i + 1}/{len(fruits_vegetables)}")
        # save file
        object_array.resize((counter, 1876))
        dataframe = pd.DataFrame(object_array)
        dataframe.to_csv(f"{datatype}_set.csv", index=False)

reduce()