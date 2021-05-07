#!/usr/bin/env python3
# coding: utf-8

from glob import glob
from os import listdir
from numpy import asarray, save
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
import sys
import os

#function to replace filename, given the pattern it is replaced by another string.
def replace_filename(input, pattern, replaceWith): 
    return input.replace(pattern, replaceWith) 

def main():
    #change the value of 'n' to any desirable value to increase the size of the dataset 

    n = 1000
    cat_path = glob('/local-work/cat-dog-dataset/PetImages/Cat/*.jpg')[:n] 
    dog_path = glob('/local-work/cat-dog-dataset/PetImages/Dog/*.jpg')[:n]

    all_files = cat_path + dog_path
    cat_files = []
    dog_files = []
    photos = list()

    #Here, we are keeping track of the image labels using file names. 
    for i in range(len(all_files)):
        if 'Cat' in os.path.split(all_files[i])[1]:
            cat_files.append(all_files[i])
            try:
                photo = load_img(all_files[i], target_size=(200, 200))
                photo.save(os.path.join('Resized/' + 'resized_Cat_' + str(i) +'.jpg'), 'JPEG', quality=90)
                photos.append(photo)
            except:
                pass
        else:
            dog_files.append(all_files[i]) 
            try:
                photo = load_img(all_files[i], target_size=(200, 200))
                photo.save(os.path.join('Resized/' + 'resized_Dog_' + str(i)) + '.jpg', 'JPEG', quality=90)
                photos.append(photo)
            except:
                pass

if __name__ == '__main__':
    main()



