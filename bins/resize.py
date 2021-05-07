#!/usr/bin/env python3
# coding: utf-8
from os import listdir
from numpy import asarray
from numpy import save
import numpy as np
import sys
import os
from glob import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


#function to replace filename, given the pattern it is replaced by another string.
def replace_filename(input, pattern, replaceWith): 
    return input.replace(pattern, replaceWith) 

def main():
    input_files = glob('*.jpg')
    photos = list()
    cat_files =[]
    dog_files =[]

    #Here, we are keeping track of the image labels using file names. 
    for i in range(len(input_files)):
        if 'Cat' in input_files[i]:
            cat_files.append(input_files[i])
            photo = load_img(input_files[i], target_size=(200, 200))
            photo.save(replace_filename(input_files[i],'Cat','resized_Cat'), 'JPEG', quality=90)
            photos.append(photo)
        else:
            dog_files.append(input_files[i]) 
            photo = load_img(input_files[i], target_size=(200, 200))
            photo.save(replace_filename(input_files[i],'Dog','resized_Dog'), 'JPEG', quality=90)
            photos.append(photo)
    return 0

if __name__ == '__main__':
    main()