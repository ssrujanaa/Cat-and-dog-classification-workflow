#!/usr/bin/env python3
# coding: utf-8
from keras.preprocessing.image import ImageDataGenerator,  array_to_img, img_to_array, load_img 
from glob import glob
from numpy import save
import os
import pickle


def augment():
    with open('training.pkl', 'rb') as f:
        resized_images = pickle.load(f)

    cat_images=[]
    dog_images=[]
    q = []
    for i in range(len(resized_images)):
        if 'resized_Cat' in resized_images[i]:
            cat_images.append(resized_images[i])
        else:
            dog_images.append(resized_images[i])

    datagen = ImageDataGenerator( rescale=1./255,
                rotation_range = 40,  
                horizontal_flip = True, 
                brightness_range = (0.5, 1.5)) 

    for image in cat_images:
        num = 1.0
        img = load_img(image)  
        x = img_to_array(img) 
        x = x.reshape((1, ) + x.shape) 
        i = 0
        for batch in datagen.flow(x, batch_size = 1,
                                  save_to_dir = 'augment/',  
                                  save_prefix ='aug_' + os.path.splitext(image.split('_')[1])[0] , save_format ='jpg'): 
            i += 1
            if i > 1: 
                break
    for image in dog_images:
            num = 0.0
            img = load_img(image)  
            x = img_to_array(img) 
            x = x.reshape((1, ) + x.shape)  
            # Generating and saving 5 augmented samples  
            i = 0
            for batch in datagen.flow(x, batch_size = 1,
                                      save_to_dir = 'augment/',  
                                      save_prefix= 'aug_' + os.path.splitext(image.split('_')[1])[0] , save_format ='jpg'): 
                i += 1
                if i > 1: 
                    break
    return cat_images, dog_images,q

def rename_augmented_files():
    try:
        cat_images, dog_images,q = augment()
        aug_cat = glob('aug_Cat*.jpg')
        aug_dog = glob('aug_Dog*.jpg')
        aug_images = aug_cat + aug_dog
        aug_cat.sort()
        for n in range(len(cat_images)):
            k = 0
            for m in range(len(aug_cat)):
                if os.path.splitext(cat_images[n].split('_')[1])[0]  in aug_cat[m]:
                    photo = load_img(aug_cat[m])
                    photo.save('aug_'+ os.path.splitext(cat_images[n].split('_')[1])[0] + '_' + str(k) +'.jpg', 
                               'JPEG', quality=90)
                    q.append('aug_'+ os.path.splitext(cat_images[n].split('_')[1])[0] + '_' +str(k) +'.jpg')
                    k+=1

        aug_dog.sort()
        for n in range(len(dog_images)):
            k = 0
            for m in range(len(aug_dog)):
                if os.path.splitext(dog_images[n].split('_')[1])[0]  in aug_dog[m]:
                    photo = load_img(aug_dog[m])
                    photo.save('aug_'+ os.path.splitext(dog_images[n].split('_')[1])[0] + '_' + str(k) + '.jpg', 
                               'JPEG', quality=90)
                    q.append('aug_'+ os.path.splitext(dog_images[n].split('_')[1])[0] + '_' + str(k) +'.jpg')
                    k+=1 
    except FileNotFoundError:
        print("FileNotFoundError")

def main():
    rename_augmented_files()
    return 0

if __name__ == '__main__':
    main()