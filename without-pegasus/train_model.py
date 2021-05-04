#!/usr/bin/env python3
# coding: utf-8
import pickle
import signal
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from keras import layers, models, optimizers
from keras.layers import Input,Dense,BatchNormalization,Flatten,Dropout,GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import keras.backend as K
import traceback
from keras.applications.vgg16 import VGG16
from keras.models import Model,load_model
import pandas as pd
import h5py
import sys
import joblib
import argparse
import keras
from glob import glob
import tensorflow as tf

def parse_args(args):
    parser = argparse.ArgumentParser(description='Cat and Dog image classification using Keras')
    parser.add_argument('-epochs',  metavar='num_epochs', type=int, default = 5, help = "Number of training epochs")
    parser.add_argument('--batch_size',  metavar='batch_size', type=int, default = 16, help = "Batch Size")
    parser.add_argument('-f')
    return parser.parse_args()

#get training, testing and validation data from the saved pickle files.
def get_data(train_data):
    train_photos, train_labels = list(), list()
    tp = list()
    for file in train_data:
        if 'Cat' in file:
            output = 1.0
        else:
            output = 0.0
        photo = load_img(file)
        photo = img_to_array(photo)
        train_photos.append(photo)
        train_labels.append(output)
    train_photos = asarray(train_photos)
    train_labels = asarray(train_labels)
    return train_photos, train_labels

#Definition of the VGG16 model and changing the output layer according to our requirements.
#i.e., 2 output classes
def get_model():
    Study = joblib.load('hpo_results.pkl')
    _dict = Study.best_trial.params
    activation_optuna = _dict['activation']
    
    conv_base = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(200, 200, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation=activation_optuna))
    model.add(layers.Dense(1, activation='sigmoid'))

    conv_base.trainable = False

    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])  
    return model


def main():
    train_data = glob('augment/*.jpg')

    with open('validation.pkl', 'rb') as f:
        val_data = pickle.load(f)

    train_photos,train_labels = get_data(train_data)
    val_photos, val_labels = get_data(val_data)
    model = get_model()
    
    #checkpoint file that saves the weights after each epoch - weights are overwritten to the same file
    checkpoint_file = 'checkpoint_file.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, mode='auto',save_weights_only = True, period=1)
    
    train_from_beginning = False
    try:
        #Since our hdf5 file contains additional data = epochs, skip_mismatch is used to avoid that column
        model.load_weights("checkpoint_file.hdf5",skip_mismatch=True)
        with h5py.File('checkpoint_file.hdf5', "r+") as file:
            data = file.get('epochs')[...].tolist()
            
        #loading the number of epochs already performed to resume training from that epoch
        initial_epoch = data
        model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])
        for i in range(initial_epoch,EPOCHS):
            model.fit(x=train_photos, y=train_labels,validation_data=(val_photos, val_labels),batch_size=32, epochs = 1,
            verbose=1,callbacks = [checkpoint])
            checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, mode='auto',
                                         save_weights_only = True, period=1)
            
            #saving the number of finished epochs to the same hdf5 file
            with h5py.File('checkpoint_file.hdf5', "a") as file:
                file['epochs'] = i
    except OSError:
        train_from_beginning = True

    if train_from_beginning:
        model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])
        for i in range(EPOCHS):
            model.fit(x=train_photos, y=train_labels,validation_data=(val_photos, val_labels),batch_size=32, epochs = 1,
            verbose=1,callbacks = [checkpoint])
            checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, mode='auto',save_weights_only = True, period=1)
            #saving the number of finished epochs to the same hdf5 file
            with h5py.File('checkpoint_file.hdf5', "a") as file:
                file['epochs']=i

    model.save('model.h5')
    return 0
    
if __name__ == '__main__':
    global EPOCHS
    global BATCH_SIZE
    args = parse_args(sys.argv[1:])
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    main()