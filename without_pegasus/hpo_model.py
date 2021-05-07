#!/usr/bin/env python3
# coding: utf-8
import sys
import shutil
from numpy import asarray
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras import layers, models, optimizers
from keras.layers import Dense,Flatten,Dropout
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.applications.vgg16 import VGG16
from optkeras.optkeras import OptKeras
import optkeras
import pickle
import optuna
import os
import tensorflow as tf
from PIL import Image
import argparse
import joblib
import pandas as pd
from glob import glob
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
optkeras.optkeras.get_trial_default = lambda: optuna.trial.FrozenTrial(
        None, None, None, None, None, None, None, None, None, None, None)

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

def replace_filename(input, pattern, replaceWith): 
    return input.replace(pattern, replaceWith) 
    
def hpo_monitor(study, trial):
    joblib.dump(study,"hpo_checkpoint.pkl")
       
#get training, testing and validation data from the saved pickle files.
def get_data(train_data):
    train_photos, train_labels = list(), list()
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

def objective(trial):
    train_data = glob('augment/*.jpg')

    with open('validation.pkl', 'rb') as f:
        val_data = pickle.load(f)

    train_photos,train_labels = get_data(train_data)
    val_photos, val_labels = get_data(val_data)
                                                 
    conv_base = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(200, 200, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation=trial.suggest_categorical('activation', ['relu', 'linear'])))
    model.add(layers.Dense(1, activation='sigmoid'))

    conv_base.trainable = False

    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])  
    model.fit(x=train_photos, y=train_labels,validation_data=(val_photos, val_labels),batch_size=32, epochs = 3) 
    return 0


def main():
    hpo_checkpoint_file = 'hpo_checkpoint.pkl'
    if not os.path.isfile(hpo_checkpoint_file):
        df = pd.DataFrame(list())
        df.to_csv(hpo_checkpoint_file)

    N_TRIALS = 6
    tune_from_beginning = False
    try:
        ok = joblib.load(hpo_checkpoint_file)
        todo_trials = N_TRIALS - len(ok.trials_dataframe())
        if todo_trials > 0 :
            ok.optimize(objective, n_trials=todo_trials, timeout=600, callbacks=[hpo_monitor])
        else:
            pass
    except KeyError:
        tune_from_beginning = True

    if tune_from_beginning: 
        study_name = "CatsAndDogs" + '_Simple'
        ok = OptKeras(study_name=study_name,monitor='val_acc',direction='maximize')
        ok.optimize(objective, n_trials=N_TRIALS, timeout=600, callbacks=[hpo_monitor])
    output_file = 'hpo_results.pkl'
    shutil.copyfile(hpo_checkpoint_file, output_file)
    
    return 0
        
if __name__ == '__main__':
    main()