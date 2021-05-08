#!/usr/bin/env python3
# coding: utf-8

#Predicting the model performance on test dataset
import pickle
import signal
from os import listdir
from numpy import asarray,save
from keras.preprocessing.image import load_img,img_to_array
from keras.models import model_from_json
import numpy as np
import pandas as pd
import h5py
import sys
import joblib
import argparse
from keras import layers, models, optimizers
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from glob import glob

def get_test_data(input_files):
    test_photos, test_labels = list(), list()
    for file in input_files:
        if 'Cat' in file:
            output = 1.0
        else:
            output = 0.0
        photo = load_img(file)
        photo = img_to_array(photo)
        test_photos.append(photo)
        test_labels.append(output)
    test_photos = asarray(test_photos)
    test_labels = asarray(test_labels)
    return test_photos,test_labels

test_data = glob('resized_*.jpg')
    
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

test_photos, test_labels = get_test_data(test_data)
    
loaded_model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])

y_pred = loaded_model.predict(test_photos)
yhat_classes = (y_pred > 0.5)

accuracy = accuracy_score(test_labels, yhat_classes)
print('Accuracy: %f' % accuracy)

precision = precision_score(test_labels, yhat_classes)
print('Precision: %f' % precision)

recall = recall_score(test_labels, yhat_classes)
print('Recall: %f' % recall)

f1 = f1_score(test_labels, yhat_classes)
print('F1 score: %f' % f1)

matrix = confusion_matrix(test_labels, yhat_classes)
print(matrix)

output_file = 'Result_Metrics.txt'
with open(output_file, 'w') as f:
    f.write("Test Accuracy:" + str(accuracy) + "\n" + "Precision:" + str(precision) + "\n" + "Recall:" + str(recall) + "\n"
           + "F1 score:" + str(f1) + "\n" + "Confusion Matrix:" + str(list(matrix)) )
