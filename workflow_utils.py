#!/usr/bin/env python3
# coding: utf-8

from Pegasus.api import *
import os
from sklearn.model_selection import train_test_split
import argparse
import sys


def create_data_split(input_images):
    cat = []
    dog = []
    for i in range(len(input_images)):
        if 'resized_Cat' in input_images[i]:
            cat.append(input_images[i])
        else:
            dog.append(input_images[i]) 

    train_cat, test_cat = train_test_split(cat, test_size=0.2, random_state=42, shuffle=True)
    train_dog, test_dog = train_test_split(dog, test_size=0.2, random_state=42, shuffle=True)

    training_cat, val_cat = train_test_split(train_cat, test_size=0.1, random_state=42, shuffle=True)
    training_dog, val_dog = train_test_split(train_dog, test_size=0.1, random_state=42, shuffle=True)

    training_data = training_cat + training_dog
    testing_data = test_cat + test_dog
    val_data = val_cat + val_dog
    
    train_data = []
    for f in training_data:
        train_data.append(File(f))
    test_data = []
    for f in testing_data:
        test_data.append(File(f))
    validation_data = []
    for f in val_data:
        validation_data.append(File(f))
        
    return train_data, test_data, validation_data

def parallelize_jobs(WORKERS,job, data, job_type):
    job_list = []
    for i in range(WORKERS):
        job_list.append(Job(job))

    for i in range(len(data)):
        curr = i % len(job_list)
        job_list[curr].add_inputs(data[i])
        if job_type == 'Augment':
            output_files = ["aug_{}_{}".format(os.path.splitext(str(data[i]).split('_')[1])[0], str(k) + '.jpg') for k 
                            in range(2)]
            for f in output_files:
                job_list[curr].add_outputs(File(f))
        
        elif job_type == 'Resize':
            output_file = File("resized_{}".format(data[i].lfn))
            job_list[curr].add_outputs(output_file)
            
    return job_list

