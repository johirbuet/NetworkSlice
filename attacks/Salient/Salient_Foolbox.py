#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:20:41 2019

@author: rangeet
"""


from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
import keras
from keras import backend
from keras.models import load_model
import tensorflow as tf
import foolbox
from foolbox.attacks import SaliencyMapAttack
from foolbox.criteria import Misclassification

from matplotlib import pyplot as plt
import scipy.misc
# Set the matplotlib figure size
plt.rc('figure', figsize = (12.0, 12.0))

# Set the learning phase to false, the model is pre-trained.
backend.set_learning_phase(False)
keras_model = load_model('Jan-13-2018.hdf5')




seed = 27

raw_data = pd.read_csv("train.csv")
print(raw_data.head())
print(raw_data.shape, 28*28)
ones = raw_data[raw_data["label"] == 1]
print(ones.shape)
zeros = raw_data[raw_data["label"] == 0]
print(zeros.shape)
two = raw_data[raw_data["label"] == 2]

three = raw_data[raw_data["label"] == 3]

four = raw_data[raw_data["label"] == 4]

five = raw_data[raw_data["label"] == 5]

six = raw_data[raw_data["label"] == 6]

seven = raw_data[raw_data["label"] == 7]

eight = raw_data[raw_data["label"] == 8]

nine = raw_data[raw_data["label"] == 9]

train, validate = train_test_split(raw_data, 
                                   test_size=0.1,
                                   random_state = seed, 
                                   stratify = raw_data['label'])

# Split into input (X) and output (Y) variables
x_validation = validate.values[:,1:].reshape(4200,28,28, 1)
y_validation = validate.values[:,0]


tf.set_random_seed(1234)

if not hasattr(backend, "tf"):
    raise RuntimeError("This tutorial requires keras to be configured"
                       " to use the TensorFlow backend.")

if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Retrieve the tensorflow session
sess =  backend.get_session()

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))

# Evaluate the model's accuracy on the validation data used in training
x_validation = x_validation.astype('float32')
x_validation /= 255
#ZERO
zeros_reshaped = zeros.values[:,1:].reshape(4132,28,28,1)
zeros_reshaped_norm = zeros_reshaped.astype('float32')
zeros_reshaped_norm = zeros_reshaped_norm/255
print(zeros_reshaped.shape)
zeros_labels = zeros.values[:,0]

predone = np.argmax(keras_model.predict(zeros_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, zeros_labels))
print(accone)
adv_x=foolbox.attacks.SaliencyMapAttack(zeros_reshaped_norm, criterion=Misclassification())
zeros_ex =adv_x * 255
zeros_ad_ex = zeros_ex.astype("int64")
pred = np.argmax(keras_model.predict(zeros_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, zeros_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Zero pred is ",adv_pred[0:20])
adv_acc =  np.mean(np.equal(adv_pred, zeros_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))
