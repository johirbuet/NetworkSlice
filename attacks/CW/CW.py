#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:35:52 2019

@author: rangeet
"""

from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
import keras
from keras import backend
from keras.models import load_model
import tensorflow as tf

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper

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
#Zero
zeros_reshaped = zeros.values[:,1:].reshape(4132,28,28,1)
zeros_reshaped_norm = zeros_reshaped.astype('float32')
zeros_reshaped_norm = zeros_reshaped_norm/255
print(zeros_reshaped.shape)
zeros_labels = zeros.values[:,0]

predone = np.argmax(keras_model.predict(zeros_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, zeros_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 2,
               'initial_const': 10}
adv_x = jsma.generate_np(zeros_reshaped_norm, **jsma_params)
zeros_ex =adv_x * 255
zeros_ad_ex = zeros_ex.astype("int64")
pred = np.argmax(keras_model.predict(zeros_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, zeros_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Zero pred is ",adv_pred[0:20])
adv_acc =  np.mean(np.equal(adv_pred, zeros_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))


##One
one_reshaped = ones.values[:,1:].reshape(4684,28,28,1)
one_reshaped_norm = one_reshaped.astype('float32')
one_reshaped_norm = one_reshaped_norm/255
print(one_reshaped.shape)
one_labels = ones.values[:,0]

predone = np.argmax(keras_model.predict(one_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, one_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 2,
               'initial_const': 10}
adv_x = jsma.generate_np(one_reshaped_norm, **jsma_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(one_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, one_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("One pred is ",adv_pred[0:20])
adv_acc =  np.mean(np.equal(adv_pred, one_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))

## Two
two_reshaped = two.values[:,1:].reshape(4177,28,28,1)
two_reshaped_norm = two_reshaped.astype('float32')
two_reshaped_norm = two_reshaped_norm/255
print(two_reshaped.shape)
two_reshaped = two_reshaped[0]
print(two_reshaped.shape)
two_labels = two.values[:,0]
predone = np.argmax(keras_model.predict(two_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, two_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 1,
               'initial_const': 10}
adv_x = jsma.generate_np(two_reshaped_norm, **jsma_params)
two_ex =adv_x * 255
two_ad_ex = two_ex.astype("int64")
pred = np.argmax(keras_model.predict(two_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, two_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Two pred is ",adv_pred[0:20])
adv_acc =  np.mean(np.equal(adv_pred, two_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))

## Three
three_reshaped = three.values[:,1:].reshape(4351,28,28,1)
three_reshaped_norm = three_reshaped.astype('float32')
three_reshaped_norm = three_reshaped_norm/255
print(three_reshaped.shape)
three_labels = three.values[:,0]

predone = np.argmax(keras_model.predict(three_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, three_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 1,
               'initial_const': 10}
adv_x = jsma.generate_np(three_reshaped_norm, **jsma_params)
three_ex =adv_x * 255
three_ad_ex = three_ex.astype("int64")
pred = np.argmax(keras_model.predict(three_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, three_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Three pred is ",adv_pred[0:20])
adv_acc =  np.mean(np.equal(adv_pred, three_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))


##Four
four_reshaped = four.values[:,1:].reshape(4072,28,28,1)
four_reshaped_norm = four_reshaped.astype('float32')
four_reshaped_norm = four_reshaped_norm/255
print(four_reshaped.shape)
four_labels = four.values[:,0]

predone = np.argmax(keras_model.predict(four_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, four_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 2,
               'initial_const': 10}
adv_x = jsma.generate_np(four_reshaped_norm, **jsma_params)
four_ex =adv_x * 255
four_ad_ex = four_ex.astype("int64")
pred = np.argmax(keras_model.predict(four_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, four_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Four pred is ",adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, four_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))


##Five
five_reshaped = five.values[:,1:].reshape(3795,28,28,1)
five_reshaped_norm = five_reshaped.astype('float32')
five_reshaped_norm = five_reshaped_norm/255
print(five_reshaped.shape)
five_labels = five.values[:,0]

predone = np.argmax(keras_model.predict(five_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, five_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 1,
               'initial_const': 10}
adv_x = jsma.generate_np(five_reshaped_norm, **jsma_params)
five_ex =adv_x * 255
five_ad_ex = five_ex.astype("int64")
pred = np.argmax(keras_model.predict(five_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, five_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Five pred is ",adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, five_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))

##Six
six_reshaped = six.values[:,1:].reshape(4137,28,28,1)
six_reshaped_norm = six_reshaped.astype('float32')
six_reshaped_norm = six_reshaped_norm/255
print(six_reshaped.shape)
six_labels = six.values[:,0]

predone = np.argmax(keras_model.predict(six_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, six_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 1,
               'initial_const': 10}
adv_x = jsma.generate_np(six_reshaped_norm, **jsma_params)
six_ex =adv_x * 255
six_ad_ex =six_ex.astype("int64")
pred = np.argmax(keras_model.predict(six_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, six_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Six pred is ",adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, six_labels))

#print("The adversarial validation accuracy is: {}".format(adv_acc))

##Seven
seven_reshaped = seven.values[:,1:].reshape(4401,28,28,1)
seven_reshaped_norm = seven_reshaped.astype('float32')
seven_reshaped_norm = seven_reshaped_norm/255
print(seven_reshaped.shape)
seven_labels = seven.values[:,0]

predone = np.argmax(keras_model.predict(seven_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, seven_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 1,
               'initial_const': 10}
adv_x = jsma.generate_np(seven_reshaped_norm, **jsma_params)
seven_ex =adv_x * 255
seven_ad_ex = seven_ex.astype("int64")
pred = np.argmax(keras_model.predict(seven_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, seven_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Seven pred is ",adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, seven_labels))

#print("The adversarial validation accuracy is: {}".format(adv_acc))

##Eight
eight_reshaped = eight.values[:,1:].reshape(4063,28,28,1)
eight_reshaped_norm = eight_reshaped.astype('float32')
eight_reshaped_norm = eight_reshaped_norm/255
print(eight_reshaped.shape)
eight_labels = eight.values[:,0]

predone = np.argmax(keras_model.predict(eight_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, eight_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 1,
               'initial_const': 10}
adv_x = jsma.generate_np(eight_reshaped_norm, **jsma_params)
eight_ex =adv_x * 255
eight_ad_ex = eight_ex.astype("int64")
pred = np.argmax(keras_model.predict(eight_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, eight_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Eight pred is ",adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, eight_labels))

#print("The adversarial validation accuracy is: {}".format(adv_acc))

#nine
nine_reshaped = nine.values[:,1:].reshape(4188,28,28,1)
nine_reshaped_norm = nine_reshaped.astype('float32')
nine_reshaped_norm = nine_reshaped_norm/255
print(nine_reshaped.shape)
nine_labels = nine.values[:,0]

predone = np.argmax(keras_model.predict(nine_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, nine_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
jsma = CarliniWagnerL2(wrap, sess=sess)
jsma_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 2,
               'initial_const': 10}
adv_x = jsma.generate_np(nine_reshaped_norm, **jsma_params)
nine_ex =adv_x * 255
nine_ad_ex = nine_ex.astype("int64")
pred = np.argmax(keras_model.predict(nine_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, nine_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print("Nine pred is ",adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, nine_labels))


#Zero
indices = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]
for i in range(0,20):
    zeroim = zeros_reshaped[i].reshape(28,28)
    scipy.misc.imsave('zero/original/zeros_new{0}_CW.jpg'.format(i), zeroim)
for i in range(0,20):
    zeroim = zeros_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('zero/adv/zeros_new_adv{0}_CW.jpg'.format(i), zeroim)
#one
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
for i in range(0,20):
    oneim = one_reshaped[i].reshape(28,28)
    scipy.misc.imsave('one/original/one_new{0}.jpg'.format(i), oneim)
for i in indices:
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('one/adv/one_new_adv{0}.jpg'.format(i), oneim)
#Two
indices = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]
for i in range(0,20):
    twoim = two_reshaped[i].reshape(28,28)
    scipy.misc.imsave('two/original/two_new{0}.jpg'.format(i), twoim)
for i in indices:
    twoim = two_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('two/adv/two_new_adv{0}.jpg'.format(i), twoim)
#Three
indices = [0,9,10,12,13,14,18]
for i in range(0,20):
    threeim = three_reshaped[i].reshape(28,28)
    scipy.misc.imsave('three/original/three_new{0}.jpg'.format(i), threeim)
for i in range(0,20):
    threeim = three_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('three/adv/three_new_adv{0}.jpg'.format(i), threeim)
#Four
indices = [0,4,6,7,8,9,10,11,12,13,14,17,18,19]
for i in range(0,20):
    fourim = four_reshaped[i].reshape(28,28)
    scipy.misc.imsave('four/original/four_new{0}.jpg'.format(i), fourim)
for i in range(0,20):
    fourim = four_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('four/adv/four_new_adv{0}.jpg'.format(i), fourim)
#Five
indices = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]
for i in indices:
    fiveim = five_reshaped[i].reshape(28,28)
    scipy.misc.imsave('five/original/five_new{0}.jpg'.format(i), fiveim)
for i in indices:
    fiveim = five_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('five/adv/five_new_adv{0}.jpg'.format(i), fiveim)
#Six
indices = [0,1,2,4,5,7,8,9,10,11,12,14,15,16,17,18]
for i in range(0,20):
    sixim = six_reshaped[i].reshape(28,28)
    scipy.misc.imsave('six/original/six_new{0}.jpg'.format(i), sixim)
for i in range(0,20):
    sixim = six_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('six/adv/six_new_adv{0}.jpg'.format(i), sixim)
#Seven
indices = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
for i in range(0,20):
    sevenim = seven_reshaped[i].reshape(28,28)
    scipy.misc.imsave('seven/original/seven_new{0}.jpg'.format(i), sevenim)
for i in range(0,20):
    sevenim = seven_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('seven/adv/seven_new_adv{0}.jpg'.format(i), sevenim)
#Eight
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]
for i in range(0,20):
    eightim = eight_reshaped[i].reshape(28,28)
    scipy.misc.imsave('eight/original/eight_new{0}.jpg'.format(i), eightim)
for i in range(0,20):
    eightim = eight_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('eight/adv/eight_new_adv{0}.jpg'.format(i), eightim)
#Nine
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]
for i in range(0,20):
    nineim = nine_reshaped[i].reshape(28,28)
    scipy.misc.imsave('nine/original/nine_new{0}.jpg'.format(i), nineim)
for i in range(0,20):
    nineim = nine_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('nine/adv/nine_new_adv{0}.jpg'.format(i), nineim)