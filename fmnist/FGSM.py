#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:03:47 2019

@author: rangeet
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from keras.datasets import mnist
import numpy as np
import keras
from keras import backend
from keras.models import load_model,model_from_json
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper

from matplotlib import pyplot as plt
import imageio
keras_model = load_model('fmnist.hdf5')


seed = 27

ones = x_train[y_train == 1]
print(ones.shape)
zeros = x_train[y_train == 0]
print(zeros.shape)
two = x_train[y_train == 2]

three = x_train[y_train == 3]

four = x_train[y_train == 4]

five = x_train[y_train == 5]

six = x_train[y_train == 6]

seven = x_train[y_train == 7]

eight = x_train[y_train == 8]

nine = x_train[y_train == 9]
sess =  backend.get_session()
import scipy.misc

## Save ones as Images
ones_reshaped = ones.reshape(6000,28,28)
ones_reshaped_norm = ones_reshaped.astype('float32')
ones_reshaped_norm = ones_reshaped_norm/255
print(ones_reshaped.shape)
ones_labels = y_train[y_train==1]

predone = np.argmax(keras_model.predict(ones_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, ones_labels))
print(accone)
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19]
for i in range(0,6000):
    oneim = ones_reshaped[i].reshape(28,28)
    scipy.misc.imsave('one/original/one_new{0}.jpg'.format(i), oneim)
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(ones_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(ones_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, ones_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print(adv_pred[0:20])
adv_acc =  np.mean(np.equal(adv_pred, ones_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('one/adv/one_new{0}.jpg'.format(i), oneim)

##

## Two
two_reshaped = two.values[:,1:].reshape(6000,28,28)
two_reshaped_norm = two_reshaped.astype('float32')
two_reshaped_norm = two_reshaped_norm/255
print(two_reshaped.shape)
two_labels = two.values[:,0]

predone = np.argmax(keras_model.predict(two_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, two_labels))
print(accone)
indices = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]
for i in range(0,6000):
    oneim = two_reshaped[i].reshape(28,28)
    scipy.misc.imsave('two/original/two_new{0}.jpg'.format(i), oneim)
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(two_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(two_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, two_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print(adv_pred[0:20])
adv_acc =  np.mean(np.equal(adv_pred, two_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('two/adv/two_new_adv{0}.jpg'.format(i), oneim)
## Three
three_reshaped = three.values[:,1:].reshape(6000,28,28)
three_reshaped_norm = three_reshaped.astype('float32')
three_reshaped_norm = three_reshaped_norm/255
print(three_reshaped.shape)
three_labels = three.values[:,0]

predone = np.argmax(keras_model.predict(three_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, three_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(three_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(three_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, three_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print(adv_pred[0:20])
adv_acc =  np.mean(np.equal(adv_pred, three_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))
indices = [0,9,10,12,13,14,18]
for i in range(0,6000):
    oneim = three_reshaped[i].reshape(28,28)
    scipy.misc.imsave('three/original/three_new{0}.jpg'.format(i), oneim)
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('three/adv/three_new_adv{0}.jpg'.format(i), oneim)

##Four
four_reshaped = four.values[:,1:].reshape(6000,28,28)
four_reshaped_norm = four_reshaped.astype('float32')
four_reshaped_norm = four_reshaped_norm/255
print(four_reshaped.shape)
four_labels = four.values[:,0]

predone = np.argmax(keras_model.predict(four_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, four_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(four_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(four_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, four_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x[0:20]), axis = 1)
print(adv_pred)
#adv_acc =  np.mean(np.equal(adv_pred, four_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))
indices = [0,4,6,7,8,9,10,11,12,13,14,17,18,19]
for i in range(0,6000):
    oneim = four_reshaped[i].reshape(28,28)
    scipy.misc.imsave('four/original/four_new{0}.jpg'.format(i), oneim)
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('four/adv/four_new_adv{0}.jpg'.format(i), oneim)

##Five
five_reshaped = five.values[:,1:].reshape(6000,28,28)
five_reshaped_norm = five_reshaped.astype('float32')
five_reshaped_norm = five_reshaped_norm/255
print(five_reshaped.shape)
five_labels = five.values[:,0]

predone = np.argmax(keras_model.predict(five_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, five_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(five_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(five_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, five_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print(adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, five_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))
indices = [0,4,6,7,8,9,10,11,12,13,14,17,18,19]
for i in range(0,6000):
    oneim = five_reshaped[i].reshape(28,28)
    scipy.misc.imsave('five/original/five_new{0}.jpg'.format(i), oneim)
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('five/adv/five_new_adv{0}.jpg'.format(i), oneim)
##Six
six_reshaped = six.values[:,1:].reshape(6000,28,28)
six_reshaped_norm = six_reshaped.astype('float32')
six_reshaped_norm = six_reshaped_norm/255
print(six_reshaped.shape)
six_labels = six.values[:,0]

predone = np.argmax(keras_model.predict(six_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, six_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(six_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(six_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, six_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print(adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, six_labels))

#print("The adversarial validation accuracy is: {}".format(adv_acc))
indices = [0,1,2,5,7,8,9,10,11,12,14,15,16,17,18]
for i in range(0,6000):
    oneim = six_reshaped[i].reshape(28,28)
    scipy.misc.imsave('six/original/six_new{0}.jpg'.format(i), oneim)
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('six/adv/six_new_adv{0}.jpg'.format(i), oneim)
##Seven
seven_reshaped = seven.values[:,1:].reshape(6000,28,28)
seven_reshaped_norm = seven_reshaped.astype('float32')
seven_reshaped_norm = seven_reshaped_norm/255
print(seven_reshaped.shape)
seven_labels = seven.values[:,0]

predone = np.argmax(keras_model.predict(seven_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, seven_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(seven_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(seven_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, seven_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print(adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, seven_labels))

#print("The adversarial validation accuracy is: {}".format(adv_acc))
indices = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
for i in range(0,6000):
    oneim = seven_reshaped[i].reshape(28,28)
    scipy.misc.imsave('seven/original/seven_new{0}.jpg'.format(i), oneim)
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('seven/adv/seven_new_adv{0}.jpg'.format(i), oneim)
##Eight
eight_reshaped = eight.values[:,1:].reshape(6000,28,28)
eight_reshaped_norm = eight_reshaped.astype('float32')
eight_reshaped_norm = eight_reshaped_norm/255
print(eight_reshaped.shape)
eight_labels = eight.values[:,0]

predone = np.argmax(keras_model.predict(eight_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, eight_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(eight_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(eight_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, eight_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print(adv_pred)
#adv_acc =  np.mean(np.equal(adv_pred, eight_labels))

#print("The adversarial validation accuracy is: {}".format(adv_acc))
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]
for i in range(0,6000):
    oneim = eight_reshaped[i].reshape(28,28)
    scipy.misc.imsave('eight/original/eight_new{0}.jpg'.format(i), oneim)
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('eight/adv/eight_new_adv{0}.jpg'.format(i), oneim)
#nine
nine_reshaped = nine.values[:,1:].reshape(6000,28,28)
nine_reshaped_norm = nine_reshaped.astype('float32')
nine_reshaped_norm = nine_reshaped_norm/255
print(nine_reshaped.shape)
nine_labels = nine.values[:,0]

predone = np.argmax(keras_model.predict(nine_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, nine_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(nine_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(nine_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, nine_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print(adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, nine_labels))

#print("The adversarial validation accuracy is: {}".format(adv_acc))
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
for i in range(0,6000):
    oneim = nine_reshaped[i].reshape(28,28)
    scipy.misc.imsave('nine/original/nine_new{0}.jpg'.format(i), oneim)
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('nine/adv/nine_new_adv{0}.jpg'.format(i), oneim)
#zeros
zero_reshaped = zeros.values[:,1:].reshape(6000,28,28)
zero_reshaped_norm = zero_reshaped.astype('float32')
zero_reshaped_norm = zero_reshaped_norm/255
print(zero_reshaped_norm.shape)
zeros_labels = zeros.values[:,0]

predone = np.argmax(keras_model.predict(zero_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, zeros_labels))
print(accone)

wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(zero_reshaped_norm, **fgsm_params)
one_ex =adv_x * 255
one_ad_ex = one_ex.astype("int64")
pred = np.argmax(keras_model.predict(zero_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, zeros_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
print(adv_pred[0:20])
#adv_acc =  np.mean(np.equal(adv_pred, nine_labels))

#print("The adversarial validation accuracy is: {}".format(adv_acc))
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
for i in range(0,6000):
    oneim = nine_reshaped[i].reshape(28,28)
    scipy.misc.imsave('zero/original/zero_new{0}.jpg'.format(i), oneim)
for i in range(0,6000):
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('zero/adv/zero_new_adv{0}.jpg'.format(i), oneim)