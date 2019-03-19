# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
import keras
from keras import backend
from keras.models import load_model
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper

from matplotlib import pyplot as plt
import imageio

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
zeros_reshaped = zeros.values[:,1:].reshape(4132,28,28,1)
zeros_reshaped_norm = zeros_reshaped.astype('float32')
zeros_reshaped_norm = zeros_reshaped_norm/255
print(zeros_reshaped.shape)
zero_labels = zeros.values[:,0]

predzero = np.argmax(keras_model.predict(zeros_reshaped_norm), axis = 1)
acczero = np.mean(np.equal(predzero, zero_labels))
print(acczero)

print(type(zeros_reshaped[0][0][0][0]))
import png
png.from_array(zeros_reshaped[0], 'L').save("zeroimg.png")



import scipy.misc
zeroim = zeros_reshaped[0].reshape(28,28)
scipy.misc.imsave('zeroim.jpg', zeroim)


indices = [2,6,7,8,9,10,11,12,13,14,15,16,17]
print(len(indices))
## Save zeros as image
for i in indices:
    zeroim = zeros_reshaped[i].reshape(28,28)
    scipy.misc.imsave('zero_new{0}.jpg'.format(i), zeroim)

##

## Create adversarial examples of zeros
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(zeros_reshaped_norm, **fgsm_params)
zer_ex =adv_x * 255
zer_ad_ex = zer_ex.astype("int64")

##
    
## Save Adversarial zeros
for i in indices:
    zeroim = zer_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('zero_new_adv{0}.jpg'.format(i), zeroim)
##    

## Save ones as Images
ones_reshaped = ones.values[:,1:].reshape(4684,28,28,1)
ones_reshaped_norm = ones_reshaped.astype('float32')
ones_reshaped_norm = ones_reshaped_norm/255
print(ones_reshaped.shape)
ones_labels = ones.values[:,0]

predone = np.argmax(keras_model.predict(ones_reshaped_norm), axis = 1)
accone = np.mean(np.equal(predone, ones_labels))
print(accone)
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19]
for i in indices:
    oneim = ones_reshaped[i].reshape(28,28)
    scipy.misc.imsave('one_new{0}.jpg'.format(i), oneim)
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
adv_pred = np.argmax(keras_model.predict(adv_x[0:20]), axis = 1)
print(adv_pred)
adv_acc =  np.mean(np.equal(adv_pred, ones_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))
for i in indices:
    oneim = one_ad_ex[i].reshape(28,28)
    scipy.misc.imsave('one_new_adv{0}.jpg'.format(i), oneim)
##



pred = np.argmax(keras_model.predict(zeros_reshaped_norm), axis = 1)
acc =  np.mean(np.equal(pred, zero_labels))

print("The normal validation accuracy is: {}".format(acc))
adv_pred = np.argmax(keras_model.predict(adv_x[0:20]), axis = 1)
print(adv_pred)
adv_acc =  np.mean(np.equal(adv_pred, zero_labels))

print("The adversarial validation accuracy is: {}".format(adv_acc))


wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(x_validation, **fgsm_params)

adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
adv_acc =  np.mean(np.equal(adv_pred, y_validation))

print("The adversarial validation accuracy is: {}".format(adv_acc))


def stitch_images(images, y_img_count, x_img_count, margin = 2):
    
    # Dimensions of the images
    img_width = images[0].shape[0]
    img_height = images[0].shape[1]
    
    width = y_img_count * img_width + (y_img_count - 1) * margin
    height = x_img_count * img_height + (x_img_count - 1) * margin
    stitched_images = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(y_img_count):
        for j in range(x_img_count):
            img = images[i * x_img_count + j]
            if len(img.shape) == 2:
                img = np.dstack([img] * 3)
            stitched_images[(img_width + margin) * i: (img_width + margin) * i + img_width,
                            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    return stitched_images

x_sample = x_validation[0].reshape(28, 28)
adv_x_sample = adv_x[0].reshape(28, 28)

adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)

plt.imshow(adv_comparison)
plt.show()