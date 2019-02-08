'''
Created on Feb 8, 2019

@author: mislam
'''

from keras.datasets import mnist

from skimage.transform import resize

import numpy as np

from keras import backend as K

import keras

import tensorflow as tf




class MNISTUitl:
    def __init__(self):
        self.name = None
        
    def getdata(self,a,b,img_rows = 28, img_cols = 28):
    # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_zo = []
        y_zo = []
        for i in range(len(y_train)):
            if y_train[i] == a or y_train[i] == b:
                A = resize(x_train[i], (img_rows,  img_cols),mode='constant')
                Ay = y_train[i]#resize(y_train[i], (img_rows, img_cols))
                x_zo.append(A)
                y_zo.append(Ay)
        xt_zo = []
        yt_zo = []
    
        for i in range(len(y_test)):
            if y_test[i] == a or y_test[i] == b:
                A = resize(x_test[i], (img_rows,  img_cols),mode='constant')
                Ay = y_test[i]#resize(y_train[i], (img_rows, img_cols))
                xt_zo.append(A)
                yt_zo.append(Ay)
        x_zo = np.array(x_zo)
        y_zo = np.array(y_zo)
        xt_zo = np.array(xt_zo)
        yt_zo = np.array(yt_zo)
        return x_zo, y_zo, xt_zo, yt_zo
    def train(self,x_zo,y_zo,xt_zo,yt_zo,img_rows = 28, img_cols = 28,numclass = 2):
        if K.image_data_format() == 'channels_first':
            x_zo = x_zo.reshape(x_zo.shape[0], 1, img_rows, img_cols)
            xt_zo = xt_zo.reshape(xt_zo.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_zo = x_zo.reshape(x_zo.shape[0], img_rows, img_cols, 1)
            xt_zo = xt_zo.reshape(xt_zo.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
    
        x_train = x_zo.astype('float32')
        x_test = xt_zo.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_zo.shape,x_train.shape[0], 'train samples', y_zo.shape)
        print(x_test.shape[0], 'test samples')
    
        y_train = y_zo#keras.utils.to_categorical(y_zo, numclass )
        y_test =  yt_zo#keras.utils.to_categorical(yt_zo, numclass)
    
        print(y_zo.shape,y_train.shape)
        nm = keras.Sequential([
            keras.layers.Flatten(input_shape=(img_rows, img_cols,1), name = "Input"),
            keras.layers.Dense(7, activation=tf.nn.relu ,name = "H"),
            keras.layers.Dense(numclass, activation=tf.nn.softmax, name = "output")
        ])
    
        nm.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    
        nm.fit(x_train, y_train, epochs=10)
        return nm, x_test, y_test