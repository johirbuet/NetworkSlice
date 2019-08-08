#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:11:12 2019

@author: rangeet
"""
# All the imports
import numpy as np
import skimage as ski
print(np.__file__)
print(np.__version__)
print(ski.__version__)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf
from skimage.transform import resize
from netviz import NetViz
from mnistutil import MNISTUitl
mn = MNISTUitl()
viz = NetViz()
sx = 28
sy = 28
X, Y, x, y = mn.getdata2(0,0,sx,sy)
X.shape
nm1 , xt1, yt1 = mn.train2(X, Y, x,y,sx,sy,10,50)
def vispredict(nm, x, y,img_rows = 28, img_cols = 28, ss = 0.2):
        #graph = NGraph()
        w1,b1 = nm.layers[1].get_weights()
        w2,b2 = nm.layers[2].get_weights()
        W1 = np.vstack([w1])
        #print(W1.shape)
        X = x.reshape(img_rows*img_cols,)
        X1 = np.dot(X,W1)
        X1 = np.add(X1, b1)
        X1[X1<0]=0
        W2 = np.vstack([w2])
        X2 = np.dot(X1,W2)
        X2 = np.add(X2,b2)
        X2 = softmax(X2)
        return X1
def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
from keras.models import model_from_json, load_model

model_json = nm1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

nm1.save_weights("model.hdf5")
print("Saved model to disk")
nm1.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.summary()
Zero= xt1[Y==0]
One= xt1[Y==1]
Two= xt1[Y==2]
Three= xt1[Y==3]
Four= xt1[Y==4]
Five= xt1[Y==5]
Six= xt1[Y==6]
Seven= xt1[Y==7]
Eight= xt1[Y==8]
Nine= xt1[Y==9]
X1_Zero=[]
X1_One=[]
X1_Two=[]
X1_Three=[]
X1_Four=[]
X1_Five=[]
X1_Six=[]
X1_Seven=[]
X1_Eight=[]
X1_Nine=[]
for i in range(0, len(Zero)):
    x1 = Zero[i]
    y1 = 0
    X1_Zero.append(vispredict(nm1,x1,y1,28,28))
X1_Zero=np.array(X1_Zero)
for i in range(0, len(One)):
    x1 = One[i]
    y1 = 1
    X1_One.append(vispredict(nm1,x1,y1,28,28))
X1_One=np.array(X1_One)
for i in range(0, len(Two)):
    x1 = Two[i]
    y1 = 2
    X1_Two.append(vispredict(nm1,x1,y1,28,28))
X1_Two=np.array(X1_Two)
for i in range(0, len(Three)):
    x1 = Three[i]
    y1 = 3
    X1_Three.append(vispredict(nm1,x1,y1,28,28))
X1_Three=np.array(X1_Three)
for i in range(0, len(Four)):
    x1 = Four[i]
    y1 = 4
    X1_Four.append(vispredict(nm1,x1,y1,28,28))
X1_Four=np.array(X1_Four)
for i in range(0, len(Five)):
    x1 = Five[i]
    y1 = 5
    X1_Five.append(vispredict(nm1,x1,y1,28,28))
X1_Five=np.array(X1_Five)
for i in range(0, len(Six)):
    x1 = Six[i]
    y1 = 6
    X1_Six.append(vispredict(nm1,x1,y1,28,28))
X1_Six=np.array(X1_Six)
for i in range(0, len(Seven)):
    x1 = Seven[i]
    y1 = 7
    X1_Seven.append(vispredict(nm1,x1,y1,28,28))
X1_Seven=np.array(X1_Seven)
for i in range(0, len(Eight)):
    x1 = Eight[i]
    y1 = 8
    X1_Eight.append(vispredict(nm1,x1,y1,28,28))
X1_Eight=np.array(X1_Eight)
for i in range(0, len(Nine)):
    x1 = Nine[i]
    y1 = 9
    X1_Nine.append(vispredict(nm1,x1,y1,28,28))
X1_Nine=np.array(X1_Nine)
arg_Zero=np.where(X1_Zero>0)
arg_One=np.where(X1_One>0)
arg_Two=np.where(X1_Two>0)
arg_Three=np.where(X1_Three>0)
arg_Four=np.where(X1_Four>0)
arg_Five=np.where(X1_Five>0)
arg_Six=np.where(X1_Six>0)
arg_Seven=np.where(X1_Seven>0)
arg_Eight=np.where(X1_Eight>0)
arg_Nine=np.where(X1_Nine>0)
node_Zero=np.unique(arg_Zero[1])
node_One=np.unique(arg_One[1])
node_Two=np.unique(arg_Two[1])
node_Three=np.unique(arg_Three[1])
node_Four=np.unique(arg_Four[1])
node_Five=np.unique(arg_Five[1])
node_Six=np.unique(arg_Six[1])
node_Seven=np.unique(arg_Seven[1])
node_Eight=np.unique(arg_Eight[1])
node_Nine=np.unique(arg_Nine[1])
x1 = xt1[ s:e]
y1 = yt1[ s:e]
X1=vispredict(nm1,x1,y1,28,28)




#--------------Check-----------------
s = 1
e = 2
x1 = Zero[0].reshape(1,28,28,1)
y1 = yt1[ s:e]
ResultDigit=[]
ResultPredict=[]
dot, A, g, ab = viz.vispredict(nm1,x1,y1,28,28)
ab.append(np.argmax(nm1.predict(x1)))
ResultDigit.append(ab)
print(nm1.predict(x1),y1)
dot8, A = viz.vispredictwithlabel(nm1,x1,y1,sx,sy)
#ResultPredict.append(nm1.predict(x1),y1)
dot.render('example.gv'.format(i))
print()

#------------------------------------