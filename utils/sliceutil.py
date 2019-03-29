'''
Created on Mar 29, 2019

@author: mislam
'''


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf
from skimage.transform import resize
from graphviz import Graph, render
from collections import defaultdict 
import queue as Q


class Slice:
    def __init__(self):
        pass
    
    
    
    '''
    Return the static weights of the model
    '''
    
    def getweights(self, nm):
        w1,b1 = nm.layers[1].get_weights()
        w2,b2 = nm.layers[2].get_weights()
        W1 = np.vstack([w1])
        W2 = np.vstack([w2])
        
        return [W1, W2, b1, b2]
    
    
