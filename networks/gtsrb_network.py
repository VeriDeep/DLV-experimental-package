#!/usr/bin/env python

from __future__ import print_function

import scipy.io as sio
import numpy as np
import struct
from array import array as pyarray
from PIL import Image

from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, UpSampling2D, Deconvolution2D
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils

# for mnist
from keras.datasets import mnist

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


#

from gtsrb import * 


batch_size = 32
nb_epoch = 30
IMG_SIZE = 32
img_rows, img_cols = 32, 32
img_channels = 3
nb_classes = 43
lr = 0.01

def build_model():
    """
    define neural network model
    """
    
    if K.backend() == 'tensorflow': 
        K.set_learning_phase(0)
    
    if K.backend() == 'tensorflow': 
        inputShape = (img_rows,img_cols,img_channels)
    else: 
        inputShape = (img_channels,img_rows,img_cols)
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=inputShape, activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))  

    # let's train the model using SGD + momentum

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
             
    return model, batch_size, nb_epoch, lr
             

             
def evaluation():

    test = pd.read_csv('GT-final_test.csv',sep=';')

    # Load test dataset
    X_test = []
    y_test = []
    i = 0
    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('networks/gtsrb/Final_Test/Images/',file_name)
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # predict and evaluate
    y_pred = model.predict_classes(X_test)
    acc = np.sum(y_pred==y_test)/np.size(y_pred)
    print("Test accuracy = {}".format(acc))
    
    return 
    
def read_model_from_file(weightFile,modelFile):
    """
    define neural network model
    :return: network model
    """
    
    model, batch_size, nb_epoch, lr = build_model()
    model.summary()
    
    weights = sio.loadmat(weightFile)
    model = model_from_json(open(modelFile).read())
    for (idx,lvl) in [(1,0),(2,1),(3,4),(4,5),(5,8),(6,9),(7,13),(8,15)]:
        
        weight_1 = 2 * idx - 2
        weight_2 = 2 * idx - 1
        model.layers[lvl].set_weights([weights['weights'][0, weight_1], weights['weights'][0, weight_2].flatten()])

    return model
    
              
def getImage(model,n_in_tests):

    X, Y = read_dataset()
    return X[n_in_tests]
    
    '''
    import pandas as pd
    import os

    test = pd.read_csv('networks/gtsrb/GT-final_test.csv',sep=';')

    file_name = test['Filename'][n_in_tests]
    img_path = os.path.join('networks/gtsrb/Final_Test/Images/',file_name)
    
    return preprocess_img(io.imread(img_path))
    '''
    
def getLabel(model,n_in_tests):

    X, Y = read_dataset()
    return Y[n_in_tests]
        
def getConfig(model):

    config = model.get_config()
    config = [ getLayerName(dict) for dict in config ]
    config = zip(range(len(config)),config)
    return config 
    
def getActivationValue(model,layer,image):

    image = np.expand_dims(image, axis=0)
    activations = get_activations(model, layer, image)
    return np.squeeze(activations)

    
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations
    
def predictWithImage(model,newInput):

    if len(newInput.shape) == 2: 
        newInput2 = np.expand_dims(np.expand_dims(newInput, axis=0), axis=0)
    else: 
        newInput2 = np.expand_dims(newInput, axis=0)
    predictValue = model.predict(newInput2)
    newClass = np.argmax(np.ravel(predictValue))
    confident = np.amax(np.ravel(predictValue))
    return (newClass,confident)    
    
def getWeightVector(model, layer2Consider):
    weightVector = []
    biasVector = []

    for layer in model.layers:
    	 index=model.layers.index(layer)
         h=layer.get_weights()
         
         if len(h) > 0 and index in [0,2]  and index <= layer2Consider: 
         # for convolutional layer
             ws = h[0]
             bs = h[1]
             
             #print("layer =" + str(index))
             #print(layer.input_shape)
             #print(ws.shape)
             #print(bs.shape)
             
             # number of filters in the previous layer
             m = len(ws)
             # number of features in the previous layer
             # every feature is represented as a matrix 
             n = len(ws[0])
             
             for i in range(1,m+1):
                 biasVector.append((index,i,h[1][i-1]))
             
             for i in range(1,m+1):
                 v = ws[i-1]
                 for j in range(1,n+1): 
                     # (feature, filter, matrix)
                     weightVector.append(((index,j),(index,i),v[j-1]))
                     
         elif len(h) > 0 and index in [7,10]  and index <= layer2Consider: 
         # for fully-connected layer
             ws = h[0]
             bs = h[1]
             
             # number of nodes in the previous layer
             m = len(ws)
             # number of nodes in the current layer
             n = len(ws[0])
             
             for j in range(1,n+1):
                 biasVector.append((index,j,h[1][j-1]))
             
             for i in range(1,m+1):
                 v = ws[i-1]
                 for j in range(1,n+1): 
                     weightVector.append(((index-1,i),(index,j),v[j-1]))
         #else: print "\n"
         
    return (weightVector,biasVector)        


    
def getLayerName(dict):

    className = dict.get('class_name')
    if className == 'Activation': 
        return dict.get('config').get('activation')
    else: 
        return className
        
    
              
