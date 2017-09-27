#!/usr/bin/env python

from __future__ import print_function

import scipy.io as sio
import numpy as np
import struct
from array import array as pyarray
from PIL import Image

from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, UpSampling2D, Deconvolution2D, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
#from keras.layers.convolutional_transpose import Convolution2D_Transpose

# for mnist
from keras.datasets import mnist
import tensorflow as tf

#
import mnist as mm



batch_size = 128
nb_classes = 10
nb_epoch = 12
img_channels = 1

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3




def read_dataset():

    # parameters for neural network
    batch_size = 128
    nb_classes = 10
    nb_epoch = 12

    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.backend() == 'tensorflow':
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    return (X_train,Y_train,X_test,Y_test, batch_size, nb_epoch)

def build_model(whichMode = "train"):
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

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    

    if whichMode != "read": 
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
                  

    return model
    
    

def build_model_and_autoencoder(layerToCut):
    """
    define autoencoder model
    this one connect the first two conv levels from the model
    
    """

    if K.backend() == 'tensorflow': 
        inputShape = (img_rows,img_cols,img_channels)
    else: 
        inputShape = (img_channels,img_rows,img_cols)

    model = Sequential()

    model.add(Convolution2D(32, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=inputShape,
                            trainable = False))
    if layerToCut >= 1: 
        model.add(Activation('relu'))
    if layerToCut >= 2: 
        model.add(Convolution2D(32, nb_conv, nb_conv,trainable = False))
    if layerToCut >= 3: 
        model.add(Activation('relu'))

    model.add(ZeroPadding2D((1, 1)))
    if layerToCut >= 2: 
        model.add(ZeroPadding2D((1, 1)))
    
    model.add(Convolution2D(16, nb_conv, nb_conv,activation='relu', border_mode='same'))
    
    model.add(MaxPooling2D((nb_pool, nb_pool), border_mode='same'))
    
    model.add(Convolution2D(8, nb_conv, nb_conv,activation='relu', border_mode='same'))
    model.add(MaxPooling2D((nb_pool, nb_pool), border_mode='same'))
    model.add(Convolution2D(8, nb_conv, nb_conv,activation='relu', border_mode='same'))
    model.add(MaxPooling2D((nb_pool, nb_pool), border_mode='same'))

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    model.add(Convolution2D(8, nb_conv, nb_conv,activation='relu', border_mode='same'))
    model.add(UpSampling2D((nb_pool, nb_pool)))
    model.add(Convolution2D(8, nb_conv, nb_conv,activation='relu', border_mode='same'))
    model.add(UpSampling2D((nb_pool, nb_pool)))
    model.add(Convolution2D(16, nb_conv, nb_conv,activation='relu'))
    model.add(UpSampling2D((nb_pool, nb_pool)))
    model.add(Convolution2D(1, nb_conv, nb_conv,activation='sigmoid', border_mode='same'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    #model.summary()

    return model
    
    
def build_autoencoder(layerToCut):
    """
    define neural network model
    this one removes the conv levels of the model 
    """

    model = Sequential()
    
    if layerToCut >= 2: 
        model.add(ZeroPadding2D((1, 1),input_shape=(32, img_rows-4, img_cols-4)))
        model.add(ZeroPadding2D((1, 1)))
    else: 
        model.add(ZeroPadding2D((1, 1),input_shape=(32, img_rows-2, img_cols-2)))
    
    model.add(Convolution2D(16, nb_conv, nb_conv,activation='relu', border_mode='same'))
    
    model.add(MaxPooling2D((nb_pool, nb_pool), border_mode='same'))
    
    model.add(Convolution2D(8, nb_conv, nb_conv,activation='relu', border_mode='same'))
    model.add(MaxPooling2D((nb_pool, nb_pool), border_mode='same'))
    model.add(Convolution2D(8, nb_conv, nb_conv,activation='relu', border_mode='same'))
    model.add(MaxPooling2D((nb_pool, nb_pool), border_mode='same'))

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    model.add(Convolution2D(8, nb_conv, nb_conv,activation='relu', border_mode='same'))
    model.add(UpSampling2D((nb_pool, nb_pool)))
    model.add(Convolution2D(8, nb_conv, nb_conv,activation='relu', border_mode='same'))
    model.add(UpSampling2D((nb_pool, nb_pool)))
    model.add(Convolution2D(16, nb_conv, nb_conv,activation='relu'))
    model.add(UpSampling2D((nb_pool, nb_pool)))
    model.add(Convolution2D(1, nb_conv, nb_conv,activation='sigmoid', border_mode='same'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    #model.summary()

    return model



def dynamic_build_model(startLayer,inputShape):
    """
    define neural network model
    """
    
    if K.backend() == 'tensorflow': 
        inputShape = (img_rows,img_cols,img_channels)
    else: 
        inputShape = (img_channels,img_rows,img_cols)
    
    firstLayerDone = False
    model = Sequential()
    
    if startLayer == 0 :
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='valid',
                                input_shape=inputShape))
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Activation('relu',input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Activation('relu'))
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv, input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Activation('relu',input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Activation('relu'))
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool),input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Dropout(0.25,input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Dropout(0.25))
    else: 
        startLayer -= 1

    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Flatten(input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Flatten())
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Dense(128,input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Dense(128))
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Activation('relu',input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Activation('relu'))
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Dropout(0.5,input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Dropout(0.5))
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Dense(nb_classes,input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Dense(nb_classes))
    else: 
        startLayer -= 1
        
    if startLayer == 0 :
        if firstLayerDone == False: 
            model.add(Activation('softmax',input_shape=inputShape))
            firstLayerDone = True
        else:  
            model.add(Activation('softmax'))
    else: 
        startLayer -= 1

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model
    

def read_model_from_file(weightFile,modelFile):
    """
    define neural network model
    :return: network model
    """
    
    model = build_model(whichMode = "read")
    model.summary()


    #print (model.get_config())
        
    if K.backend() == 'tensorflow': 
        model.load_weights('networks/mnist/mnist_tensorflow.h5')
    else: 
        weights = sio.loadmat(weightFile)
        model = model_from_json(open(modelFile).read())
        for (idx,lvl) in [(1,0),(2,2),(3,7),(4,10)]:
        
            weight_1 = 2 * idx - 2
            weight_2 = 2 * idx - 1
            model.layers[lvl].set_weights([weights['weights'][0, weight_1], weights['weights'][0, weight_2].flatten()])
    
    return model
    
def read_autoencoder_from_file(weightFile,modelFile,layerToCut):
    """
    define neural network model
    :return: network model
    """
    
    model = build_autoencoder(layerToCut)
    model.summary()
    
    if layerToCut < 0: 
        return model 
        
    weights = sio.loadmat(weightFile)
    #model = model_from_json(open(modelFile).read())
    if layerToCut >= 2: 
        l = 3
        k = 0
    else: 
        l = 2
        k = 1
    for (idx,lvl) in [(l,7-k),(l+1,9-k),(l+2,11-k),(l+3,13-k),(l+4,15-k),(l+5,17-k),(l+6,19-k)]:
        
        weight_1 = 2 * idx - 2
        weight_2 = 2 * idx - 1
        model.layers[lvl-5].set_weights([weights['weights'][0, weight_1], weights['weights'][0, weight_2].flatten()])

    return model
    
def read_model_and_autoencoder_from_file(model,weightFile,modelFile,cutLayer):
    """
    define neural network model
    :return: network model
    """
    
    weights = sio.loadmat(weightFile)
    layerList = [(1,0),(2,2),(3,7),(4,10)]
    layerList = [ (x,y) for (x,y) in layerList if y <= cutLayer ]
    for (idx,lvl) in layerList:
        
        weight_1 = 2 * idx - 2
        weight_2 = 2 * idx - 1
        model.layers[lvl].set_weights([weights['weights'][0, weight_1], weights['weights'][0, weight_2].flatten()])

    return model

    
def dynamic_read_model_from_file(cutmodel,weightFile,modelFile,startLayer):
    """
    define neural network model
    :return: network model
    """
    
    cutmodel.summary()
    
    weights = sio.loadmat(weightFile)
    for (idx,lvl) in [(1,0),(2,2),(3,7),(4,10)]:
        
        weight_1 = 2 * idx - 2
        weight_2 = 2 * idx - 1
        if lvl-startLayer >= 0: 
            cutmodel.layers[lvl-startLayer].set_weights([weights['weights'][0, weight_1], weights['weights'][0, weight_2].flatten()])

    return cutmodel
    

    
"""
   The following function gets the activations for a particular layer
   for an image in the test set. 
   FIXME: ideally I would like to be able to 
          get activations for a particular layer from the inputs of another layer. 
"""

def getImage(model,n_in_tests):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if K.backend() == 'tensorflow': 
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    else: 
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_test = X_test.astype('float32')
    X_test /= 255
    
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    image = X_test[n_in_tests:n_in_tests+1]
    if K.backend() == 'tensorflow':
        return image[0]
    else: 
        return np.squeeze(image)
        
def getLabel(model,n_in_tests):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if K.backend() == 'tensorflow': 
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    else: 
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_test = X_test.astype('float32')
    X_test /= 255
    
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    image = Y_test[n_in_tests:n_in_tests+1]
    if K.backend() == 'tensorflow':
        return image[0]
    else: 
        return np.squeeze(image)
    
def getImages(model,n_in_tests1,n_in_tests2):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if K.backend() == 'tensorflow': 
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    else: 
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_test = X_test.astype('float32')
    X_test /= 255
    
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    image2 = X_test[n_in_tests1:n_in_tests2]
    return np.squeeze(image2)
    
def readImage(path):

    #import cv2

    #im = cv2.resize(cv2.imread(path), (img_rows, img_cols)).astype('float32')
    #im = im / 255
    #im = im.transpose(2, 0, 1)

    #print("ERROR: currently the reading of MNIST images are not correct, so the classifications are incorrect. ")
    
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import PIL
    from PIL import Image
    img=rgb2gray(mpimg.imread(path))
        
    img = img.resize((img_cols, img_rows))
    return np.squeeze(img)
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def getActivationValue(model,layer,image):

    if len(image.shape) == 2: 
        image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    elif len(image.shape) == 3: 
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
    #newInput2.astype('float32')
    #print(newInput2.shape)
    #print(model.get_config())
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

def getConfig(model):

    config = model.get_config()
    if 'layers' in config: config = config['layers']
    config = [ getLayerName(dict) for dict in config ]
    config = zip(range(len(config)),config)
    return config 
    
def getLayerName(dict):

    className = dict.get('class_name')
    if className == 'Activation': 
        return dict.get('config').get('activation')
    else: 
        return className