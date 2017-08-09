#!/usr/bin/env python

"""
The main file for load models 

author: Xiaowei Huang
"""

import sys
sys.path.append('networks')


import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pylab import *

# keras
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense
import keras.optimizers
from keras import backend as K
import tensorflow as tf

# visualisation
#from keras.utils.visualize_util import plot
#
from keras.datasets import mnist
from keras.utils import np_utils

# for training cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint



from configuration import *


# training the model from data
# or read the model from saved data file
# then start analyse the model 
def loadData():
    
    # construct model according to the flage
    # whichMode == "read" read from saved file 
    # whichMode == "train" training from the beginning 
    if whichMode == "train" and dataset == "mnist" and trainingModel == "normal": 
    
        (X_train, Y_train, X_test, Y_test, batch_size, nb_epoch) = NN.read_dataset()
        X_train_transferability = X_train[1:150]
        Y_train_transferability = Y_train[1:150]
        nb_epoch_transferability = 5
    
        print "Building network model ......"
        model = NN.build_model()

        start_time = time.time()
        model.fit(X_train, Y_train, batch_size=nb_epoch, nb_epoch=nb_epoch,
                  verbose=1, validation_data=(X_test, Y_test))
        #model.fit(X_train_transferability, Y_train_transferability, batch_size=batch_size, nb_epoch=nb_epoch_transferability,
        #          verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        print("Fitting time: --- %s seconds ---" % (time.time() - start_time))   
        print("Training finished!")
    
        # save model
        if K.backend() == 'tensorflow':
            ae = "_tensorflow"
        else: 
            ae = "_theano" # "transferability" #  "_autoencoder"  #
        json_string = model.to_json()
        open('%s/mnist%s.json'%(directory_model_string,ae), 'w').write(json_string)
        model.save_weights('%s/mnist%s.h5'%(directory_model_string,ae), overwrite=True)
        sio.savemat('%s/mnist%s.mat'%(directory_model_string,ae), {'weights': model.get_weights()})
        print("Model saved!")
        
    elif whichMode == "read" and dataset == "mnist" and trainingModel == "normal": 
        print("Start loading model ... ")
        if K.backend() == 'tensorflow': 
            ae = "_tensorflow"
        else: 
            ae = "_theano" # "transferability" #  "_autoencoder"  #
        model = NN.read_model_from_file('%s/mnist%s.mat'%(directory_model_string,ae),'%s/mnist%s.json'%(directory_model_string,ae))
        #print model.get_config()
        print("Model loaded!")
        #test(model)
        
    elif whichMode == "train" and dataset == "mnist" and trainingModel == "autoencoder": 
    
        (X_train, Y_train, X_test, Y_test, batch_size, nb_epoch) = NN.read_dataset()
        
        # redefine nb_epoch, need to remove this for better training
        #nb_epoch = 30

        # consider the layer 3 of the original model
        layerToCut = startLayer
        
        print "Building network model_and_autoencoder ......"
        model = NN.build_model_and_autoencoder(layerToCut) 
        
        ae =  ""  
        model = NN.read_model_and_autoencoder_from_file(model,'%s/mnist%s.mat'%(directory_model_string,ae),'%s/mnist%s.json'%(directory_model_string,ae),layerToCut)
        
        start_time = time.time()
        model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

        #model.fit(X_train_transferability, Y_train_transferability, batch_size=batch_size, nb_epoch=nb_epoch_transferability,
        #          verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, X_test, verbose=0, batch_size=batch_size)

        print('Test score:', score)
        print("Fitting time: --- %s seconds ---" % (time.time() - start_time))   
        print("Training finished!")
    
        # save model
        ae =  "_autoencoder_layer%s"%layerToCut  
        json_string = model.to_json()
        open('%s/mnist%s.json'%(directory_model_string,ae), 'w').write(json_string)
        model.save_weights('%s/mnist%s.h5'%(directory_model_string,ae), overwrite=True)
        sio.savemat('%s/mnist%s.mat'%(directory_model_string,ae), {'weights': model.get_weights()})
        print("Model saved!")

    elif whichMode == "read" and dataset == "mnist" and trainingModel == "autoencoder": 
        print("Start loading model ... ")
        
        # consider the layer 3 of the original model
        layerToCut = startLayer
        
        ae = "" # "transferability" #  "_autoencoder"  # "_normal"
        model = NN.read_model_from_file('%s/mnist%s.mat'%(directory_model_string,ae),'%s/mnist%s.json'%(directory_model_string,ae))
        print("Start loading autoencoder ... ")
        ae = "_autoencoder_layer%s"%layerToCut  # "_normal"
        autoencoder = NN.read_autoencoder_from_file('%s/mnist%s.mat'%(directory_model_string,ae),'%s/mnist%s.json'%(directory_model_string,ae),layerToCut)
        print("Model loaded!")
        model = (model,autoencoder)
        #test(model)
        
    elif whichMode == "train" and dataset == "gtsrb": 
    
        X_train, Y_train = dataBasics.read_dataset()
        
        #print X_train.shape, Y_train.shape, Y_train[0]

        print "Building network model ......"
        model, batch_size, nb_epoch, lr = NN.build_model()
        
        def lr_schedule(epoch):
            return lr*(0.1**int(epoch/10))

        start_time = time.time()

        model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    validation_split=0.2,
                    shuffle=True,
                    callbacks=[LearningRateScheduler(lr_schedule),
                                ModelCheckpoint('model.h5',save_best_only=True)]
                  )

        #score = model.evaluate(X_test, Y_test, verbose=0)
        # save model
        ae =  "_32" # "_48" # "_28" #   "_normal" # "_autoencoder"  #
        json_string = model.to_json()
        open('%s/gtsrb%s.json'%(directory_model_string,ae), 'w').write(json_string)
        model.save_weights('%s/gtsrb%s.h5'%(directory_model_string,ae), overwrite=True)
        sio.savemat('%s/gtsrb%s.mat'%(directory_model_string,ae), {'weights': model.get_weights()})
        print("Model saved!")
        
    elif whichMode == "read" and dataset == "gtsrb": 
        print("Start loading model ... ")
        ae =  "_32" # "_48" # "_28" #   "_normal" # "_autoencoder"  #
        model = NN.read_model_from_file('%s/gtsrb%s.mat'%(directory_model_string,ae),'%s/gtsrb%s.json'%(directory_model_string,ae))
        print("Model loaded!")
        #test(model)

    elif whichMode == "train" and dataset == "cifar10" and trainingModel == "normal" : 
    
        (X_train,Y_train,X_test,Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()
           
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        
        print "Building network model ......"
        model = NN.build_model(img_channels, img_rows, img_cols, nb_classes)
        ae = ""

        start_time = time.time()
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      validation_data=(X_test, Y_test),
                      shuffle=True)
        else:
            print('Using real-time data augmentation.')
            print X_train.shape[0]

            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(X_train)

            # fit the model on the batches generated by datagen.flow()
            model.fit_generator(datagen.flow(X_train, Y_train,
                                batch_size=batch_size),
                                samples_per_epoch=X_train.shape[0],
                                nb_epoch=1,
                                validation_data=(X_test, Y_test))
                                
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:%s'%score)
        print("Fitting time: --- %s seconds ---" % (time.time() - start_time))   
        print("Training finished!")
    
        # save model
        json_string = model.to_json()
        open('%s/cifar10%s.json'%(directory_model_string,ae), 'w').write(json_string)
        model.save_weights('%s/cifar10%s.h5'%(directory_model_string,ae), overwrite=True)
        sio.savemat('%s/cifar10%s.mat'%(directory_model_string,ae), {'weights': model.get_weights()})
        print("Model saved!")
        
        
    elif whichMode == "read" and dataset == "cifar10" and trainingModel == "normal": 
        print("Start loading model ... ")
        ae = ""
        (X_train,Y_train,X_test,Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()
        model = NN.read_model_from_file(img_channels, img_rows, img_cols, nb_classes, '%s/cifar10%s.mat'%(directory_model_string,ae),'%s/cifar10%s.json'%(directory_model_string,ae))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        print("Model loaded!")
        
    elif whichMode == "train" and dataset == "cifar10" and trainingModel == "autoencoder" : 
    
        (X_train,Y_train,X_test,Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()
           
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        
        # temporary setting
        #nb_epoch = 1
        
        print "Building network model ......"
        model = NN.build_model_and_autoencoder(img_channels, img_rows, img_cols, nb_classes)
        
        # consider the layer of the original model
        layerToCut = startLayer

        ae =  ""  
        model = NN.read_model_and_autoencoder_from_file(model,'%s/cifar10%s.mat'%(directory_model_string,ae),'%s/cifar10%s.json'%(directory_model_string,ae),layerToCut)

        Y_train = X_train
        Y_test = X_test
        ae = "_%s_layer%s"%(trainingModel,layerToCut)
        
        start_time = time.time()
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      validation_data=(X_test, Y_test),
                      shuffle=True)
        else:
            print('Using real-time data augmentation.')
            print X_train.shape[0]

            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(X_train)

            # fit the model on the batches generated by datagen.flow()
            model.fit_generator(datagen.flow(X_train, Y_train,
                                batch_size=batch_size),
                                samples_per_epoch=X_train.shape[0],
                                nb_epoch=nb_epoch,
                                validation_data=(X_test, Y_test))
                                
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:%s'%score)
        print("Fitting time: --- %s seconds ---" % (time.time() - start_time)) 
        print("Training finished!")
    
        # save model
        json_string = model.to_json()
        open('%s/cifar10%s.json'%(directory_model_string,ae), 'w').write(json_string)
        model.save_weights('%s/cifar10%s.h5'%(directory_model_string,ae), overwrite=True)
        sio.savemat('%s/cifar10%s.mat'%(directory_model_string,ae), {'weights': model.get_weights()})
        print("Model saved!")
  
        
    elif whichMode == "read" and dataset == "cifar10" and trainingModel == "autoencoder": 
        print("Start loading model ... ")
        
        # consider the layer of the original model
        layerToCut = startLayer
        
        ae = ""
        (X_train,Y_train,X_test,Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()
        model = NN.read_model_from_file(img_channels, img_rows, img_cols, nb_classes, '%s/cifar10%s.mat'%(directory_model_string,ae),'%s/cifar10%s.json'%(directory_model_string,ae))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        ae = "_%s_layer%s"%(trainingModel,layerToCut)
        autoencoder = NN.read_autoencoder_from_file(img_channels, img_rows, img_cols, nb_classes, '%s/cifar10%s.mat'%(directory_model_string,ae),'%s/cifar10%s.json'%(directory_model_string,ae), layerToCut)
        autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        model = (model,autoencoder)
        print("Model loaded!")

        
    elif whichMode == "train" and dataset == "imageNet": 
    
        (img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()
        
        print "Building network model ......"
        model = NN.build_model(img_channels, img_rows, img_cols, nb_classes)
    
        # load weights
        model.load_weights('%s/imageNet.h5'%directory_model_string)

        # save model
        json_string = model.to_json()
        open('%s/imageNet.json'%directory_model_string, 'w').write(json_string)
        model.save_weights('%s/imageNet.h5'%directory_model_string, overwrite=True)
        sio.savemat('%s/imageNet.mat'%directory_model_string, {'weights': model.get_weights()})
        print("Model saved!")
        
    elif whichMode == "read" and dataset == "imageNet": 
        print("Start loading model ... ")
        (img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()
        model = NN.read_model_from_file(img_channels, img_rows, img_cols, nb_classes, '%s/imageNet.mat'%directory_model_string,'%s/imageNet.json'%directory_model_string)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        print("Model loaded!")
        
    elif whichMode == "train" and dataset == "twoDcurve": 
        # define and construct model

        # load data
        N_samples = 5000
        N_tests = 1000
        x_train, y_train, x_test, y_test = NN.load_data(N_samples,N_tests)
        
        print "Building network model ......"
        model = NN.build_model()
    
        plot(model, to_file='twoDcurve_pic/model.png')

        # visualisation

        # configure learning process
        sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss={'output': 'mse'})

        model.summary()

        start_time = time.time()
        model.fit({'data': x_train}, {'output': y_train}, nb_epoch=3000, validation_split=0.1, verbose=0)
        print("Fitting time: --- %s seconds ---" % (time.time() - start_time))
    
        print("Training finished!")

        # save model
        json_string = model.to_json()
        open('%s/MLP.json'%directory_model_string, 'w').write(json_string)
        model.save_weights('%s/MLP.h5'%directory_model_string, overwrite=True)
        sio.savemat('%s/MLP.mat'%directory_model_string, {'weights': model.get_weights()})
        print("Model saved!")

    elif whichMode == "read" and dataset == "twoDcurve": 
        print("Start loading model ... ")
        model = NN_twoDcurve.read_model_from_file('%s/MLP.mat'%directory_model_string,'%s/MLP.json'%directory_model_string)
        #model.summary()
        print("Model loaded!")

    
    return (model)




