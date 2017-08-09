#!/usr/bin/env python

"""
main file

author: Xiaowei Huang
"""

import sys
sys.path.append('networks')
sys.path.append('configuration')
sys.path.append('MCTS')
sys.path.append('optimisation')

import time
import numpy as np
import copy 
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

from keras import backend as K

from loadData import loadData 

from configuration import *
from basics import *
from networkBasics import *

from mcts import mcts
from mcts_geo import mcts_geo

from dataCollection import dataCollection

from superPixels import superPixel_slic
from mnist_network import dynamic_build_model 

from inputManipulation import applyManipulation,assignManipulationSimple
from re_training import re_training

from test_attack import test_attack

import theano
import theano.tensor as T
import tensorflow as tf
        
def main():

    with tf.Session() as sess:
    
        model = loadData()
            
        if whichMode == "train": return
        if trainingModel == "autoencoder":
            (model,autoencoder) = model
            if startLayer == -1: autoencoder = model
        else: autoencoder = model
    
        images = []
        labels = []
    
        for i in range(dataProcessingBatchNum): 
    
            imageIndex = startIndexOfImage + i
    
            image = NN.getImage(model,imageIndex) 
            label = NN.getLabel(model,imageIndex)
    
            # keep information for the original image
            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            origClassStr = dataBasics.LABELS(int(originalClass))
            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,imageIndex,origClassStr,originalConfident)
            dataBasics.save(-1, np.squeeze(image), path0)
        
            images.append(image - 0.5)
            labels.append(label)
            
        end_vars = tf.global_variables()
        test_attack(sess,model,np.array(images),np.array(labels))
    
    return
            

if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    