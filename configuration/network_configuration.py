#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""
import numpy as np
import os

import mnist_network as NN_mnist
import cifar10_network as NN_cifar10
import imageNet_network as NN_imageNet
import gtsrb_network as NN_gtsrb


import mnist
import cifar10
import imageNet
import gtsrb


#######################################################
#
#  some auxiliary parameters that are used in several files
#  they can be seen as global parameters for an execution
#
#######################################################

def network_parameters(dataset): 
    
# which dataset to analyse
    if dataset == "mnist": 
        NN = NN_mnist
        dataBasics = mnist
        filterSize = 3 
        directory_model_string = makedirectory("networks/mnist")
        directory_statistics_string = makedirectory("data/mnist_statistics")
        directory_pic_string = makedirectory("data/mnist_pic")
        
        featureDims = 4 # 20 #  20
        span = 255/float(255)
        numSpan = 1

    elif dataset == "cifar10": 
        NN = NN_cifar10
        dataBasics = cifar10
        filterSize = 3 
        directory_model_string = makedirectory("networks/cifar10")
        directory_statistics_string = makedirectory("data/cifar10_statistics")
        directory_pic_string = makedirectory("data/cifar10_pic")
 
        featureDims = 9
        span = 255/float(255)
        numSpan = 1
        
    elif dataset == "gtsrb": 
        NN = NN_gtsrb
        dataBasics = gtsrb
        filterSize = 3 
        directory_model_string = makedirectory("networks/gtsrb")
        directory_statistics_string = makedirectory("data/gtsrb_statistics")
        directory_pic_string = makedirectory("data/gtsrb_pic")
        
        featureDims = 9 # 81 #
        span = 255/float(255)
        numSpan = 1
                
    elif dataset == "imageNet": 
        NN = NN_imageNet
        dataBasics = imageNet
        filterSize = 3 
        directory_model_string = makedirectory("networks/imageNet")
        directory_statistics_string = makedirectory("data/imageNet_statistics")
        directory_pic_string = makedirectory("data/imageNet_pic")

        featureDims = 256
        span = 125
        numSpan = 1

    return (featureDims,span,numSpan,NN,dataBasics,directory_model_string,directory_statistics_string,directory_pic_string,filterSize)

def makedirectory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name