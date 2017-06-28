#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""

from network_configuration import *
from usual_configuration import * 


#######################################################
#
#  The following are parameters to indicate how to work 
#   with a problem
#
#######################################################

# which dataset to work with
#dataset = "mnist"
dataset = "gtsrb"
#dataset = "cifar10"
#dataset = "imageNet"

# the network is trained from scratch
#  or read from the saved files
whichMode = "read"
#whichMode = "train"

# whether working with hidden layer
# normal means working with input layer
# autoencoder means working with hidden layer
#trainingModel = "autoencoder"
trainingModel = "normal"

# the number of images to be handled 
dataProcessingBatchNum = 1


firstRound_manipulations = ["slic", "squares", "pixelSets"]
sndRound_manipulations = ["squares", "pixelSets", "slic"]


#######################################################
#  get parameters from network_configuration
#######################################################

(featureDims,span,numSpan,NN,dataBasics,directory_model_string,directory_statistics_string,directory_pic_string,filterSize) = network_parameters(dataset)


#######################################################
#  specific parameters for datasets
#######################################################


(startIndexOfImage,startLayer, explorationRate,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,effectiveConfidenceWhenChanging) = usual_configuration(dataset)
    

#######################################################
#
#  show detailedInformation or not
#  FIXME: check to see if they are really needed/used
#
#######################################################

def nprint(str):
    return      
        