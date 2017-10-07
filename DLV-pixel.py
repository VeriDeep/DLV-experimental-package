#!/usr/bin/env python

"""
main file

author: Xiaowei Huang
"""

import sys
sys.path.append('networks')
sys.path.append('configuration')
sys.path.append('MCTS')

import time
import numpy as np
import copy 
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

from loadData import loadData 

from configuration import *
from basics import *
from networkBasics import *

from mcts import mcts
from mcts_twoPlayer import mcts_twoPlayer
from mcts_geo import mcts_geo

from dataCollection import dataCollection

from superPixels import superPixel_slic
from mnist_network import dynamic_build_model 

from inputManipulation import applyManipulation,assignManipulationSimple
from re_training import re_training

import theano
import theano.tensor as T
from keras import backend as K
        
def main():

    model = loadData()
    
    if whichMode == "train": return
    if trainingModel == "autoencoder":
        (model,autoencoder) = model
        if startLayer == -1: autoencoder = model
    else: autoencoder = model

    # finding adversarial examples from original model
    dc = dataCollection("%s_%s_%s"%(startIndexOfImage,dataProcessingBatchNum,manipulations[0]))
    succNum = 0 
    for i in range(dataProcessingBatchNum): 
        re = handleOne(model,autoencoder,startIndexOfImage+i,manipulations[0],dc)
        if re == True: succNum += 1
    dc.addSuccPercent(succNum/float(dataProcessingBatchNum))
    dc.provideDetails()
    dc.summarise()
    dc.close()


###########################################################################
#
#  checking with MCTS
#
############################################################################

def handleOne(model,autoencoder,startIndexOfImage,manipulationType,dc):
        
    print("start working on image %s ... "%startIndexOfImage)
        
    # get an image to interpolate
    global np
    image = np.squeeze(NN.getImage(model,startIndexOfImage))
    print("the shape of the input is "+ str(image.shape))
            
    # keep information for the original image
    (originalClass,originalConfident) = NN.predictWithImage(model,image)
    origClassStr = dataBasics.LABELS(int(originalClass))
    path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,originalConfident)
    dataBasics.save(-1,image, path0)
    
    dc.initialiseIndex(startIndexOfImage)
    dc.initialiseLayer(-1)
    
    # keep information for the activations
    if startLayer == -1: 
        activations = image
    else: 
        activations = NN.getActivationValue(model,startLayer,image)
    if len(activations.shape) == 2 and K.backend() == 'tensorflow':  
        output = np.squeeze(autoencoder.predict(np.expand_dims(np.expand_dims(activations,axis=2),axis=0)))
    elif len(activations.shape) == 2 and K.backend() == 'theano':  
        output = np.squeeze(autoencoder.predict(np.expand_dims(np.expand_dims(activations,axis=0),axis=0)))
    else:        
        output = np.squeeze(autoencoder.predict(np.expand_dims(activations,axis=0)))
        
    if startLayer > -1: 
        path0="%s/%s_autoencoder.png"%(directory_pic_string,startIndexOfImage)
        dataBasics.save(-1,output, path0)
        print("handling activations of layer %s with shape %s ... "%(startLayer, str(activations.shape)))
    
    # initialise a search tree
    if manipulationType == "sift_twoPlayer": 
        st= mcts_twoPlayer(model,autoencoder,image,activations,startLayer)
    else: 
        st = mcts(model,autoencoder,image,activations,startLayer)
    if startLayer > -1: 
        visualizeOneLayer(model,image,startLayer)
        st.visualizationMCTS()
    st.setManipulationType(manipulationType)
    st.initialiseActions()

    start_time_all = time.time()
    runningTime_all = 0
    numberOfMoves = 0
    while st.terminalNode(st.rootIndex) == False and st.terminatedByControlledSearch(st.rootIndex) == False and runningTime_all <= MCTS_all_maximal_time: 
        print("the number of moves we have made up to now: %s"%(numberOfMoves))
        eudist = st.euclideanDist(st.rootIndex)
        l1dist = st.l1Dist(st.rootIndex)
        l0dist = st.l0Dist(st.rootIndex)
        percent = st.diffPercent(st.rootIndex)
        diffs = st.diffImage(st.rootIndex)
        print("euclidean distance %s"%(eudist))
        print("L1 distance %s"%(l1dist))
        print("L0 distance %s"%(l0dist))
        print("manipulated percentage distance %s"%(percent))
        print("manipulated dimensions %s"%(diffs))

        start_time_level = time.time()
        runningTime_level = 0
        childTerminated = False
        while runningTime_level <= MCTS_level_maximal_time: 
            (leafNode,availableActions) = st.treeTraversal(st.rootIndex)
            newNodes = st.initialiseExplorationNode(leafNode,availableActions)
            for node in newNodes: 
                (childTerminated, value) = st.sampling(node,availableActions)
                #if childTerminated == True: break
                st.backPropagation(node,value)
            #if childTerminated == True: break
            runningTime_level = time.time() - start_time_level   
            nprint("best possible one is %s"%(str(st.bestCase)))
        bestChild = st.bestChild(st.rootIndex)
        #st.collectUselessPixels(st.rootIndex)
        st.makeOneMove(bestChild)
                
        image1 = st.applyManipulationToGetImage(st.spans[st.rootIndex],st.numSpans[st.rootIndex])
        diffs = st.diffImage(st.rootIndex)
        path0="%s/%s_temp_%s.png"%(directory_pic_string,startIndexOfImage,len(diffs))
        dataBasics.save(-1,image1,path0)
        (newClass,newConfident) = NN.predictWithImage(model,image1)
        print("confidence: %s"%(newConfident))
                
        if childTerminated == True: break
                
        # store the current best
        (_,bestSpans,bestNumSpans) = st.bestCase
        image1 = st.applyManipulationToGetImage(bestSpans,bestNumSpans)
        path0="%s/%s_currentBest.png"%(directory_pic_string,startIndexOfImage)
        dataBasics.save(-1,image1,path0)
                
        numberOfMoves += 1
        runningTime_all = time.time() - start_time_all  
        
    #print(st.terminalNode(st.rootIndex),st.terminatedByControlledSearch(st.rootIndex),runningTime_all,MCTS_all_maximal_time,childTerminated) 

    (_,bestSpans,bestNumSpans) = st.bestCase
    #image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
    image1 = st.applyManipulationToGetImage(bestSpans,bestNumSpans)
    (newClass,newConfident) = NN.predictWithImage(model,image1)
    newClassStr = dataBasics.LABELS(int(newClass))
    re = newClass != originalClass
                
    if re == True:     
        path0="%s/%s_%s_%s_modified_into_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,manipulationType, origClassStr,newClassStr,newConfident)
        dataBasics.save(-1,image1,path0)
        path0="%s/%s_diff.png"%(directory_pic_string,startIndexOfImage)
        dataBasics.save(-1,np.subtract(image,image1),path0)
        print("difference between images: %s"%(diffImage(image,image1)))

        #st.showDecisionTree()
        #pixelImages = st.analysePixels()
        #advImages = st.analyseAdv.analyse()
        #for (v,pixelImage) in pixelImages: 
        #    path0="%s/%s_useful_%s.png"%(directory_pic_string,startIndexOfImage,v)
        #    dataBasics.save(-1,pixelImage,path0)
            
        #for i in range(len(advImages)): 
        #    (advnum,advimg) = advImages[i]
        #    (advClass,advConfident) = NN.predictWithImage(model,advimg)
        #    advClassStr = dataBasics.LABELS(int(advClass))
        #    path0="%s/%s_adv_%s_%s_%s.png"%(directory_pic_string,startIndexOfImage,i,advnum,advClassStr)
        #    dataBasics.save(-1,advimg,path0)
        
        print("number of adversarial examples found: %s"%(st.numAdv))
    
        eudist = euclideanDistance(st.image,image1)
        l1dist = l1Distance(st.image,image1)
        l0dist = l0Distance(st.image,image1)
        percent = diffPercent(st.image,image1)
        print("euclidean distance %s"%(eudist))
        print("L1 distance %s"%(l1dist))
        print("L0 distance %s"%(l0dist))
        print("manipulated percentage distance %s"%(percent))
        print("class is changed into %s with confidence %s\n"%(newClassStr, newConfident))
        dc.addRunningTime(time.time() - start_time_all)
        dc.addConfidence(newConfident)
        dc.addManipulationPercentage(percent)
        dc.addEuclideanDistance(eudist)
        dc.addl1Distance(l1dist)
        dc.addl0Distance(l0dist)
    else: 
        print("failed to find an adversary image within prespecified bounded computational resource. ")
                
    newXtrain,newYtrain = st.re_training.returnData()
    st.destructor()
                
    runningTime = time.time() - start_time   

    return re
            
if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    