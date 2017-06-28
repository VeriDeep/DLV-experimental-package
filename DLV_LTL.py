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
from mcts_geo import mcts_geo

from dataCollection import dataCollection

from superPixels import superPixel_slic
from mnist_network import dynamic_build_model 

from inputManipulation import applyManipulation,assignManipulationSimple
from re_training import re_training

import theano
import theano.tensor as T

        
def main():

    re-training()
    # explaination()

def explaination():

    model = loadData()
    
    if whichMode == "train": return
    
    if trainingModel == "autoencoder":
        (model,autoencoder) = model
        if startLayer == -1: autoencoder = model
    else: autoencoder = model
    
    # initialise a dataCollection instance
    phase = "firstRound"
    dc = dataCollection("%s_%s_%s"%(startIndexOfImage,dataProcessingBatchNum, controlledSearch))
    # initialise a re_training instance
    reTrain = re_training(model, NN.getImage(model,startIndexOfImage).shape)
    
    #originalScore = reTrain.evaluateWithOriginalModel()
    #dc.addComment("original test score: %s\n\n"%originalScore)
    # finding adversarial examples from original model
    succNum = 0
    for whichIndex in range(startIndexOfImage,startIndexOfImage + dataProcessingBatchNum):
        print "\n\nprocessing input of index %s in the dataset: " %(str(whichIndex))
        succ = handleOne(model,autoencoder,dc,reTrain,phase,whichIndex,firstRound_manipulations[0])
        if succ == True: succNum += 1
    dc.addSuccPercent(succNum/float(dataProcessingBatchNum))
            
    # output statistics for original model
    print("Please refer to the file %s for statistics."%(dc.fileName))
    dc.provideDetails()
    dc.summarise()
    dc.close()



def re-training():


    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)

    for firstRound_manipulation in firstRound_manipulations: 
        for sndRound_manipulation in sndRound_manipulations: 
            reTraining_experiment(firstRound_manipulation,sndRound_manipulation)
            
def reTraining_experiment(firstRound_manipulation,sndRound_manipulation):

    model = loadData()
    
    if whichMode == "train": return
    
    if trainingModel == "autoencoder":
        (model,autoencoder) = model
        if startLayer == -1: autoencoder = model
    else: autoencoder = model
    
    # initialise a dataCollection instance
    phase = "firstRound"
    dc = dataCollection("%s_%s_%s_%s_%s_firstRound"%(startIndexOfImage,dataProcessingBatchNum, controlledSearch, firstRound_manipulation, sndRound_manipulation))
    # initialise a re_training instance
    reTrain = re_training(model, NN.getImage(model,startIndexOfImage).shape)
    
    originalScore = reTrain.evaluateWithOriginalModel()
    dc.addComment("original test score: %s\n\n"%originalScore)
    # finding adversarial examples from original model
    succNum = 0
    for whichIndex in range(startIndexOfImage,startIndexOfImage + dataProcessingBatchNum):
        print "\n\nprocessing input of index %s in the dataset: " %(str(whichIndex))
        succ = handleOne(model,autoencoder,dc,reTrain,phase,whichIndex,firstRound_manipulation)
        if succ == True: succNum += 1
    dc.addSuccPercent(succNum/float(dataProcessingBatchNum))
            
    # output statistics for original model
    print("Please refer to the file %s for statistics."%(dc.fileName))
    dc.provideDetails()
    dc.summarise()
    dc.close()
    
    if reTrain.numberOfNewExamples() == 0: 
        print "failed to find new examples for further training"
        return
    else: 
        print "ready for re-training ... "
    
    # initialise a dataCollection instance
    phase = "sndRound"
    dc = dataCollection("%s_%s_%s_%s_%s_secondRound"%(startIndexOfImage,dataProcessingBatchNum, controlledSearch, firstRound_manipulation, sndRound_manipulation))
    dc.addComment("%s new examples are founded.\n\n"%(reTrain.numberOfNewExamples()))
    # update model with new data
    reTrain.setReTrainedModelName("%s_%s_%s_%s"%(startIndexOfImage,dataProcessingBatchNum, controlledSearch,firstRound_manipulation))
    model = reTrain.training()
    
    updatedScore = reTrain.evaluateWithUpdatedModel()
    dc.addComment("updated test score: %s\n\n"%updatedScore)
    # finding adversarial examples from updated model
    succNum = 0
    for whichIndex in range(startIndexOfImage,startIndexOfImage + dataProcessingBatchNum):
        print "\n\nprocessing input of index %s in the dataset: " %(str(whichIndex))
        succ = handleOne(model,autoencoder,dc,reTrain,phase,whichIndex,sndRound_manipulation)
        if succ == True: succNum += 1
    dc.addSuccPercent(succNum/float(dataProcessingBatchNum))

    # output statistics for updated model
    print("Please refer to the file %s for statistics."%(dc.fileName))
    dc.provideDetails()
    dc.summarise()
    dc.close()


    
      
###########################################################################
#
#  checking with MCTS
#
############################################################################

def handleOne(model,autoencoder,dc,reTrain,phase,startIndexOfImage,manipulationType):
        
    # visualisation, switch on if needed
    #visualization(model,501)
    #return

    # get an image to interpolate
    global np
    image = NN.getImage(model,startIndexOfImage)
    print("the shape of the input is "+ str(image.shape))
    
    #superPixel_slic(image)
    #return
            
    dc.initialiseIndex(startIndexOfImage)    
    dc.initialiseLayer(startLayer)
            
    # keep information for the original image
    (originalClass,originalConfident) = NN.predictWithImage(model,image)
    origClassStr = dataBasics.LABELS(int(originalClass))
    path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,originalConfident)
    dataBasics.save(-1,image, path0)
    
    # keep information for the activations
    if startLayer == -1: 
        activations = image
    else: 
        activations = NN.getActivationValue(model,startLayer,image)
    if len(activations.shape) == 2:  
        output = np.squeeze(autoencoder.predict(np.expand_dims(np.expand_dims(activations,axis=0),axis=0)))
    else:        
        output = np.squeeze(autoencoder.predict(np.expand_dims(activations,axis=0)))
        
    if startLayer > -1: 
        path0="%s/%s_autoencoder.png"%(directory_pic_string,startIndexOfImage)
        dataBasics.save(-1,output, path0)
        print "handling activations of layer %s with shape %s ... "%(startLayer, str(activations.shape))
    
    # initialise a search tree
    st = mcts(model,autoencoder,image,activations,startLayer)
    #st = mcts_geo(model,autoencoder,image,activations,startLayer)
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
        percent = st.diffPercent(st.rootIndex)
        diffs = st.diffImage(st.rootIndex)
        print "euclidean distance %s"%(eudist)
        print "L1 distance %s"%(l1dist)
        print "manipulated percentage distance %s"%(percent)
        print "manipulated dimensions %s"%(diffs)

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
        print "confidence: %s"%(newConfident)
                
        if childTerminated == True: break
                
        # store the current best
        (_,bestSpans,bestNumSpans) = st.bestCase
        image1 = st.applyManipulationToGetImage(bestSpans,bestNumSpans)
        path0="%s/%s_currentBest.png"%(directory_pic_string,startIndexOfImage)
        dataBasics.save(-1,image1,path0)
                
        numberOfMoves += 1
        runningTime_all = time.time() - start_time_all   


    (_,bestSpans,bestNumSpans) = st.bestCase
    #image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
    image1 = st.applyManipulationToGetImage(bestSpans,bestNumSpans)
    (newClass,newConfident) = NN.predictWithImage(model,image1)
    newClassStr = dataBasics.LABELS(int(newClass))
    re = newClass != originalClass
                
    if re == True:     
        path0="%s/%s_%s_%s_%s_modified_into_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,phase,manipulationType, origClassStr,newClassStr,newConfident)
        dataBasics.save(-1,image1,path0)
        path0="%s/%s_diff.png"%(directory_pic_string,startIndexOfImage)
        dataBasics.save(-1,np.subtract(image,image1),path0)
        print("difference between images: %s"%(diffImage(image,image1)))

        st.showDecisionTree()
        pixelImages = st.analysePixels()
        advImages = st.analyseAdv.analyse()
        for (v,pixelImage) in pixelImages: 
            path0="%s/%s_useful_%s.png"%(directory_pic_string,startIndexOfImage,v)
            dataBasics.save(-1,pixelImage,path0)
            
        for i in range(len(advImages)): 
            (advnum,advimg) = advImages[i]
            (advClass,advConfident) = NN.predictWithImage(model,advimg)
            advClassStr = dataBasics.LABELS(int(advClass))
            path0="%s/%s_adv_%s_%s_%s.png"%(directory_pic_string,startIndexOfImage,i,advnum,advClassStr)
            dataBasics.save(-1,advimg,path0)
        
        print "number of adversarial examples found: %s"%(st.numAdv)
    
        eudist = euclideanDistance(st.image,image1)
        l1dist = l1Distance(st.image,image1)
        percent = diffPercent(st.image,image1)
        print "euclidean distance %s"%(eudist)
        print "L1 distance %s"%(l1dist)
        print "manipulated percentage distance %s"%(percent)
        print "class is changed into %s with confidence %s\n"%(newClassStr, newConfident)
        dc.addEuclideanDistance(eudist)
        dc.addl1Distance(l1dist)
        dc.addManipulationPercentage(percent)
    else: 
        print "failed to find an adversary image within prespecified bounded computational resource. "
                
    newXtrain,newYtrain = st.re_training.returnData()
    reTrain.addData(newXtrain,newYtrain)
    st.destructor()
                
    runningTime = time.time() - start_time   
    dc.addRunningTime(runningTime)

    return re
            
if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    