#!/usr/bin/env python

"""
author: Xiaowei Huang

"""

import numpy as np
import copy
from scipy import ndimage
from random import randint, random

import mnist
import cifar10
import imageNet
import mnist_network as NN_mnist
import cifar10_network as NN_cifar10
import imageNet_network as NN_imageNet


from basics import *
from networkBasics import *
from configuration import * 

    
    
############################################################
#
#  preparation functions, selecting heuristics
#
################################################################


randomRate = 0.0

def initialisePixelSets(model,image,manipulated):
    allRegions = []
    num = image.size/featureDims  # numOfFeatures
    newManipulated1 = []
    newManipulated2 = manipulated
    while num > 0 : 
        oneRegion = initialisePixelSetActivation(model,newManipulated2,image)
        allRegions.append(oneRegion)
        newManipulated1 = copy.deepcopy(newManipulated2)
        newManipulated2 = list(set(newManipulated2 + oneRegion[0].keys()))
        if newManipulated1 == newManipulated2: break
        num -= 1
    print "%s manipulations have been initialised."%(len(allRegions))
    return allRegions


############################################################
#
#  initialise a region for the input 
#
################################################################   

 
def initialisePixelSetActivation(model,manipulated,image): 

    nextSpan = {}
    nextNumSpan = {}
    if len(image.shape) == 2: 
        # decide how many elements in the input will be considered
        if image.size < featureDims : 
            numDimsToMani = image.size 
        else: numDimsToMani = featureDims
        # get those elements with maximal/minimum values
        randnum = random()
        if randnum > randomRate : 
            ls = getTop2DActivation(image,manipulated,[],numDimsToMani,-1)
        else:  
            ls = getRandom2DActivation(image,manipulated,[],numDimsToMani,-1)
                
    elif len(image.shape) == 3:
        # decide how many elements in the input will be considered
        if image.size < featureDims : 
            numDimsToMani = image.size
        else: numDimsToMani = featureDims
        # get those elements with maximal/minimum values
        randnum = random()
        if randnum > randomRate : 
            ls = getTop3DActivation(image,manipulated,[],numDimsToMani,-1)
        else: 
            ls = getRandom3DActivation(image,manipulated,[],numDimsToMani,-1)

    for i in ls: 
        nextSpan[i] = span
        nextNumSpan[i] = numSpan
            
    return (nextSpan,nextNumSpan,numDimsToMani)
    
    
############################################################
#
#  auxiliary functions
#
################################################################
    
# This function only suitable for the input as a list, not a multi-dimensional array

def getTopActivation(image,manipulated,layerToConsider,numDimsToMani): 

    avoid = True # repeatedManipulation == "disallowed"
    
    #avg = np.sum(image)/float(len(image))
    #nimage = list(map(lambda x: abs(avg - x),image))
    avg = np.average(image)
    nimage = np.absolute(image - avg) 

    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        if len(topImage) < numDimsToMani and ((not avoid) or (i not in manipulated)): 
            topImage[i] = nimage[i]
        else: 
            bl = False
            for k, v in topImage.iteritems():
                if v < nimage[i] and not (k in toBeDeleted) and ((not avoid) or (i not in manipulated)): 
                        toBeDeleted.append(k)
                        bl = True
                        break
            if bl == True: 
                topImage[i] = nimage[i]
    for k in toBeDeleted: 
        del topImage[k]
    return topImage.keys()
    
def getRandom2DActivation(image,manipulated,ps,numDimsToMani,layerToConsider): 

    avoid = True # repeatedManipulation == "disallowed"
            
    oldmanipulated = copy.deepcopy(manipulated)
    i = copy.deepcopy(numDimsToMani)
    while i > 0 and (len(oldmanipulated) + i) <= image.size: 
        
        randnum = randint(1,image.size) - 1
        fst = randnum / image.shape[1]
        snd = randnum % image.shape[1]
        
        if (fst,snd) not in manipulated: 
            oldmanipulated.append((fst,snd))
            i -= 1
        
    return list(set(oldmanipulated) - set(manipulated))
    
def getTop2DActivation(image,manipulated,ps,numDimsToMani,layerToConsider): 

    avoid = True #repeatedManipulation == "disallowed"
            
    avg = np.average(image)
    nimage = np.absolute(image - avg) 
            
    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if len(topImage) < numDimsToMani and ((not avoid) or ((i,j) not in manipulated)): 
                topImage[(i,j)] = nimage[i][j]
            else: 
                bl = False 
                for (k1,k2), v in topImage.iteritems():
                    if v < nimage[i][j] and not ((k1,k2) in toBeDeleted) and ((not avoid) or ((i,j) not in manipulated)):  
                        toBeDeleted.append((k1,k2))
                        bl = True
                        break
                if bl == True: 
                    topImage[(i,j)] = nimage[i][j]
    for (k1,k2) in toBeDeleted: 
        del topImage[(k1,k2)]
        
    return topImage.keys()

# ps are indices of the previous layer

def getRandom3DActivation(image,manipulated,ps,numDimsToMani,layerToConsider): 

    #print numDimsToMani, ps
    avoid = True # repeatedManipulation == "disallowed"

    #avg = np.sum(image)/float(len(image)*len(image[0]*len(image[0][0])))
    #nimage = copy.deepcopy(image)
    #for i in range(len(image)): 
    #    for j in range(len(image[0])):
    #        for k in range(len(image[0][0])):
    #            nimage[i][j][k] = abs(avg - image[i][j][k])
    
    # find a two-dimensional with maximal variance
    ind = randint(1,image.shape[0])
    
    if len(ps) > 0: 
        if len(ps[0]) == 3: 
            (p1,p2,p3) = zip(*ps)
            ps = zip(p2,p3)
    
    pointsToConsider = []
    for i in range(numDimsToMani): 
        if i <= len(ps) - 1: 
            (x,y) = ps[i] 
            nps = [ (x-x1,y-y1) for x1 in range(filterSize) for y1 in range(filterSize) if x-x1 >= 0 and y-y1 >=0 ]
            pointsToConsider = pointsToConsider + nps
    pointsToConsider = list(set(pointsToConsider))
    
    ks = getRandom2DActivation(image[maxVarInd],manipulated,ps,numDimsToMani,layerToConsider,pointsToConsider)
    
    #print ks, pointsToConsider
    
    return map(lambda (x,y): (ind,x,y),ks)


def getTop3DActivation(image,manipulated,ps,numDimsToMani,layerToConsider): 

    avoid = True #repeatedManipulation == "disallowed"

    #avg = np.sum(image)/float(len(image)*len(image[0]*len(image[0][0])))
    #nimage = copy.deepcopy(image)
    #for i in range(len(image)): 
    #    for j in range(len(image[0])):
    #        for k in range(len(image[0][0])):
    #            nimage[i][j][k] = abs(avg - image[i][j][k])
    
    avg = np.average(image)
    nimage = np.absolute(image - avg) 
                
    # do not care about the first dimension
    # only care about individual convolutional node
    if len(ps) > 0: 
        if len(ps[0]) == 3: 
            (p1,p2,p3) = zip(*ps)
            ps = zip(p2,p3)
    ks = []
    pointsToConsider = []
    for i in range(numDimsToMani): 
        if i <= len(ps) - 1: 
            (x,y) = ps[i] 
            nps = [ (x-x1,y-y1) for x1 in range(filterSize) for y1 in range(filterSize) if x-x1 >= 0 and y-y1 >=0 ]
            pointsToConsider = pointsToConsider + nps
            ks = ks + findFromArea3D(image,manipulated,avoid,nimage,nps,1,ks)
        else: 
            ks = ks + findFromArea3D(image,manipulated,avoid,nimage,pointsToConsider,1,ks)
    return ks
    
    
def findFromArea3D(image,manipulated,avoid,nimage,ps,numDimsToMani,ks):
    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(image[0][0])):
                if len(topImage) < numDimsToMani and ((j,k) in ps or len(ps) == 0) and (i,j,k) not in ks: 
                    topImage[(i,j,k)] = nimage[i][j][k]
                elif ((j,k) in ps or len(ps) == 0) and (i,j,k) not in ks: 
                    bl = False 
                    for (k1,k2,k3), v in topImage.iteritems():
                        if v < nimage[i][j][k] and not ((k1,k2,k3) in toBeDeleted) and ((not avoid) or ((i,j,k) not in manipulated)):  
                            toBeDeleted.append((k1,k2,k3))
                            bl = True
                            break
                    if bl == True: 
                        topImage[(i,j,k)] = nimage[i][j][k]
    for (k1,k2,k3) in toBeDeleted: 
        del topImage[(k1,k2,k3)]
    return topImage.keys()
