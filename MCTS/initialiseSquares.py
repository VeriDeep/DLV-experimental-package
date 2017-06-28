#!/usr/bin/env python

"""
author: Xiaowei Huang

"""

import numpy as np
import copy
from scipy import ndimage
from random import randint, random
from math import sqrt

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

def initialiseSquares(model,image,manipulated):
    allRegions = []
    num = image.size/featureDims  # numOfFeatures
    if len(image.shape) == 2: 
        imageLevel = 1
        imageSize = len(image)
    else: 
        imageLevel = 3
        imageSize = len(image[0])
    squareSize = findSquareSize(featureDims, imageSize)
    print "working with squares of size %s*%s"%(squareSize,squareSize)
    ps = findUpperLeftPoints(squareSize, imageSize)
    #print ps
    for i in ps: 
        for j in ps: 
            oneRegion = getOneRegion(i,j,squareSize,imageLevel)
            allRegions.append(oneRegion)
    
    print "%s manipulations have been initialised."%(len(allRegions))
    return allRegions


############################################################
#
#  auxiliary functions
#
################################################################   

def findSquareSize(num, imageSize):
    i = 1
    while i <= imageSize: 
        if i**2 < num and (i+1)**2 >= num: 
            return i+1
        i += 1
    
def findUpperLeftPoints(squareSize, imageSize): 
    i = 0
    ls = []
    while i+squareSize <= imageSize: 
        ls.append(i)
        i += squareSize
    return ls
    
def getOneRegion(i,j,squareSize,imageLevel): 
    span = {}
    numSpan = {}
    for k in range(squareSize): 
        for l in range(squareSize):
            if imageLevel == 1: 
                span[(i+k,j+l)] = 1.0
                numSpan[(i+k,j+l)] = 1            
            else: 
                for m in range(imageLevel): 
                    span[(m,i+k,j+l)] = 1.0
                    numSpan[(m,i+k,j+l)] = 1
    return (span,numSpan,squareSize**2)