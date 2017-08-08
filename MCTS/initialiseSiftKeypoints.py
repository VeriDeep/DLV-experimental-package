#!/usr/bin/env python

"""
author: Xiaowei Huang

"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np, cv


from basics import *
from networkBasics import *
from configuration import * 

    
    
############################################################
#
#  preparation functions, selecting heuristics
#
################################################################


def initialiseSiftKeypoints(model,image,manipulated):

    image1 = copy.deepcopy(image)
    image1 = (image1*255).transpose(1, 2, 0)
    image1=image1.astype(np.uint8)
    kp, des = SIFT_Filtered(image1)
    
    print "%s keypoints are found. "%(len(kp))

    allRegions = []
    num = len(kp)/featureDims  # numOfFeatures
    i = 0
    while i < num :
        nextSpan = {}
        nextNumSpan = {}    
        ls = [] 
        for j in range(featureDims): 
            x = int(kp[i*featureDims + j].pt[0])
            y = int(kp[i*featureDims + j].pt[1])
            if len(image1.shape) == 2:  
                ls.append((x,y))
            else: 
                ls.append((0,x,y))
                ls.append((1,x,y))
                ls.append((2,x,y))
            
        for j in ls: 
            nextSpan[j] = span
            nextNumSpan[j] = numSpan
            
        oneRegion = (nextSpan,nextNumSpan,featureDims)
        allRegions.append(oneRegion)
        i += 1
    print "%s manipulations have been initialised."%(len(allRegions))
    return allRegions
    
    
def SIFT_Filtered(image, threshold=0.0):
    sift = cv2.SIFT() # cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
        
    #print kp[0], kp[0].response, kp[0].pt, kp[0].class_id, kp[0].octave, kp[0].size, len(des[0])

    #FILTER RESPONSES:
    
    actions = sorted(zip(kp,des), key=lambda x: x[0].response)

    return zip(*actions)

