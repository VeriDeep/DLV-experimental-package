#!/usr/bin/env python

"""
author: Xiaowei Huang

"""

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np, cv
from keras import backend as K
from scipy.stats import truncnorm

from basics import *
from networkBasics import *
from configuration import * 

    
    
############################################################
#
#  preparation functions, selecting heuristics
#
################################################################


def initialiseSiftKeypoints(model,image,manipulated):

    numOfPoints = 100

    image1 = copy.deepcopy(image)
    #if len(image1.shape) > 2: 
    #    image1 = (image1*255).transpose(1, 2, 0)
    image1=(image1*255).astype(np.uint8)
    image1 = cv2.resize(image1, (0,0), fx=2, fy=2) 
    #kp, des = SIFT_Filtered(image1)
    kp = SIFT_Filtered(image1,numOfPoints)
    for i in range(len(kp)):
    #     print kp[i].pt
        kp[i] = (kp[i][0]/2, kp[i][1]/2)
        #print "%s:%s"%(i,des[i])
         
    print "%s keypoints are found. "%(len(kp))
    
    allRegions = []
    num = len(kp)/featureDims  # numOfFeatures
    i = 0
    while i < num :
        nextSpan = {}
        nextNumSpan = {}    
        ls = [] 
        for j in range(featureDims): 
            x = int(kp[i*featureDims + j][0])
            y = int(kp[i*featureDims + j][1])
            if len(image1.shape) == 2:  
                ls.append((x,y))
            elif K.backend() == 'tensorflow': 
                ls.append((x,y,0))
                ls.append((x,y,1))
                ls.append((x,y,2))
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
    
    
def SIFT_Filtered(image, numOfPoints): #threshold=0.0):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = image.astype(np.uint8)
    #equ = cv2.equalizeHist(image)
    #image = np.hstack((image,equ))
    sift = cv2.SIFT() # cv2.SURF(400) #    cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    return  getPoints(image, kp, numOfPoints)
        
    #print kp[0], kp[0].response, kp[0].pt, kp[0].class_id, kp[0].octave, kp[0].size, len(des[0])
    '''
    if len(kp) == 0: 
        print("There is no keypont found in the image. \nPlease try approaches other than SIFT in processing this image. ")
        sys.exit()
    
    actions = sorted(zip(kp,des), key=lambda x: x[0].response)

    return zip(*actions)
    '''


def getPoints(image, kp, n): 
    values = getValues(image, kp)
    indices = [] #np.zeros(values.shape)
    for i in range(len(values)):
        for j in range(len(values[0])): 
            indices.append(i*len(values)+j) # [i][j] = (i,j)
    l =  np.random.choice(indices,n, p = values.flatten()  / np.sum(values))
    l2 = []
    for ind in l: 
        l2.append(getPixelLoc(ind,image))
    return list(set(l2))
    #print("value = %s"%(values))
    #print np.max(values)
    #print values.flatten().reshape(values.shape)


def getPixelLoc(ind, image):
    return (ind/len(image), ind%len(image))

def getValues(image, kp):

    import matplotlib.pyplot as plt
    import scipy
    from scipy.stats import multivariate_normal
    import scipy.stats
    import numpy.linalg
    
    values = np.zeros(image.shape[:2])
    for  k in kp: 
        a = np.array((k.pt[0],k.pt[1]))
        for i in range(len(values)): 
            for j in range(len(values[0])): 
                b = np.array((i,j))
                dist = numpy.linalg.norm(a - b)
                values[i][j] += scipy.stats.norm.pdf(dist, loc=0.0, scale=k.size)
    
    return values
                
                
    '''
        mean = [k.pt[0], k.pt[1]]
        cov = [0, k.size], [k.size, 0] 
        
        var = scipy.stats.multivariate_normal(mean,cov)
        print var.pdf([1,1])
        x, y = np.random.multivariate_normal(mean, cov, 5000).T
        
        
        numpy.linalg.norm()
        stats.norm.pdf(0, loc=0.0, scale=k.size)

        plt.plot(x, y, 'x')
        plt.axis('equal')
        plt.show()\
    '''
        
        
def CreateGaussianMixtureModel(image, kp, dimension=0):
        
    
    if(len(kp) == 0):
        myclip_a = 0
        myclip_b = 28
        my_mean = 10
        my_std = 3

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        x_range = np.linspace(myclip_a,myclip_b,28)
        sampled = truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std)
        
        return sampled, 28
    
    shape = image.shape
    observations = 0
    if (dimension == 0):
        observations = shape[1]
        index_to_use = 1
    else:
        observations = shape[0]
        index_to_use = 0
    distributions = []
    sum_of_weights = 0
    for k in kp:
        mu, sigma = int(round(k.pt[index_to_use])), k.size
        
        myclip_a = 0
        myclip_b = observations
        my_mean = mu
        my_std = sigma

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        x_range = np.linspace(myclip_a,myclip_b,observations)
        lamb = truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std/2)
        
        distributions.append(lamb)
        sum_of_weights += k.response
    gamma = []
    for k in kp:
        gamma.append(k.response/sum_of_weights)
    A = []
    sum_of_densitys = 0
    #print("observations: %s distributions: %s "%(observations, len(distributions)))
    # here we assume that the shape returns a size and not a highest index... may be problematic
    for i in range(observations-1):
        prob_of_observation = 0
        for d in distributions:
            prob_of_observation = prob_of_observation + d[i]
        A.append(prob_of_observation)
        sum_of_densitys = sum_of_densitys + prob_of_observation
    
    A = np.divide(A, np.sum(A))
    if(np.sum(A) != 1):
        val_to_add = 1 - np.sum(A)
        #NEED TO GET MAX INDEX HERE
        A[np.argmax(A)] = A[np.argmax(A)] + val_to_add
    return A, observations

