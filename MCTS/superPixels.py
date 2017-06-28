#!/usr/bin/env python

"""
author: Xiaowei Huang

"""

import matplotlib.pyplot as plt
import numpy as np
import math

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import copy

from configuration import *


def superPixel_slic(img):
    image = copy.deepcopy(img)
    image = image.transpose(1, 2, 0)
    
     
    
    if dataset == "imageNet": 
        image[:,:,0] += 103.939
        image[:,:,1] += 116.779
        image[:,:,2] += 123.68
        maxPixel = np.max(image)
        image = image/maxPixel
            
    #segments_fz = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    segments_slic = slic(image, n_segments=int(image.size/(featureDims*5)), sigma=1)
    #segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    #gradient = sobel(rgb2gray(img))
    #segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    
    #print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
    print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
    
    #print segments_slic
    #print segments_fz
    
    #fig = plt.figure("Superpixels -- 259 segments")
    #ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(mark_boundaries(image, segments_slic))
    #plt.axis("off")
    
    #plt.tight_layout()
    #plt.show()
        
    image1 = mark_boundaries(image, segments_slic)
    
    if dataset == "imageNet": 
        image *= maxPixel
        image[:,:,0] -= 103.939
        image[:,:,1] -= 116.779
        image[:,:,2] -= 123.68
        image1 *= maxPixel
        image1[:,:,0] -= 103.939
        image1[:,:,1] -= 116.779
        image1[:,:,2] -= 123.68
    
    image1 = image1.transpose(2, 0, 1)
    path0="%s/%s_slic.png"%(directory_pic_string,startIndexOfImage)
    dataBasics.save(-1,image1, path0)
    

    
    image = image.transpose(2, 0, 1)
    
    act = []
    for n in np.unique(segments_slic): 
        span = {}
        numSpan = {}
        for x in range(len(segments_slic)): 
            for y in range(len(segments_slic[0])): 
                if segments_slic[x][y] == n: 
                    span[(0,x,y)] = 1.0
                    span[(1,x,y)] = 1.0
                    span[(2,x,y)] = 1.0
                    numSpan[(0,x,y)] = 1.0
                    numSpan[(1,x,y)] = 1.0
                    numSpan[(2,x,y)] = 1.0
        act.append((span,numSpan,1))
    
    return act
