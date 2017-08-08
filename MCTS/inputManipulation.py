#!/usr/bin/env python

import numpy as np
import math
import ast
import copy
import random
import time
import stopit
from keras import backend as K


from scipy import ndimage



def applyManipulation(image,span,numSpan):

    # toggle manipulation
    image1 = copy.deepcopy(image)
    maxVal = np.max(image1)
    minVal = np.min(image1)

    for elt in span.keys(): 
        if len(elt) == 2: 
            (fst,snd) = elt 
            if maxVal - image[fst][snd] < image[fst][snd] : image1[fst][snd] -= numSpan[elt] * span[elt]
            else: image1[fst][snd] += numSpan[elt] * span[elt]
            if image1[fst][snd] < minVal: image1[fst][snd] = minVal
            elif image1[fst][snd] > maxVal: image1[fst][snd] = maxVal
        elif len(elt) == 3: 
            (fst,snd,thd) = elt 
            if maxVal - image[fst][snd][thd] < image[fst][snd][thd] : image1[fst][snd][thd] -= numSpan[elt] * span[elt]
            else: image1[fst][snd][thd] += numSpan[elt] * span[elt]
            if image1[fst][snd][thd] < minVal: image1[fst][snd][thd] = minVal
            elif image1[fst][snd][thd] > maxVal: image1[fst][snd][thd] = maxVal
    return image1
    

    '''
    # mean filter manipulation
    image1 = copy.deepcopy(image)
    size = 3
    for point in span.keys(): 
        if len(point) == 2: 
            (x,y) = point
            total = 0 
            for i in range(-1* size/2, size): 
                for j in range(-1* size/2, size): 
                    total += image[x+i][y+j]
            image1[x][y] = total/float(size**2)
        else: 
            (x,y,z) = point 
            if K.backend() == 'theano': 
                for i in range(0,3): 
                    total = 0 
                    for j in range(-1* size/2, size): 
                        for k in range(-1* size/2, size): 
                            total += image[i][y+j][z+k]
                    image1[i][y][z] = total/float(size**2)
            else: 
                for i in range(0,3): 
                    total = 0 
                    for j in range(-1* size/2, size): 
                        for k in range(-1* size/2, size): 
                            total += image[x+j][y+k][i]
                    image1[x][y][i] = total/float(size**2)
    return image1
    '''
    
def assignManipulationSimple(image,span,numSpan):

    image1 = copy.deepcopy(image)

    for elt in span.keys(): 
        if len(elt) == 2: 
            (fst,snd) = elt 
            image1[fst][snd] += numSpan[elt] * span[elt]
            if image1[fst][snd] < 0: image1[fst][snd] = 0
            elif image1[fst][snd] > 1: image1[fst][snd] = 1
        elif len(elt) == 3: 
            (fst,snd,thd) = elt 
            image1[fst][snd][thd] += numSpan[elt] * span[elt]
            if image1[fst][snd][thd] < 0: image1[fst][snd][thd] = 0
            elif image1[fst][snd][thd] > 1: image1[fst][snd][thd] = 1
    return image1
    
    
def assignManipulation(image,span,numSpan):

    image1 = copy.deepcopy(image)

    for elt in span.keys(): 
        if len(elt) == 2: 
            (fst,snd) = elt 
            image1[fst][snd] = numSpan[elt] * span[elt]
            if image1[fst][snd] < 0: image1[fst][snd] = 0
            elif image1[fst][snd] > 1: image1[fst][snd] = 1
        elif len(elt) == 3: 
            (fst,snd,thd) = elt 
            image1[fst][snd][thd] = numSpan[elt] * span[elt]
            if image1[fst][snd][thd] < 0: image1[fst][snd][thd] = 0
            elif image1[fst][snd][thd] > 1: image1[fst][snd][thd] = 1
    return image1
    