#!/usr/bin/env python

import numpy as np
import math
import ast
import copy
import random
import time
import stopit



from scipy import ndimage



def applyManipulation(image,span,numSpan):

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
    