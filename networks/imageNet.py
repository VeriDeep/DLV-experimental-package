import os, struct
from array import array as pyarray
from cvxopt.base import matrix
import numpy as np
from keras import backend as K
import PIL.Image


# input image dimensions
img_rows, img_cols = 224, 224
# the CIFAR10 images are RGB
img_channels = 3

# FIXME: need actual class names
def LABELS(index): 
    ls = labels()
    if len(ls) > 0: 
       return ls[index]
    else: return range(1000)[index] 
    
    
def next_index(index,index2):
    if index < len(labels()) - 1: index1 = index + 1
    else: index1 = 0
    if index1 != index2: return index1
    else: return next_index(index1,index2)
    
    
    

def labels():

    file = open('networks/imageNet/caffe_ilsvrc12/synset_words.txt', 'r')
    data = file.readlines()
    ls = []
    for line in data:
        words = line.split()
        ls.append(' '.join(words[1:]))
    return ls

def save(layer,image,filename):
    """
    """
    import cv2
    import copy
    
    image_cv = copy.deepcopy(image)
    
    if K.backend() == 'tensorflow':
        image_cv = image_cv.reshape(img_channels,img_rows,img_cols)


    if len(image_cv) == 3: 
    
        image_cv = image_cv.transpose(1, 2, 0)
    
        image_cv[:,:,0] += 103.939
        image_cv[:,:,1] += 116.779
        image_cv[:,:,2] += 123.68
        
    else: 
    
        image_cv = image_cv.transpose(1, 2, 0)      
        
        image_cv[:,:,0] += 103.939  
    
    #print(np.amax(image_cv),np.amin(image_cv))

    
    cv2.imwrite(filename, image_cv)


    # from matplotlib import pyplot
    # import matplotlib as mpl
    # fig = pyplot.figure()
    # ax = fig.add_subplot(1,1,1)
    # # image = image.reshape(3,32,32).transpose(1,2,0)
    # imgplot = ax.imshow(image.T, cmap=mpl.cm.Greys)
    # imgplot.set_interpolation('nearest')
    # ax.xaxis.set_ticks_position('top')
    # ax.yaxis.set_ticks_position('left')
    # pyplot.savefig(filename)


def show(image):
    """
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    #image = image.reshape(3,32,32).transpose(1,2,0)
    imgplot = ax.imshow(image.T, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
    
