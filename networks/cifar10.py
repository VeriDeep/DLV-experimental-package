import os, struct
from array import array as pyarray
from cvxopt.base import matrix
from keras import backend as K
import numpy as np

import PIL.Image

NUM_CLASSES = 10

def next_index(index,index2):
    if index < NUM_CLASSES - 1: index1 = index + 1
    else: index1 = 0
    if index1 != index2: return index1
    else: return next_index(index1,index2)
    
def LABELS(index):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']
    return labels[index]

def save(layer,image,filename):
    """
    """
    import cv2
    import copy

    image_cv = copy.deepcopy(image)
    if K.backend() == 'theano': 
        image_cv = image_cv.transpose(1, 2, 0)
    
    #print(np.amax(image),np.amin(image))

    params = list()
    params.append(cv2.cv.CV_IMWRITE_PNG_COMPRESSION)
    params.append(0)
    
    cv2.imwrite(filename, image_cv * 255.0, params)

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
    
