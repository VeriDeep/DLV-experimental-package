## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time

from configuration import *
from basics import *

#from setup_cifar import CIFAR, CIFARModel
#from setup_mnist import MNIST, MNISTModel
#from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    for i in range(len(img)): 
        if math.isnan(img[i]): img[i] = 0
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets
    
def getTargets(label):
    label2 = []
    label1 = copy.deepcopy(label)
    for l in label1: 
        l = l.tolist()
        if l.index(max(l)) < NN.nb_classes - 1: 
            l = [0] + l 
        else: 
            l = [1] + l 
        label2.append(l[:NN.nb_classes])
    return np.array(label2)


def test_attack(sess,model,data,label): 
        
    attack = CarliniL2(sess, model, len(data[0][1]), NN.img_channels, NN.nb_classes, batch_size=len(data), max_iterations=1000, confidence=0)
    
    #inputs, targets = generate_data(data, samples=1, targeted=True,
    #                                start=0, inception=False)
    timestart = time.time()
    #adv = attack.attack(inputs, targets)
    adv = attack.attack(data, getTargets(label))
    print(adv.shape)
    timeend = time.time()
        
    print("Took",timeend-timestart,"seconds to run",len(data),"samples.")

    for i in range(len(adv)):
        print("Valid:")
        show(data[i])
        print("Adversarial:")
        show(adv[i])
            
        # keep information for the original image
        (newClass,newConfident) = NN.predictWithImage(model,adv[i]+0.5)
        newClassStr = dataBasics.LABELS(int(newClass))
        path0="%s/%s_converted_into_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage+i,newClassStr,newConfident)
        dataBasics.save(-1,np.squeeze(adv[i]), path0)
            
        print("Classification:", model.predict(adv[i:i+1]+0.5))
        print("Total distortion:", np.sum((adv[i]-data[i])**2)**.5)        
        print("L1 distance:", l1Distance(data[i],adv[i]))
