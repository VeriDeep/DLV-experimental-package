#!/usr/bin/env python

"""
A data structure for organising search

author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy
import sys
import operator
import random
import math
import cv2

from configuration import *

from basics import mergeTwoDicts, diffPercent, euclideanDistance, l1Distance, numDiffs, diffImage

from decisionTree import decisionTree
from re_training import re_training



class mcts_geo:

    def __init__(self, model, autoencoder, image, activations, layer):
        self.image = image
        self.activations = activations
        self.model = model
        self.autoencoder = autoencoder
        
        self.actSeq = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}
        
        self.indexToNow = 0
        # current root node
        self.rootIndex = 0
     
        # current layer
        self.layer = layer
        
        # initialise root node
        self.actSeq[-1] = []
        self.initialiseLeafNode(0,-1,[])
        
        # local actions
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

        # best case
        self.bestCase = (0,{},{})
        
        (self.originalClass,self.originalConfident) = self.predictWithActivations(self.activations)
        
        self.decisionTree = 0
        self.re_training = re_training(model,self.image.shape)

    def predictWithActivations(self,activations):
        if self.layer > -1: 
            output = np.squeeze(self.autoencoder.predict(np.expand_dims(activations,axis=0)))
            return NN.predictWithImage(self.model,output)
        else: 
            return NN.predictWithImage(self.model,activations)
            
    def visualizationMCTS(self):
        for k in range(len(self.activations)): 
            activations1 = copy.deepcopy(self.activations)
            # use a random node to replace the feature node
            emptyNode = np.zeros_like(self.activations[0])
            activations1[k] = emptyNode
            output = np.squeeze(self.autoencoder.predict(np.expand_dims(activations1,axis=0)))
            path0="%s/%s_autoencoder_%s.png"%(directory_pic_string,startIndexOfImage,k)
            dataBasics.save(-1,output, path0)
        
    def initialiseActions(self): 
    
        translationAction = [("Translation", 0.1), ("Translation", 0.2)]
        guassianFilteringAction = [("guassianFiltering", 5, 0), ("guassianFiltering", 3, 1)]
        twoDConvAction = [("twoDConv", 3, float(1/25)), ("twoDConv", 5, float(1/10))]
        thresholdingGeoAction = [("thresholdingGeoAction", 127, 255), ("thresholdingGeoAction", 200, 255)]
        
        allChildren = translationAction + guassianFilteringAction + twoDConvAction + thresholdingGeoAction
        for i in range(len(allChildren)): 
            self.actions[i] = allChildren[i] 
        print "%s actions have been initialised. "%(len(self.actions))
        # initialise decision tree
        self.decisionTree = decisionTree(self.actions)
        
    def applyGeoManipulation(self, activations, actionSequence):     
        if actionSequence == []: 
            return activations
        
        action = actionSequence[0]
        if action[0] == "Translation": 
            activations2 = self.translationGeoAction(activations,action[1])
        elif action[0] == "guassianFiltering": 
            activations2 = self.guassianFiltering(activations,action[1],action[2])
        elif action[0] == "twoDConv": 
            activations2 = self.twoDConv(activations,action[1],action[2])
        elif action[0] == "thresholdingGeoAction": 
            activations2 = self.thresholdingGeoAction(activations,action[1],action[1])
        return self.applyGeoManipulation(activations2, actionSequence[1:])
        
    def initialiseLeafNode(self,index,parentIndex,action):
        print("initialising a leaf node %s from the node %s"%(index,parentIndex))
        self.actSeq[index] = self.actSeq[parentIndex] + action
        self.cost[index] = 0
        self.parent[index] = parentIndex 
        self.children[index] = []
        self.fullyExpanded[index] = False
        self.numberOfVisited[index] = 0    

    def destructor(self): 
        self.image = 0
        self.activations = 0
        self.model = 0
        self.autoencoder = 0
        self.actSeq = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}
        
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}
        
    # move one step forward
    # it means that we need to remove children other than the new root
    def makeOneMove(self,newRootIndex): 
        print "making a move into the new root %s, whose value is %s and visited number is %s"%(newRootIndex,self.cost[newRootIndex],self.numberOfVisited[newRootIndex])
        self.removeChildren(self.rootIndex,[newRootIndex])
        self.rootIndex = newRootIndex
    
    def removeChildren(self,index,indicesToAvoid): 
        if self.fullyExpanded[index] == True: 
            for childIndex in self.children[index]: 
                if childIndex not in indicesToAvoid: self.removeChildren(childIndex,[])
        self.actSeq.pop(index,None)
        self.cost.pop(index,None) 
        self.parent.pop(index,None) 
        self.children.pop(index,None) 
        self.fullyExpanded.pop(index,None)
        self.numberOfVisited.pop(index,None)
            
    def bestChild(self,index):
        allValues = {}
        for childIndex in self.children[index]: 
            allValues[childIndex] = self.cost[childIndex]
        print("finding best children from %s"%(allValues))
        return max(allValues.iteritems(), key=operator.itemgetter(1))[0]
        
    def treeTraversal(self,index):
        if self.fullyExpanded[index] == True: 
            nprint("tree traversal on node %s"%(index))
            allValues = {}
            for childIndex in self.children[index]: 
                allValues[childIndex] = (self.cost[childIndex] / float(self.numberOfVisited[childIndex])) + explorationRate * math.sqrt(math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex]))
            nextIndex = max(allValues.iteritems(), key=operator.itemgetter(1))[0]
            self.usedActionsID.append(self.indexToActionID[nextIndex])
            return self.treeTraversal(nextIndex)
        else: 
            nprint("tree traversal terminated on node %s"%(index))
            availableActions = copy.deepcopy(self.actions)
            for i in self.usedActionsID: 
                availableActions.pop(i, None)
            return (index,availableActions)
        
    def initialiseExplorationNode(self,index,availableActions):
        nprint("expanding %s"%(index))
        for (actionId, action) in availableActions.iteritems() : 
            self.indexToNow += 1
            self.indexToActionID[self.indexToNow] = actionId
            self.initialiseLeafNode(self.indexToNow,index,[action])
            self.children[index].append(self.indexToNow)
        self.fullyExpanded[index] = True
        self.usedActionsID = []
        return self.children[index]

    def backPropagation(self,index,value): 
        self.cost[index] += value
        self.numberOfVisited[index] += 1
        if self.parent[index] in self.parent : 
            print("start backPropagating the value %s from node %s, whose parent node is %s"%(value,index,self.parent[index]))
            self.backPropagation(self.parent[index],value)
        else: 
            print("backPropagating ends on node %s"%(index))
            
    # start random sampling and return the eclidean value as the value
    def sampling(self,index,availableActions):
        nprint("start sampling node %s"%(index))
        availableActions2 = copy.deepcopy(availableActions)
        availableActions2.pop(self.indexToActionID[index], None)
        sampleValues = []
        i = 0
        for i in range(MCTS_multi_samples): 
            #allChildren = copy.deepcopy(self.actions)
            (childTerminated, val) = self.sampleNext(self.actSeq[index],0,availableActions2.keys(),[])
            sampleValues.append(val)
            if childTerminated == True: break
            i += 1
        return (childTerminated, max(sampleValues))
        #return self.sampleNext(self.spans[index],self.numSpans[index])
        #allChildren = initialisePixelSets(model,self.image,self.spans[index].keys()) 
    
    def sampleNext(self,actionSequence,depth,availableActionIDs,usedActionIDs): 
        #print spansPath.keys()
        activations1 = self.applyGeoManipulation(self.activations,actionSequence)
        (newClass,newConfident) = self.predictWithActivations(activations1)
        #print euclideanDistance(self.activations,activations1), newConfident, newClass
        (distMethod,distVal) = controlledSearch
        if distMethod == "euclidean": 
            dist = 1 - euclideanDistance(activations1,self.activations) 
            termValue = 0.0
            termByDist = dist < 1 - distVal
        elif distMethod == "L1": 
            dist = 1 - l1Distance(activations1,self.activations) 
            termValue = 0.0
            termByDist = dist < 1 - distVal
        elif distMethod == "Percentage": 
            dist = 1 - diffPercent(activations1,self.activations)
            termValue = 0.0
            termByDist = dist < 1 - distVal
        elif distMethod == "NumDiffs": 
            dist = self.activations.size - diffPercent(activations1,self.activations) * self.activations.size
            termValue = 0.0
            termByDist = dist < self.activations.size - distVal

        if newClass != self.originalClass: 
            print("sampling a path ends in a terminal node with depth %s... "%depth)
            self.decisionTree.addOnePath(dist,actionSequence)
            self.re_training.addDatum(activations1,self.originalClass)
            if self.bestCase[0] < dist: self.bestCase = (dist,spansPath,numSpansPath)
            return (depth == 0, dist)
        elif termByDist == True: 
            print("sampling a path ends by controlled search with depth %s ... "%depth)
            return (depth == 0, termValue)
        else: 
            #print("continue sampling node ... ")
            #allChildren = initialisePixelSets(self.model,self.activations,spansPath.keys())

            randomActionIndex = random.choice(list(set(availableActionIDs)-set(usedActionIDs))) #random.randint(0, len(allChildren)-1)
            action = self.actions[randomActionIndex]
            availableActionIDs.remove(randomActionIndex)
            usedActionIDs.append(randomActionIndex)

            newActSeq = actionSequence + [actioin]
            return self.sampleNext(newActSeq,depth+1,availableActionIDs,usedActionIDs)
            
    def terminalNode(self,index): 
        activations1 = self.applyGeoManipulation(self.activations,self.actSeq[index])
        (newClass,_) = self.predictWithActivations(activations1)
        return newClass != self.originalClass 
        
    def terminatedByControlledSearch(self,index): 
        activations1 = self.applyGeoManipulation(self.activations,self.actSeq[index])
        (distMethod,distVal) = controlledSearch
        if distMethod == "euclidean": 
            dist = euclideanDistance(activations1,self.activations) 
        elif distMethod == "L1": 
            dist = l1Distance(activations1,self.activations) 
        elif distMethod == "Percentage": 
            dist = diffPercent(activations1,self.activations)
        elif distMethod == "NumDiffs": 
            dist = diffPercent(activations1,self.activations)
        print("terminated by controlled search")
        return dist > distVal 
        
    def applyManipulationToGetImage(self,actionSequence):
        activations1 = self.applyGeoManipulation(self.activations,actionSequence)
        if self.layer > -1: 
            return np.squeeze(self.autoencoder.predict(np.expand_dims(activations1,axis=0)))
        else: 
            return activations1
            
    def euclideanDist(self,index): 
        activations1 = self.applyGeoManipulation(self.activations,self.actSeq[index])
        return euclideanDistance(self.activations,activations1)
        
    def l1Dist(self,index): 
        activations1 = self.applyGeoManipulation(self.activations,self.actSeq[index])
        return l1Distance(self.activations,activations1)
        
    def diffImage(self,index): 
        activations1 = self.applyGeoManipulation(self.activations,self.actSeq[index])
        return diffImage(self.activations,activations1)
        
    def diffPercent(self,index): 
        activations1 = self.applyGeoManipulation(self.activations,self.actSeq[index])
        return diffPercent(self.activations,activations1)

                
    def showDecisionTree(self):
        self.decisionTree.show()
    
    # Translation
    def translationGeoAction(self,img,epsilon):
    
        (chs,rows,cols) = img.shape 
        M = np.float32([[1, 0, int(cols*epsilon)], [0, 1, int(rows*epsilon)]])
        img_tras = cv2.warpAffine(img, M, dsize=(cols, rows))
        #cv2.imshow('Translation1', img_tras)
        return img_tras
        
    # Gaussian Filtering
    def guassianFiltering(self,img, filterSize, standardDeviation):
        filterSize = 5
        standardDeviation = 0
        img_blur = cv2.GaussianBlur(img, (filterSize, filterSize), standardDeviation)
        #cv2.imshow('Gaussian Filtering', img_blur)
        return img_blur
        
    # 2D Convolution
    def twoDConv(self,img, filterSize, epsilon):
        filterSize = 5
        epsilon = 1/ 25
        kernel = np.ones((filterSize, filterSize), np.float32) * epsilon
        img_filter = cv2.filter2D(img, -1, kernel)
        #cv2.imshow('2D Convolution', img_filter)
        return img_filter
        
    # Thresholding
    def thresholdingGeoAction(self,thresholdValue,maxVal):
        thresholdValue = 127
        maxVal = 255
        ret, thresh1 = cv2.threshold(img, thresholdValue, maxVal, cv2.THRESH_BINARY)
        #cv2.imshow('Thresholding', thresh1)
        return thresh1
        
    
        