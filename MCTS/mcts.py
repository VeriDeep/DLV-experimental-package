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

from configuration import *

from inputManipulation import applyManipulation
from basics import mergeTwoDicts, diffPercent, euclideanDistance, l1Distance, numDiffs, diffImage

from decisionTree import decisionTree
from initialisePixelSets import initialisePixelSets
from initialiseSquares import initialiseSquares
from re_training import re_training
from analyseAdv import analyseAdv
from superPixels import superPixel_slic

class mcts:

    def __init__(self, model, autoencoder, image, activations, layer):
        self.image = image
        self.activations = activations
        self.model = model
        self.autoencoder = autoencoder
        self.manipulationType = "pixelSets"
        
        self.spans = {}
        self.numSpans = {}
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
        self.spans[-1] = {}
        self.numSpans[-1] = {}
        self.initialiseLeafNode(0,-1,[],[])
        
        # local actions
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

        # best case
        self.bestCase = (0,{},{})
        
        # number of adversarial exmaples
        self.numAdv = 0
        self.analyseAdv = analyseAdv(activations)
        
        # useless points
        self.usefulPixels = {}
        
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
            
    def setManipulationType(self,typeStr): 
        self.manipulationType = typeStr 
        
    def initialiseActions(self): 
        # initialise actions according to the type of manipulations
        if self.manipulationType == "pixelSets": 
            allChildren = initialisePixelSets(self.autoencoder,self.activations,[])
        elif self.manipulationType == "squares": 
            allChildren = initialiseSquares(self.autoencoder,self.activations,[])
        elif self.manipulationType == "slic": 
            allChildren = superPixel_slic(self.activations)
            
        for i in range(len(allChildren)): 
            self.actions[i] = allChildren[i] 
        print "%s actions have been initialised. "%(len(self.actions))
        # initialise decision tree
        self.decisionTree = decisionTree(self.model, self.actions,self.activations)
        
    def initialiseLeafNode(self,index,parentIndex,newSpans,newNumSpans):
        nprint("initialising a leaf node %s from the node %s"%(index,parentIndex))
        self.spans[index] = mergeTwoDicts(self.spans[parentIndex],newSpans)
        self.numSpans[index] = mergeTwoDicts(self.numSpans[parentIndex],newNumSpans)
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
        self.spans = {}
        self.numSpans = {}
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
        self.spans.pop(index,None)
        self.numSpans.pop(index,None)
        self.cost.pop(index,None) 
        self.parent.pop(index,None) 
        self.children.pop(index,None) 
        self.fullyExpanded.pop(index,None)
        self.numberOfVisited.pop(index,None)
            
    def bestChild(self,index):
        allValues = {}
        for childIndex in self.children[index]: 
            allValues[childIndex] = self.cost[childIndex]
        nprint("finding best children from %s"%(allValues))
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
        for (actionId, (span,numSpan,_)) in availableActions.iteritems() : #initialisePixelSets(self.model,self.image,list(set(self.spans[index].keys() + self.usefulPixels))): 
            self.indexToNow += 1
            self.indexToActionID[self.indexToNow] = actionId
            self.initialiseLeafNode(self.indexToNow,index,span,numSpan)
            self.children[index].append(self.indexToNow)
        self.fullyExpanded[index] = True
        self.usedActionsID = []
        return self.children[index]

    def backPropagation(self,index,value): 
        self.cost[index] += value
        self.numberOfVisited[index] += 1
        if self.parent[index] in self.parent : 
            nprint("start backPropagating the value %s from node %s, whose parent node is %s"%(value,index,self.parent[index]))
            self.backPropagation(self.parent[index],value)
        else: 
            nprint("backPropagating ends on node %s"%(index))
        
            
    def analysePixels(self):
        #self.usefulPixels = self.decisionTree.collectUsefulPixels()
        usefulPixels = [] 
        for index in self.usefulPixels: 
            usefulPixels.append(self.actions[index])
        #print("%s useful pixels = %s"%(len(self.usefulPixels),self.usefulPixels))
        values = self.usefulPixels.values()
        images = []
        for v in values: 
            pixels = [ dim for key, value in self.usefulPixels.iteritems() for dim in self.actions[key][0].keys() if value >= v ]
            ndim = self.image.ndim
            usefulimage = copy.deepcopy(self.image)
            span = {}
            numSpan = {}
            for p in pixels: 
                span[p] = 1
                numSpan[p] = 1
            #usefulimage = applyManipulation(usefulimage,span,numSpan)
            #'''
            if ndim == 2: 
                for x in range(len(usefulimage)): 
                    for y in range(len(usefulimage[0])): 
                        if (x,y) not in pixels: 
                            usefulimage[x][y] = 0
            elif ndim == 3:
                for x in range(len(usefulimage)): 
                    for y in range(len(usefulimage[0])): 
                        for z in range(len(usefulimage[0][0])):
                            if (x,y,z) not in pixels: 
                                usefulimage[x][y][z] = 0
            #'''
            images.append((v,usefulimage))
        return images
        
    def addUsefulPixels(self,dims):
        for dim in dims: 
            if dim in self.usefulPixels.keys(): 
                self.usefulPixels[dim] += 1
            else: 
                self.usefulPixels[dim] = 1
                
    def getUsefulPixels(self,accDims,d): 
        import operator
        sorted_accDims = sorted(accDims, key=operator.itemgetter(1), reverse=True)
        needed_accDims = sorted_accDims[:d-1]
        self.addUsefulPixels([x for (x,y) in needed_accDims])
            
    # start random sampling and return the Euclidean value as the value
    def sampling(self,index,availableActions):
        nprint("start sampling node %s"%(index))
        availableActions2 = copy.deepcopy(availableActions)
        availableActions2.pop(self.indexToActionID[index], None)
        sampleValues = []
        i = 0
        for i in range(MCTS_multi_samples): 
            (childTerminated, val) = self.sampleNext(self.spans[index],self.numSpans[index],0,availableActions2.keys(),[],[],2)
            sampleValues.append(val)
            #if childTerminated == True: break
            i += 1
        return (childTerminated, max(sampleValues))
    
    def sampleNext(self,spansPath,numSpansPath,depth,availableActionIDs,usedActionIDs,accDims,d): 
        activations1 = applyManipulation(self.activations,spansPath,numSpansPath)
        (newClass,newConfident) = self.predictWithActivations(activations1)
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

        #if termByDist == False and newConfident < 0.5 and depth <= 3: 
        #    termByDist = True

        if newClass != self.originalClass and newConfident > effectiveConfidenceWhenChanging:
            # and newClass == dataBasics.next_index(self.originalClass,self.originalClass): 
            nprint("sampling a path ends in a terminal node with depth %s... "%depth)
            
            (spansPath,numSpansPath) = self.scrutinizePath(spansPath,numSpansPath,newClass)
            
            self.decisionTree.addOnePath(dist,spansPath,numSpansPath)
            self.numAdv += 1
            self.analyseAdv.addAdv(activations1)
            self.getUsefulPixels(accDims,d)
                
            self.re_training.addDatum(activations1,self.originalClass)
            if self.bestCase[0] < dist: self.bestCase = (dist,spansPath,numSpansPath)
            return (depth == 0, dist)
        elif termByDist == True: 
            nprint("sampling a path ends by controlled search with depth %s ... "%depth)
            self.re_training.addDatum(activations1,self.originalClass)
            return (depth == 0, termValue)
        elif list(set(availableActionIDs)-set(usedActionIDs)) == []: 
            nprint("sampling a path ends with depth %s because no more actions can be taken ... "%depth)
            return (depth == 0, termValue)        
        else: 
            #print("continue sampling node ... ")
            #allChildren = initialisePixelSets(self.model,self.activations,spansPath.keys())
            randomActionIndex = random.choice(list(set(availableActionIDs)-set(usedActionIDs))) #random.randint(0, len(allChildren)-1)
            (span,numSpan,_) = self.actions[randomActionIndex]
            availableActionIDs.remove(randomActionIndex)
            usedActionIDs.append(randomActionIndex)
            newSpanPath = self.mergeSpan(spansPath,span)
            newNumSpanPath = self.mergeNumSpan(numSpansPath,numSpan)
            activations2 = applyManipulation(self.activations,newSpanPath,newNumSpanPath)
            (newClass2,newConfident2) = self.predictWithActivations(activations2)
            confGap2 = newConfident - newConfident2
            if newClass2 == newClass: 
                accDims.append((randomActionIndex,confGap2))
            else: accDims.append((randomActionIndex,1.0))

            return self.sampleNext(newSpanPath,newNumSpanPath,depth+1,availableActionIDs,usedActionIDs,accDims,d)
            
    def scrutinizePath(self,spanPath,numSpanPath,changedClass): 
        lastSpanPath = copy.deepcopy(spanPath)
        for key, (span,numSpan,_) in self.actions.iteritems(): 
            if set(span.keys()).issubset(set(spanPath.keys())): 
                tempSpanPath = copy.deepcopy(spanPath)
                tempNumSpanPath = copy.deepcopy(numSpanPath)
                for k in span.keys(): 
                    tempSpanPath.pop(k)
                    tempNumSpanPath.pop(k) 
                activations1 = applyManipulation(self.activations,tempSpanPath,tempNumSpanPath)
                (newClass,newConfident) = self.predictWithActivations(activations1)
                #if changedClass == newClass: 
                if newClass != self.originalClass and newConfident > effectiveConfidenceWhenChanging:
                    for k in span.keys(): 
                        spanPath.pop(k)
                        numSpanPath.pop(k)
        if len(lastSpanPath.keys()) != len(spanPath.keys()): 
            return self.scrutinizePath(spanPath,numSpanPath,changedClass)
        else: 
            return (spanPath,numSpanPath)
            
    def terminalNode(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        (newClass,_) = self.predictWithActivations(activations1)
        return newClass != self.originalClass 
        
    def terminatedByControlledSearch(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        (distMethod,distVal) = controlledSearch
        if distMethod == "euclidean": 
            dist = euclideanDistance(activations1,self.activations) 
        elif distMethod == "L1": 
            dist = l1Distance(activations1,self.activations) 
        elif distMethod == "Percentage": 
            dist = diffPercent(activations1,self.activations)
        elif distMethod == "NumDiffs": 
            dist = diffPercent(activations1,self.activations)
        nprint("terminated by controlled search")
        return dist > distVal 
        
    def applyManipulationToGetImage(self,spans,numSpans):
        activations1 = applyManipulation(self.activations,spans,numSpans)
        if self.layer > -1: 
            return np.squeeze(self.autoencoder.predict(np.expand_dims(activations1,axis=0)))
        else: 
            return activations1
        
    def euclideanDist(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return euclideanDistance(self.activations,activations1)
        
    def l1Dist(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return l1Distance(self.activations,activations1)
        
    def diffImage(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return diffImage(self.activations,activations1)
        
    def diffPercent(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return diffPercent(self.activations,activations1)

    def mergeSpan(self,spansPath,span): 
        return mergeTwoDicts(spansPath, span)
        
    def mergeNumSpan(self,numSpansPath,numSpan):
        return mergeTwoDicts(numSpansPath, numSpan)
        
    def showDecisionTree(self):
        self.decisionTree.show()
        self.decisionTree.outputTree()
    
        