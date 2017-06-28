#!/usr/bin/env python


import sys
from PIL import Image
import numpy as np
import imp
from basics import *
from networkBasics import *
from configuration import * 
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib as mpl
from inputManipulation import applyManipulation



class decisionTree:

    def __init__(self, model, actions, image):
        self.model = model
        self.image = image
        self.imagesOnTree = {}
        self.actions = actions
        self.indexUpToNow = 1
        self.tree = {}
        self.tree[0] = []

    def addOnePath(self,dist,spansPath,numSpansPath): 
        alldims = spansPath.keys()
        currentNode = 0
        while alldims != []: 
            (act, alldims) = self.getAction(alldims,currentNode)
            nodeExists = False
            for (act2,nextNode) in self.tree[currentNode]: 
                if act == act2: 
                    currentNode = nextNode
                    nodeExists = True
                    break
            if nodeExists == False: # and (currentNode == 0 or self.tree[currentNode] != []) : 
                self.indexUpToNow += 1
                self.tree[self.indexUpToNow] = []
                self.tree[currentNode].append((act,self.indexUpToNow))
                currentNode = self.indexUpToNow

        #self.removeEmptyLeaves()
        
    def collectUsefulPixels(self):
        acts = []
        for key, value in self.tree.iteritems(): 
            for (key2,value2) in value: 
                if self.tree[value2] == []: 
                    acts.append(key2) 
        pixels = []
        for act in acts: 
            pixels += self.actions[act][0].keys()
        return pixels

    def getAction(self,dims,currentNode):
        existingDims = []
        if self.tree[currentNode] != []:
            for act in (zip (*self.tree[currentNode]))[0]: 
                existingDims += self.actions[act][0].keys()
        
        intersectSet = list(set(dims).intersection(set(existingDims)))
        if intersectSet == []: 
            e = dims[0]
        else: 
            e = intersectSet[0]
        for k,ls in self.actions.iteritems():
            if e in ls[1].keys() : 
                return (k, [ e2 for e2 in dims if e2 not in ls[1].keys() ])
        print "decisionTree: getAction: cannot find action "
        
        
    def removeEmptyLeaves(self): 
        emptyLeaves = []
        for key, value in self.tree.iteritems(): 
            if value == []: 
                emptyLeaves.append(key)
        for key in emptyLeaves: 
            del self.tree[key]
            
        for key, value in self.tree.iteritems(): 
            value2 = []
            for (act,nextNode) in value: 
                if nextNode not in emptyLeaves: 
                    value2.append((act,nextNode))
            self.tree[key] = value2
          
    def outputTree(self):
    
        nodes = sorted(self.tree.keys())
        self.imagesOnTree[nodes[0]] = self.image
        activeNodes = [nodes[0]]
        (origClass,origConfident) = NN.predictWithImage(self.model,self.image)
        while activeNodes != []: 
            activeNodes = self.getNodes(activeNodes)
            
        for node,image in self.imagesOnTree.iteritems(): 
            (newClass,newConfident) = NN.predictWithImage(self.model,image)
            if self.tree[node] == []: 
                if newClass == origClass: 
                    print("outputTree: it is expected that the leaf nodes have class change.")
                path0="%s/%s_decisionTree_%s_leaf_%s_%s.png"%(directory_pic_string,startIndexOfImage,node,dataBasics.LABELS(int(newClass)),newConfident)
            else: 
                nodeStr = (zip(*(self.tree[node])))[1]
                if newClass != origClass and newConfident > effectiveConfidenceWhenChanging: 
                    print("outputTree: it is expected that the internal nodes have the same class.")
                path0="%s/%s_decisionTree_%s_%s_%s_%s.png"%(directory_pic_string,startIndexOfImage,node,str(nodeStr),dataBasics.LABELS(int(newClass)),newConfident)
            dataBasics.save(-1,image,path0)
        
    def getNodes(self,activeNodes):
        node = activeNodes[0]
        for (act, nextNode) in self.tree[node]:
            (spansPath,numSpansPath,_) = self.actions[act] 
            self.imagesOnTree[nextNode] = applyManipulation(self.imagesOnTree[node],spansPath,numSpansPath)
            activeNodes.append(nextNode) 
        return activeNodes[1:]
            
        
    def show(self):
    
        #print("actions on decision tree: %s"%self.actions)
        #print("decision tree: %s"%self.tree)
        
        import graphviz as gv
        import networkx as nx
        
        graph = gv.Digraph(format='svg')
        vNodes = {}
        for node in self.tree.keys():
            graph.node('%s'%(node))
            
        for node in self.tree.keys():
            for (act, nextNode) in self.tree[node]: 
                graph.edge('%s'%node, '%s'%nextNode)
        filename = '%s/%s_decisionTree.gv'%(directory_pic_string,startIndexOfImage)
        graph.render(filename)
        
        print("please find the decision tree from file: %s"%(filename))
        
