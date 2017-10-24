#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""

import math


def usual_configuration(dataset):

        
    if dataset == "mnist": 

        # which image to start with or work with 
        # from the database
        startIndexOfImage = 3
        
        # the layer to work with 
        # -1 is input layer
        startLayer = -1

        ## control by distance
        #controlledSearch = ("euclidean",4)
        controlledSearch = ("L1",10)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        distanceConst = 0
        
        effectiveConfidenceWhenChanging = 0.0
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 100
        MCTS_all_maximal_time = 300
        MCTS_multi_samples = 1
        
        # tunable parameter for MCTS
        explorationRate = math.sqrt(2)
            
    elif dataset == "cifar10": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 142
        
        # the start layer to work from 
        startLayer = -1
        
        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",10)
        distanceConst = 0
        
        effectiveConfidenceWhenChanging = 0.0
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 1
 
        explorationRate = math.sqrt(2)        
        
    elif dataset == "gtsrb": 

        # which image to start with or work with 
        # from the database
        startIndexOfImage = 1
        
        # the layer to work on 
        startLayer = -1

        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",10)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        distanceConst = 0.04
        
        effectiveConfidenceWhenChanging = 0.0
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 1

        explorationRate = math.sqrt(2)

    elif dataset == "imageNet": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 8
        
        # the start layer to work from 
        startLayer = -1

        ## control by distance
        controlledSearch = ("L1",0.5)
        #controlledSearch = ("L1",0.05)
        distanceConst = 0.04
        
        effectiveConfidenceWhenChanging = 0.0
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 1

        explorationRate = math.sqrt(2)
    
    return (distanceConst,startIndexOfImage,startLayer,explorationRate,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,effectiveConfidenceWhenChanging)