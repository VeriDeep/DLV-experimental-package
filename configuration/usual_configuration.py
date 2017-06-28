#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""

def usual_configuration(dataset):

        
    if dataset == "mnist": 

        # which image to start with or work with 
        # from the database
        startIndexOfImage = 2239
        
        # the layer to work with 
        # -1 is input layer
        startLayer = -1

        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",0.03)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        
        effectiveConfidenceWhenChanging = 0.8
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 1
        
        # tunable parameter for MCTS
        explorationRate = 0.5
            
    elif dataset == "cifar10": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 594
        
        # the start layer to work from 
        startLayer = -1
        
        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",0.02)
        
        effectiveConfidenceWhenChanging = 0.8
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 1
 
        explorationRate = 0.5        
        
    elif dataset == "gtsrb": 

        # which image to start with or work with 
        # from the database
        startIndexOfImage = 1121
        
        # the layer to work on 
        startLayer = -1

        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",0.03)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        
        effectiveConfidenceWhenChanging = 0.8
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 1

        explorationRate = 0.5

    elif dataset == "imageNet": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 8
        
        # the start layer to work from 
        startLayer = -1

        ## control by distance
        controlledSearch = ("L1",2000.00)
        #controlledSearch = ("L1",0.05)
        
        effectiveConfidenceWhenChanging = 0.1
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 1

        explorationRate = 0.5
    
    return (startIndexOfImage,startLayer,explorationRate,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,effectiveConfidenceWhenChanging)