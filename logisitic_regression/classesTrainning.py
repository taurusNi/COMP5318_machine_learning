#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:46:58 2017

@author: taurus
"""

import numpy as np
import logisticRegre as lr


training_set = np.load('/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/new_traning_set.npy')
labels = np.load('/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/new_traning_set_labels.npy')
group = np.load('/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/seperateClass.npy').item()

regNum = 0.1
n_labels = 30
X_new = np.c_[np.ones((training_set.shape[0],1)),training_set]

training_result = lr.oneVsAllClassifier(X_new,labels,n_labels,regNum)
lr.predictForTraining(training_result,training_set,labels)