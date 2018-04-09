#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:59:14 2017

@author: taurus
"""

import numpy as np
import logisticRegre as lr
class_label_pairs = {}
data = list()
array = list()
app = list()
group = np.load('/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/seperateClass.npy').item()

for i,v in enumerate(group):
    class_label_pairs[i] = v


#读取映射矩阵U_reduce
#读取训练模块training_result

with open('/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/test_data.csv') as d:
    data_f = d.readlines()
    for i in data_f:
        data.append(i.rstrip().split(","))
    for i in range(len(data)):
        temp = data[i][1:]
        temp.pop(10221)
        temp.pop(7617)
        array.append(temp)
        app.append(data[i][0])
        
testing_set = np.array(array,float)
temp_testing_set = (testing_set-testing_set.mean(0))

#PCA
new_testing_set = temp_testing_set.dot(U_reduce)


X_new = np.c_[np.ones((new_testing_set.shape[0],1)),new_testing_set]

labels = lr.predictForTest(traning_result,X_new,class_label_pairs)

fw = open('/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/real_labels_for_training_set.txt','w')

for i in range(labels):
     fw.write(app[i]+' -- '+labels[i])
     
fw.close
#print(class_label_pairs)