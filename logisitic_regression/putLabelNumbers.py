#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:10:57 2017

@author: taurus
"""

import numpy as np
group = np.load('/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/seperateClass.npy').item()


    

#---------get seperate labels value-------1 to 30---------
num=0
totalNums =0
array = list()
for i,v in enumerate(group.keys()):
    totalNums += len(group[v]) # count total number of the records
labels = np.zeros((totalNums,1))
for i,v in enumerate(group.keys()):
    for j,k in enumerate(group[v]):
        labels[j+num] = i+1
        k.pop(10221) #remove thoes two cloumns as they are all zero
        k.pop(7617)  
        array.append(k)
    num +=len(group[v])
#-------------------------------------------------    
training_set = np.array(array,float) # transfer to array where row is sample and column is feature 
#mean nomalization and feature scaling
temp_training_set = (training_set-training_set.mean(0))

#/np.std(training_set,axis=0)
#-----PCA---------
#covriance matrix
#training_setT = temp_training_set.T
#cov = np.cov(training_setT)
cov = temp_training_set.T.dot(temp_training_set)/temp_training_set.shape[0]
print('Starting.....PCA')
U,s,V = np.linalg.svd(cov)
#conponent pick
print('Considering...K of conponents')
valueTotal = np.sum(s)
tempTotal = 0;
k0=k1=k2=k3=k4=k5=k6=k7=k8=0
for i,v in enumerate(s):
    tempTotal +=v
    if(tempTotal/valueTotal>0.4):
        if(k0==0):
            k0=i
            print('{} % variance retained in {} dimensions'.format(tempTotal/valueTotal, k0+1))
    if(tempTotal/valueTotal>0.5):
        if(k1==0):
            k1=i
            print('{} % variance retained in {} dimensions'.format(tempTotal/valueTotal, k1+1))
    if(tempTotal/valueTotal>0.6):
        if(k2==0):
            k2=i
            print('{} % variance retained in {} dimensions'.format(tempTotal/valueTotal, k2+1))
    if(tempTotal/valueTotal>0.7):
        if(k3==0):
            k3=i
            print('{} % variance retained in {} dimensions'.format(tempTotal/valueTotal, k3+1))
    if(tempTotal/valueTotal>0.8):
        if(k4==0):
            k4=i
            print('{} % variance retained in {} dimensions'.format(tempTotal/valueTotal, k4+1))
    if(tempTotal/valueTotal>0.85):
        if(k5==0):
            k5=i
            print('{} % variance retained in {} dimensions'.format(tempTotal/valueTotal, k5+1))
    if(tempTotal/valueTotal>0.9):
        if(k6==0):
            k6=i
            print('{} % variance retained in {} dimensions'.format(tempTotal/valueTotal, k6+1))
    if(tempTotal/valueTotal>0.95):
        if(k7==0):
            k7=i
            print('{} % variance retained in {} dimensions'.format(tempTotal/valueTotal, k7+1))
    if(tempTotal/valueTotal>0.99):
        if(k8==0):
            k8=i
            print('{} % variance retained in {} dimensions'.format(tempTotal/valueTotal, k8+1))
        else:
            break

#accum_value = np.array([np.sum(s[: i + 1]) for i in range(training_set.shape[1])])
#k0 = len(accum_value[accum_value < 0.4])
#k1 = len(accum_value[accum_value < 0.5])
#k2 =len(accum_value[accum_value < 0.6])
#k3 = len(accum_value[accum_value < 0.7])
#k4 = len(accum_value[accum_value < 0.8])
#k5 = len(accum_value[accum_value < 0.85])
#k6 = len(accum_value[accum_value < 0.9])
#k7 = len(accum_value[accum_value < 0.95])
#k8 = len(accum_value[accum_value < 0.99])
# +1 will refer to the real dimensions
#print('{} % variance retained in {} dimensions'.format(accum_value[k0], k0+1)) 
#print('{} % variance retained in {} dimensions'.format(accum_value[k1], k1+1))
#print('{} % variance retained in {} dimensions'.format(accum_value[k2], k2+1))
#print('{} % variance retained in {} dimensions'.format(accum_value[k3], k3+1))
#print('{} % variance retained in {} dimensions'.format(accum_value[k4], k4+1))
#print('{} % variance retained in {} dimensions'.format(accum_value[k5], k5+1))
#print('{} % variance retained in {} dimensions'.format(accum_value[k6], k6+1))
#print('{} % variance retained in {} dimensions'.format(accum_value[k7], k7+1))
#print('{} % variance retained in {} dimensions'.format(accum_value[k8], k8+1))

# we choose 95% remained
U_reduced = U[:, : k7+1] #as pyhon will not include the last one so add 1 
#new_traning_set = U_reduced.T.dot(training_setT)
new_traning_set =  temp_training_set.dot(U_reduced)
print('Starting.....write')
#write to file for next traninig step
np.save('/Users/taurus/Desktop/COMP5318 Machine learning/assignment1/assignment1_2017S1/new_traning_set.npy', new_traning_set)
np.save('/Users/taurus/Desktop/COMP5318 Machine learning/assignment1/assignment1_2017S1/new_traning_set_labels.npy', labels)



