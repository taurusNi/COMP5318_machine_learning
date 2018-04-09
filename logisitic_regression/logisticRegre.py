#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:35:35 2017

@author: taurus
"""

import numpy as np
from scipy import optimize

def sigm(x):
    return(1 / (1 + np.exp(-x)))

def reguCostFunction(theta, regNum, traning_set, traning_set_real_result):
    records = traning_set.shape[0]
    prediction = sigm(traning_set.dot(theta))
    part1 =  np.log(prediction).T.dot(traning_set_real_result)
    part2 =  np.log(1-prediction).T.dot(1-traning_set_real_result)
    Jcost1 = -1 * (part1+part2) / records
    #Jcost1 = -(traning_set_real_result.T.dot(np.log(prediction))+(1-traning_set_real_result).T.dot(np.log(1-prediction)))/records
    temp = np.sum(theta**2)-theta[0]**2
    Jcost2 = temp*regNum/(2*records)
    Jcost = Jcost1+Jcost2
    #J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    return Jcost[0]

def reguGrad(theta, regNum, traning_set, traning_set_real_result):
    records = traning_set.shape[0]
    prediction = sigm(traning_set.dot(theta.reshape(-1,1))) #just flatten the theta as it must be 1-D
    temp = (1/records)*(traning_set.T).dot(prediction-traning_set_real_result) + regNum/records*theta.reshape(-1,1)
    temp[0] = temp[0]-regNum/records*theta[0].reshape(-1,1)
    grad = temp
    return grad.flatten()

    

def oneVsAllClassifier(traning_set, traning_set_real_result, n_labels, regNum):
    theta = np.zeros((traning_set.shape[1],1))
    traning_result = np.zeros((n_labels,traning_set.shape[1]))
    for class_ in range(n_labels): # 0 to 29 equals to classes labels 1 to 30
        #True*1 = 1, False*1 = 0
        print('Starting training class {} '.format(class_+1))
        inter = optimize.minimize(reguCostFunction, theta, args=(regNum, traning_set, (traning_set_real_result == class_+1)*1), method=None,
                       jac=reguGrad, options={'maxiter':50})
        traning_result[class_] = inter.x
    return traning_result 


def predictForTraining(traning_result,traning_set,traning_set_real_result):
    hyp = sigm(traning_set.dot(traning_result.T))
    predict_y =np.argmax(hyp, axis=1) # to find the sample belongs to which class
    real_y = traning_set_real_result-1 # as the index starting from 0
    match_result  = (predict_y == real_y.flatten())*1 #to be the same direction
    print('Clasifier\'s accuracy is : {} %'.format(np.round(np.mean(match_result)*100,4)))
    return predict_y
        
def conMatrix(predict_y,group,n_labels):
    conMat = np.zeros((n_labels,n_labels))
    total=0
    for i,v in enumerate(group):
        for j,k in enumerate(group[v]):
            conMat[i,predict_y[total+j]]+=1
        total += len(group[v])
    return conMat

def evaluate(conMat):
    rows = conMatrix.shape[0]
    temp_for_presicion = np.sum(conMat,axis=0)
    temp_for_recall = np.sum(conMat,axis=1)
    for i in range(rows):
        TP = conMat[i,i]
        #precision
        presicion = TP/temp_for_presicion[i]
        print("The accuracy of class {} is {}".format(i+1,presicion))
        #recall
        recall = TP/temp_for_recall[i]
        print("The recall of class {} is {}".format(i+1,recall))
        #F-Measure
        f_measure = 2*recall*presicion/(recall+presicion)
        print("The f_measure of class {} is {}".format(i+1,f_measure))
            
def predictForTest(traning_result,testing_set,class_label_pairs):
    labels = list()
    hyp = sigm(testing_set.dot(traning_result.T))
    predict_y =np.argmax(hyp, axis=1)
    for i,v in enumerate(predict_y):
        labels.append(class_label_pairs[v])
    return labels
    
    
