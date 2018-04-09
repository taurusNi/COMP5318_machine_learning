#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:45:48 2017

@author: taurus
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import time

#change string to int
def workclass(string):
    state = False #if it is unknown, it is False
    place = 0
    choises = 'Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'
    choises_separate = choises.replace(" ","").split(",")
    for i,v in enumerate(choises_separate):
        if(string==v):
            state = True
            place = i+1
            break
    if(state):
        return place
    else:
        return len(choises_separate)+1
def education(string):
    state = False #if it is unknown, it is False
    place = 0
    choises = 'Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'
    choises_separate = choises.replace(" ","").split(",")
    for i,v in enumerate(choises_separate):
        if(string==v):
            state = True
            place = i+1
            break
    if(state):
        return place
    else:
        return len(choises_separate)+1
def marital(string):
    state = False #if it is unknown, it is False
    place = 0
    choises = 'Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse'
    choises_separate = choises.replace(" ","").split(",")
    for i,v in enumerate(choises_separate):
        if(string==v):
            state = True
            place = i+1
            break
    if(state):
        return place
    else:
        return len(choises_separate)+1
def occupation(string):
    state = False #if it is unknown, it is False
    place = 0
    choises = 'Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces'
    choises_separate = choises.replace(" ","").split(",")
    for i,v in enumerate(choises_separate):
        if(string==v):
            state = True
            place = i+1
            break
    if(state):
        return place
    else:
        return len(choises_separate)+1
def relationship(string):
    state = False #if it is unknown, it is False
    place = 0
    choises = 'Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'
    choises_separate = choises.replace(" ","").split(",")
    for i,v in enumerate(choises_separate):
        if(string==v):
            state = True
            place = i+1
            break
    if(state):
        return place
    else:
        return len(choises_separate)+1      
def race(string):
    state = False #if it is unknown, it is False
    place = 0
    choises = 'White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'
    choises_separate = choises.replace(" ","").split(",")
    for i,v in enumerate(choises_separate):
        if(string==v):
            state = True
            place = i+1
            break
    if(state):
        return place
    else:
        return len(choises_separate)+1
def sex(string):
    state = False #if it is unknown, it is False
    place = 0
    choises = 'Female, Male'
    choises_separate = choises.replace(" ","").split(",")
    for i,v in enumerate(choises_separate):
        if(string==v):
            state = True
            place = i+1
            break
    if(state):
        return place
    else:
        return len(choises_separate)+1 
def native(string):
    state = False #if it is unknown, it is False
    place = 0
    choises = 'United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands'
    choises_separate = choises.replace(" ","").split(",")
    for i,v in enumerate(choises_separate):
        if(string==v):
            state = True
            place = i+1
            break
    if(state):
        return place
    else:
        return len(choises_separate)+1
def readTrainData():
    with open('/Users/taurus/Desktop/COMP5318 Machine learning/assignment2/adult_train.csv') as d:
        train_data = list()
        train_label = list()
        header = d.readline()
        data_f = d.readlines()
        for i in data_f:
            temp = i.replace(" ","").split(",")
            if(len(temp)==15): #remove invalied records
              temp[1]=workclass(temp[1])
              temp[3]=education(temp[3])
              temp[5]=marital(temp[5])
              temp[6]=occupation(temp[6])
              temp[7]=relationship(temp[7])
              temp[8]=race(temp[8])
              temp[9]=sex(temp[9])
              temp[13]=native(temp[13])
              train_data.append(temp[0:14]) #get train data
              if(temp[14:15][0]=='>50K\n'):
                train_label.append(1)#get label >50K is 1
              else:
                train_label.append(0)#get label <=50K is 0
        train_data = np.array(train_data,float)
        train_label = np.array(train_label,float)
    return train_data, train_label

def readTestData():
    with open('/Users/taurus/Desktop/COMP5318 Machine learning/assignment2/adult_test.csv') as d:
        data_test = list()
        label_test = list()
        header = d.readline()
        data_f = d.readlines()
        for i in data_f:
            temp = i.replace(" ","").split(",")
            if(len(temp)==15): #remove invalied records
                temp[1]=workclass(temp[1])
                temp[3]=education(temp[3])
                temp[5]=marital(temp[5])
                temp[6]=occupation(temp[6])
                temp[7]=relationship(temp[7])
                temp[8]=race(temp[8])
                temp[9]=sex(temp[9])
                temp[13]=native(temp[13])
                data_test.append(temp[0:14]) #get test data
                if(temp[14:15][0]=='>50K.\n'):
                    label_test.append(1)#get label >50K is 1
                else:
                    label_test.append(0)#get label <=50K is 0
        data_test = np.array(data_test,float)
        label_test = np.array(label_test,float)
        return data_test, label_test
def chageFeatures(data):
    temp = np.ones((data.shape[0],1))
    for i in range(0,14):
        temp = np.c_[temp,data[:,i]]
        for j in range(i,14):
            temp = np.c_[temp,data[:,i]*data[:,j]]
    return temp

def class_precision_recall_fscore(conMat):
    rows = conMat.shape[0]
    temp_for_presicion = np.sum(conMat, axis=0)
    temp_for_recall = np.sum(conMat, axis=1)
    for i in range(rows):
        TP = conMat[i, i]
        # precision
        presicion = TP / temp_for_presicion[i]
        print("The precision of class {} is {}".format(i + 1, presicion))
        # recall
        recall = TP / temp_for_recall[i]
        print("The recall of class {} is {}".format(i + 1, recall))
        # f_measure
        f_measure = 2 * recall * presicion / (recall + presicion)
        print("The f_measure of class {} is {}".format(i + 1, f_measure))
        print()
    
    

if __name__ == "__main__":
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    train_data, train_label = readTrainData()
    data_test, label_test = readTestData()
    train_data = chageFeatures(train_data)
    data_test = chageFeatures(data_test)
    scaler = StandardScaler()
    scaler.fit(train_data)
    scaler.transform(train_data)
    scaler.fit(data_test)
    scaler.transform(data_test)
    mlp = MLPClassifier(hidden_layer_sizes=(130),activation='logistic',solver='sgd',batch_size=100,random_state=64, max_iter=30)
    mlp.fit(train_data,train_label)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    #print("training prediction......")
    #predictions = mlp.predict(train_data)
    #cm = confusion_matrix(train_label,predictions)
    #print("total accuracy is {}".format(mlp.score(train_data, train_label)))
    #class_precision_recall_fscore(cm)
    #print("the macro precision_recall_fscore is ",precision_recall_fscore_support(train_label,predictions,average='macro'))
    #print()
    #print("the micro precision_recall_fscore is ",precision_recall_fscore_support(train_label,predictions,average='micro'))
    
    print("testing prediction......")
    predictions = mlp.predict(data_test)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    cm = confusion_matrix(label_test,predictions)
    print(cm)
    print("total accuracy is {}".format(mlp.score(data_test, label_test)))
    print()
    class_precision_recall_fscore(cm)
    print("the macro precision_recall_fscore is ",precision_recall_fscore_support(label_test,predictions,average='macro'))
    print()
    print("the micro precision_recall_fscore is ",precision_recall_fscore_support(label_test,predictions,average='micro'))
    
    
    