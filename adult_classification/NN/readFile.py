#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:11:32 2017

@author: taurus
"""
#read file in

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
from sklearn.model_selection import KFold

def workclass(string):
    state = False
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
    state = False
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
    state = False
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
    state = False
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
    state = False
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
    state = False
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
    state = False
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
    state = False
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
with open('/Users/taurus/Desktop/COMP5318 Machine learning/assignment2/adult.data.txt') as d:
    #group_data = {}
    #group_label = {}
    data = list()
    label = list()
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
          data.append(temp[0:14]) #get data
          if(temp[14:15][0]=='>50K\n'):
            label.append(1)#get label >50K is 1
          else:
            label.append(0)#get label <=50K is 0
with open('/Users/taurus/Desktop/COMP5318 Machine learning/assignment2/adult.test.txt') as d:
    #group_data = {}
    #group_label = {}
    data_test = list()
    label_test = list()
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
          data_test.append(temp[0:14]) #get data
          if(temp[14:15][0]=='>50K.\n'):
            label_test.append(1)#get label >50K is 1
          else:
            label_test.append(0)#get label <=50K is 0
train_data = np.array(data,float)
train_label = np.array(label,float)
test_data = np.array(data_test,float)
test_label = np.array(label_test,float)
#train_data = np.c_[np.ones((train_data.shape[0],1)),train_data]
#test_data = np.c_[np.ones((test_data.shape[0],1)),test_data]
temp = np.ones((train_data.shape[0],1))
for i in range(0,14):
    temp = np.c_[temp,train_data[:,i]]
   #print(i)
    for j in range(i,14):
        temp = np.c_[temp,train_data[:,i]*train_data[:,j]]
        #for k in range(j,14):
           #temp = np.c_[temp,train_data[:,i]*train_data[:,j]*train_data[:,k]]
train_data = temp
print(train_data.shape)       


temp = np.ones((test_data.shape[0],1))
for i in range(0,14):
    temp = np.c_[temp,test_data[:,i]]
   #print(i)
    for j in range(i,14):
        temp = np.c_[temp,test_data[:,i]*test_data[:,j]]
        #for k in range(j,14):
           #temp = np.c_[temp,train_data[:,i]*train_data[:,j]*train_data[:,k]]
test_data = temp
print(test_data.shape) 

    
#numD=0
#numX=0
#for i,v in enumerate(train_label):
#    if(v==0):
#        numX=numX+1
#    else:
#        numD=numD+1
#print("<=50K is",numX)
#print(">50K is",numD)
    
bs = open('/Users/taurus/Desktop/COMP5318 Machine learning/assignment2/units2.csv','w')
#pca = decomposition.PCA(n_components=13)
#train_data = pca.fit_transform(train_data)
#temp_train_data = (train_data - train_data.mean(0))
#temp_test_dummyX = (test_dummyX - test_dummyX.mean(0))
#cov = np.cov(temp_train_data, rowvar=False)
#U, s, V = np.linalg.svd(cov)
#valueTotal = np.sum(s)
#tempTotal = 0
#K=0
#for i, v in enumerate(s):
#    tempTotal += v
#    if (tempTotal / valueTotal >= 0.95):
#        K = i
#        print('{} % variance retained in {} dimensions'.format(tempTotal / valueTotal, K + 1))
#        break
#U_reduced = U[:, : K + 1]
#train_data = temp_train_data.dot(U_reduced) 
#test_dummyX = temp_test_dummyX.dot(U_reduced)
kf = KFold(n_splits=10)
scaler = StandardScaler()
scaler.fit(train_data)
scaler.transform(train_data)
scaler = StandardScaler()
scaler.fit(test_data)
scaler.transform(test_data)


accur = 0
ap = 0.0001
#title = ","
#for i in range(60,210,10):
#    title = title+str(i)+','
#bs.writelines(title)
#bs.write("\n")

#for j in range(70,210,10):
p=0
num=0
#print(j)
#value = str(j)+","
#print(value)
for i in range(1,1000):
    print(i)
    #if(i==100):
    #    print(i)
    #if(i==500):
    #    print(i)
    #if(i==900):
    #    print(i)

    mlp = MLPClassifier(hidden_layer_sizes=(130),activation='logistic',solver='sgd',alpha=0.01,batch_size=100,random_state=64, max_iter=100000)
            #print(time.strftime('%Y-%m-%d %H:%M:%S'))
    mlp.fit(train_data,train_label) 
    predictions = mlp.predict(test_data)
            #print(time.strftime('%Y-%m-%d %H:%M:%S'))
    t = confusion_matrix(test_label,predictions)
            #recall = t[1,1]/(t[1,1]+t[1,0])
            #print(recall)
            #value = value+str(recall)+","
                              #ap = ap*3
                              #print("recall is {}".format(recall))
    if(p<t[1,1]):    
        p=t[1,1]    
        num=i
                                  #accur = np.mean((predictions==train_label)*100)
        print(p)
        print(t)
                                  #print(value)
value = str(p)+","+str(num)
bs.writelines(value)
print(num)
print(p)
bs.write("\n")
bs.close

print(accur)
#mlp = MLPClassifier(hidden_layer_sizes=(140),activation='logistic',solver='sgd',alpha=0.1,batch_size=200,random_state=738, max_iter=100000)
#print(time.strftime('%Y-%m-%d %H:%M:%S'))
#mlp.fit(train_data,train_label) 
#predictions = mlp.predict(test_data)
#print(time.strftime('%Y-%m-%d %H:%M:%S'))
#t = confusion_matrix(test_label,predictions)
#print("recall is {}".format(t[1,1]/(t[1,1]+t[1,0])))
#print(t)
#print()
#predictions = mlp.predict(train_data)
#t = confusion_matrix(train_label,predictions)
#print("recall is {}".format(t[1,1]/(t[1,1]+t[1,0])))
#print(t)
#print()
#print("TEST")
#predictions = mlp.predict(test_data)
#numD=0
#numX=0
#for i,v in enumerate(test_label):
#    if(v==1):
#        numD=numD+1
#    else:
#        numX=numX+1
#print(numD)
#print(numX)
#print(mlp.score(test_data,test_label))
#t = confusion_matrix(test_label,predictions)
#print(t) 

