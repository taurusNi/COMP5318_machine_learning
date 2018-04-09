from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time
import numpy as np
from sklearn.metrics import confusion_matrix

recordList = list()
labelList = list()
test_recordList = list()
test_labelList = list()


with open('/Users/taurus/Desktop/COMP5318 Machine learning/assignment2/adult_train.csv') as f:
     headReader = f.readline()
     train_temp = f.readlines()
     header = headReader.strip().split(",")
     for i in train_temp:
        labelList.append(i.strip().split(",")[-1])
        row = {}
        for a in range(0,len(i.strip().split(","))-1):
            #print(i.strip().split(",")[a])
            if i.strip().split(",")[a].strip().isdigit():
                row[header[a]] = int(i.strip().split(",")[a])
                #print(row)
            else:
                row[header[a]] = i.strip().split(",")[a]
        recordList.append(row)

with open('/Users/taurus/Desktop/COMP5318 Machine learning/assignment2/adult_test.csv') as f:
    test_headReader = f.readline()
    test_temp = f.readlines()
    testHeader = test_headReader.strip().split(",")
    for i in test_temp:
        test_labelList.append(i.strip().split(",")[-1])
        test_row = {}
        for a in range(0,len(i.strip().split(","))-1):
            if i.strip().split(",")[a].strip().isdigit():
                test_row[testHeader[a]] = int(i.strip().split(",")[a])
                #print(int(i.rstrip().split(",")[a]))
            else:
                test_row[testHeader[a]] = i.strip().split(",")[a]
                #print(test_row)
        test_recordList.append(test_row)

test_labelList.pop()
test_recordList.pop()
recordList.pop()
labelList.pop()

pca = decomposition.PCA(n_components=105)

vec = DictVectorizer()
vec_test =  DictVectorizer()
dummyX = vec.fit_transform(recordList).toarray()
dummyX = pca.fit_transform(dummyX)
test_dummyX = vec_test.fit_transform(test_recordList).toarray()
test_dummyX = pca.fit_transform(test_dummyX)


label = preprocessing.LabelBinarizer()
dummyY = label.fit_transform(labelList)
test_dummyY = label.fit_transform(test_labelList)

scaler = StandardScaler()
scaler.fit(dummyX)
scaler.transform(dummyX)
#t = dummyX[0:7000,:]
#y = dummyY[0:7000]
mlp = MLPClassifier(hidden_layer_sizes=(30,30), activation='logistic',solver='sgd',alpha=0.1,batch_size=100,max_iter=10000)
print(time.strftime('%Y-%m-%d %H:%M:%S'))
#mlp.fit(dummyX,dummyY)
mlp.fit(dummyX,dummyY)
#print(mlp.validation_scores_)
predictions = mlp.predict(dummyX)
print(np.mean(predictions==dummyY)*100)
print(time.strftime('%Y-%m-%d %H:%M:%S'))
t = confusion_matrix(dummyY,predictions)
print(t)

scaler.fit(test_dummyX)
scaler.transform(test_dummyX)
predic  = mlp.predict(test_dummyX)
print(np.mean(predic==test_dummyY)*100)
t = confusion_matrix(test_dummyY,predic)
print(t)





