import numpy as np
import csv
from collections import namedtuple

# my_matrix = np.loadtxt(open("/Users/yangyizhou/Documents/workspace/Python/Assignment/assignment1_2017S1/training_data.csv","rb"),delimiter=",",skiprows=0)
group = {}
label=list()
data = list()
with open('/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/training_labels.csv') as f, open(
        '/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/training_data.csv') as d:
    label_f = f.readlines()
    data_f = d.readlines()

    for r in label_f:
        label.append(r.rstrip().split(","))
        # t = r.rstrip().split(",")
        # label[t[0]]=t[1]

    for i in data_f:
        data.append(i.rstrip().split(","))
    # print(data[0])

    for i in range(len(data)): #len(data)这个data有多少个元素
        for j in range(len(label)):
            if data[i][0] == label[j][0]: #二维数组，i,j表示元素~后面的零表示第一位
                #data.append(label[j][1]) #app name 一样， 把class类型加进去
                if label[j][1] in group.keys(): # group.keys() 获得字典的键组

                    group[label[j][1]].append(data[i][1:])
                else:
                    temp = label[j][1]
                    temp = []
                    group[label[j][1]] = list([data[i][1:]])

    print(group.keys())
    np.save('/Users/taurus/Desktop/COMP5318 Machine learning\
/assignment1/assignment1_2017S1/seperateClass.npy', group)

# read_dictionary = np.load('anssignment1/test.npy').item()  用这个来读上面生成的文件，获得字典，就类似于java的map键值对

# 结构： key：label 的名字，一共30个，对应的value都是一个list，就是每个label对应的record，list 的每个元素都是一条记录，也是一个list，
#       第0位是app的名字，第1位开始到末尾就都是数字
# 729，726，479，699，671，654，716，723，709，743，692，710，715，657，726，713，656，725，730，726，359，738，621，703，766，
# 684，716，702，432，484 这些是每个label对应的数据数量，合集20104条，和原始数据相同




