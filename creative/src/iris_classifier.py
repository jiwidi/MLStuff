from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import tensorflow as tf
import numpy as np
import csv

#Load data
dataTrain=np.array([])
targetTrain=[]
dataTest=np.array([])
targetTest=[]

with open('../data/iris.csv', 'rb') as csvfile:
    data=csvfile.read().decode().split('\n')
    train=int(len(data)*0.8)
    test=len(data)-train
    print(train)
    print(test)
    for u in range(1,len(data)-1):
        dat=data[u].split(',')
        label=dat[len(dat)-1]
        if label.strip()=='Iris-setosa':
            label=0
        elif label.strip()=='Iris-versicolor':
            label=1
        elif label.strip()=='Iris-virginica':
            label=2
        print(label)
        dat=dat[1:len(dat)-1]
        dat=[ float(x) for x in dat ]
        dat=np.array(dat)
        #for a in range(len(dat)):
        #   dat[u]=float(dat[a])
        print(str(dat)+'  '+str(label)+'  '+str(u))
        if(u<train):
            #dataTrain.append(dat)
            np.append(dataTrain, dat)
            targetTrain.append(label)
            #np.append(targetTrain, label)
        else:
            #dataTest.append(dat)
            np.append(dataTest, dat)
            targetTest.append(label)
            #np.append(targetTest, label)





#Shaping feature columns for classifier
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

#Let's build a DNN with the classifier class
classifier=tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[32],
                                          model_dir="/tmp/iris_dnn",
                                          n_classes=3,
                                          dropout=None,)

print('Hola')
#Train the model
print(targetTrain)
classifier.fit(x=dataTrain,
               y=targetTrain,
               steps=2000)
print('he')
#Let's use the test set to test accuracy
accuracy_score=classifier.evaluate(x=dataTest,y=targetTest)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))

