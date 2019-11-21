# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:43:41 2019

@author: abhisek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset=pd.read_csv("worderful.tsv",delimiter='\t',quoting=3,encoding='latin-1')

X=dataset.iloc[:,1:4].values #selecting independent variables rows and columns
y=dataset.iloc[:,4].values




from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)





"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

from sklearn.linear_model import LogisticRegression
classifier1=LogisticRegression(random_state=0)
classifier1.fit(X_train,y_train)
y_pred1 = classifier1.predict(X_test)

from sklearn.naive_bayes import GaussianNB
classifier2=GaussianNB()
classifier2.fit(X_train,y_train)
y_pred2=classifier2.predict(X_test)

from sklearn.svm import SVC
classifier3=SVC(gamma='auto')
classifier3.fit(X_train,y_train)
y_pred3=classifier3.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier4=KNeighborsClassifier()
classifier4.fit(X_train,y_train)
y_pred4=classifier4.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier5=DecisionTreeClassifier()
classifier5.fit(X_train,y_train)
y_pred5=classifier5.predict(X_test)

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pred1)
cm2=confusion_matrix(y_test,y_pred2)
cm3=confusion_matrix(y_test,y_pred3)
cm4=confusion_matrix(y_test,y_pred4)
cm5=confusion_matrix(y_test,y_pred5)

accuracy1=(cm1[0][0]+cm1[1][1])/(cm1[0][0]+cm1[1][1]+cm1[0][1]+cm1[1][0])
print("Accuraccy using Logistic Regression : ",accuracy1)

accuracy2=(cm2[0][0]+cm2[1][1])/(cm2[0][0]+cm2[1][1]+cm2[0][1]+cm2[1][0])
print("Accuraccy using Naive Bayes : ",accuracy2)

accuracy3=(cm3[0][0]+cm3[1][1])/(cm3[0][0]+cm3[1][1]+cm3[0][1]+cm3[1][0])
print("Accuraccy using SVM : ",accuracy3)

accuracy4=(cm4[0][0]+cm4[1][1])/(cm4[0][0]+cm4[1][1]+cm4[0][1]+cm4[1][0])
print("Accuraccy using KNN : ",accuracy4)

accuracy5=(cm5[0][0]+cm5[1][1])/(cm5[0][0]+cm5[1][1]+cm5[0][1]+cm5[1][0])
print("Accuraccy using Decision Tree : ",accuracy5)





for i in range(0,len(y_pred1)):
    x=X_test[i][2]
    Y=y_pred1[i]
    if(Y==y_test[i] and Y==1):
        plt.scatter(x,Y,color="green")
    elif(Y==y_test[i] and Y==0):
        plt.scatter(x,Y,color="blue")
    elif(Y!=y_test[i] and Y==1):
        plt.scatter(x,Y,color="red")
    else:
         plt.scatter(x,Y,color="yellow")
    plt.xlabel("Ratio of length and suffix")
    plt.ylabel("Predicted value")
plt.show()

for i in range(0,len(y_pred2)):
    x=X_test[i][2]
    Y=y_pred2[i]
    if(Y==y_test[i] and Y==1):
        plt.scatter(x,Y,color="green")
    elif(Y==y_test[i] and Y==0):
        plt.scatter(x,Y,color="blue")
    elif(Y!=y_test[i] and Y==1):
        plt.scatter(x,Y,color="red")
    else:
         plt.scatter(x,Y,color="yellow")
    plt.xlabel("Ratio of length and suffix")
    plt.ylabel("Predicted value")
plt.show()

for i in range(0,len(y_pred3)):
    x=X_test[i][2]
    Y=y_pred3[i]
    if(Y==y_test[i] and Y==1):
        plt.scatter(x,Y,color="green")
    elif(Y==y_test[i] and Y==0):
        plt.scatter(x,Y,color="blue")
    elif(Y!=y_test[i] and Y==1):
        plt.scatter(x,Y,color="red")
    else:
         plt.scatter(x,Y,color="yellow")
    plt.xlabel("Ratio of length and suffix")
    plt.ylabel("Predicted value")
plt.show()

for i in range(0,len(y_pred4)):
    x=X_test[i][2]
    Y=y_pred4[i]
    if(Y==y_test[i] and Y==1):
        plt.scatter(x,Y,color="green")
    elif(Y==y_test[i] and Y==0):
        plt.scatter(x,Y,color="blue")
    elif(Y!=y_test[i] and Y==1):
        plt.scatter(x,Y,color="red")
    else:
         plt.scatter(x,Y,color="yellow")
    plt.xlabel("Ratio of length and suffix")
    plt.ylabel("Predicted value")
plt.show()

for i in range(0,len(y_pred5)):
    x=X_test[i][2]
    Y=y_pred5[i]
    if(Y==y_test[i] and Y==1):
        plt.scatter(x,Y,color="green")
    elif(Y==y_test[i] and Y==0):
        plt.scatter(x,Y,color="blue")
    elif(Y!=y_test[i] and Y==1):
        plt.scatter(x,Y,color="red")
    else:
         plt.scatter(x,Y,color="yellow")
    plt.xlabel("Ratio of length and suffix")
    plt.ylabel("Predicted value")
plt.show()

'''pickle.dump(classifier3,open('diseasemodel.pkl','wb'))
model=pickle.load(open('diseasemodel.pkl','rb'))'''

