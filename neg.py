# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:16:34 2019

@author: abhisek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset=pd.read_csv("dst.tsv",delimiter='\t',quoting=3,encoding='latin-1')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,2243):
        review=re.sub('[^a-zA-Z]',' ',dataset['Record'][i])
        review=review.lower()
        review=review.split()
        ps=PorterStemmer()
        review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
        review=' '.join(review)
        corpus.append(review)
        
#createing the bag of words model
'''from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()'''
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#classifier............

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)

'''from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)'''

'''from sklearn.svm import SVC
classifier=SVC(gamma='auto')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)'''

'''from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)'''

'''Xinp=["jaundice","pneumonia"]
Xinp=vectorizer.transform(Xinp)
Xinp=Xinp.toarray()



print("After Conversion to dense matrix : ")
print(Xinp)


print(classifier.predict(Xinp))'''


'''pickle.dump(vectorizer,open('vector.pkl','wb'))
vec=pickle.load(open('vector.pkl','rb'))
Xinp=vec.transform(Xinp)
Xinp=Xinp.toarray()

pickle.dump(classifier,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict(Xinp))'''






'''accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])
print(accuracy)'''

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred1 = classifier.predict(X_test)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred2=classifier.predict(X_test)

from sklearn.svm import SVC
classifier=SVC(gamma='auto')
classifier.fit(X_train,y_train)
y_pred3=classifier.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)
y_pred4=classifier.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred5=classifier.predict(X_test)

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



