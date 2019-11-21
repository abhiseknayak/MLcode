# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:54:56 2019

@author: abhisek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("worderful.tsv",delimiter='\t',quoting=3,encoding='latin-1')

X=dataset.iloc[:,1:5].values

from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans=KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)



group1=[]
group2=[]
for i in range(len(dataset)):
    if y_kmeans[i]==1:
        group1.append(dataset.iloc[i,0])
    else:
        group2.append(dataset.iloc[i,0])
        

    