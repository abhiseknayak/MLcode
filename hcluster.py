# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:34:13 2019

@author: abhisek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv("worderful.tsv",delimiter='\t',quoting=3,encoding='latin-1')

X=dataset.iloc[:,1:5].values

import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title("Dendrogram")
plt.xlabel("features")
plt.ylabel("Euclidean Distances")
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')

y_hc=hc.fit_predict(X)

group1=[]
group2=[]
for i in range(len(dataset)):
    if y_hc[i]==1:
        group1.append(dataset.iloc[i,0])
    else:
        group2.append(dataset.iloc[i,0])
        