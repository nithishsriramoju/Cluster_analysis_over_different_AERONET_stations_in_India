# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 21:40:31 2022

@author: NITHISH
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
sc = StandardScaler()
mm = MinMaxScaler()
ma = MaxAbsScaler()
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


########### Import Data from Excel File ####################################

df = pd.read_excel('Total_output2.xlsx')
data = df.to_numpy()

#from scipy import stats as st
# data = st.stats.zscore(data)
# sc.fit(data)
# data = sc.transform(data)
#print(data)
print(np.shape(data))
data_org = data

########### Standardizing Data #############################################

data = mm.fit_transform(data)
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
no = 3


########### Performing K-Means Cluster Analysis ############################

kmeans = KMeans(n_clusters=no, random_state=0).fit(data)
predictions_kmeans = kmeans.predict(data)

############ Principle Component Analysis & Visualization 3D ##################################

# pca = PCA(n_components=3)
# ll = pca.fit_transform(data)

# fig = plt.figure()
# ax = Axes3D(fig)

# for i in range(len(ll)):
#         if predictions_kmeans[i]==0:
#             ax.scatter3D(ll[i,0],ll[i,1],ll[i,2],c='r')
#         elif predictions_kmeans[i]==1:
#             ax.scatter3D(ll[i,0],ll[i,1],ll[i,2],c='b')
#         else:
#             ax.scatter3D(ll[i,0],ll[i,1],ll[i,2],c='g')

# plt.show()

############# Principle Component Analysis & Visualization 2D ##################################################

plt.figure()

pca = PCA(n_components=2)
ll = pca.fit_transform(data)

for i in range(len(ll)):
    print(i)
    if predictions_kmeans[i]==0:
        plt.scatter(ll[i,0],ll[i,1],c='r')
    elif predictions_kmeans[i]==1:
        plt.scatter(ll[i,0],ll[i,1],c='b')
    else:
        plt.scatter(ll[i,0],ll[i,1],c='g')
plt.show()