from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy import stats as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
sc = StandardScaler()
mm = MinMaxScaler()
ma = MaxAbsScaler()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
# df = pd.read_excel('Total_output2.xlsx')
df = pd.read_excel('Total_output2.xlsx')
data = df.to_numpy()
data_org = data
# data = st.stats.zscore(data)
# sc.fit(data)
# data = sc.transform(data)
#print(data)
print(np.shape(data))
# data = sc.fit_transform(data)  ######## standardization
data = mm.fit_transform(data)
# data = ma.fit_transform(data)
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

################### Silhouette_Score ####################
sil_score = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    sil_score = np.append(sil_score,silhouette_avg)
plt.subplot(2,2,1)
plt.plot(range_n_clusters,sil_score)
plt.title('Silhouette_Score')
plt.xlabel("Number of clusters")
# plt.show()

################## David Bouldin Score ############################
db_score = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(data)
    db_avg = davies_bouldin_score(data, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average Davies_Bouldin_score is :",
        db_avg,
    )
    db_score = np.append(db_score,db_avg)
plt.subplot(2,2,2)
plt.plot(range_n_clusters,db_score)
plt.title('David Bouldin Score')
plt.xlabel("Number of clusters")
# plt.show()

################ Calinski Harbasz Score #######################
ch_score=[]
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(data)
    ch_avg = calinski_harabasz_score(data, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average calinski_harabasz_score is :",
        ch_avg,
    )
    ch_score = np.append(ch_score,ch_avg)
plt.subplot(2,2,3)
plt.plot(range_n_clusters,ch_score)
plt.title('Calinski Harbasz Score')
plt.xlabel("Number of clusters")
# plt.show()

############### Distortion Score ##############################

sse = {}
for k in range(2, 12):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=10).fit(data)
    # data["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    
plt.subplot(2,2,4)
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.title('Elbow Method')
plt.show()
