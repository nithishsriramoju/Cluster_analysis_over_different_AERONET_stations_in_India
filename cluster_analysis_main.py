from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
sc = StandardScaler()
mm = MinMaxScaler()
ma = MaxAbsScaler()
from sklearn_som.som import SOM
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn import metrics 


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
# data = sc.fit_transform(data)
data = mm.fit_transform(data)
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
no = 3

########### Performing Self Organizing Mapping CLuster Analysis ############
data_som = SOM(m=no, n=1, dim=22, random_state=0)
data_som.fit(data)
predictions_som = data_som.predict(data)

########### Performing K-Means Cluster Analysis ############################

kmeans = KMeans(n_clusters=no, random_state=0).fit(data)
predictions_kmeans = kmeans.predict(data)

print(np.size(predictions_kmeans),np.size(predictions_som))


######### Extracting Cluster Centres for K-Means #########################

k_means_cluster_centres=[]
for j in range(no):
    n=0
    sum_asy440=0;sum_asy675=0;sum_asy870=0;sum_asy1020=0
    sum_rri440=0;sum_rri675=0;sum_rri870=0;sum_rri1020=0
    sum_iri440=0;sum_iri675=0;sum_iri870=0;sum_iri1020=0
    sum_ssa440=0;sum_ssa675=0;sum_ssa870=0;sum_ssa1020=0
    sum_volcc=0;sum_volcf=0
    sum_vmrc=0;sum_vmrf=0
    sum_stdc=0;sum_stdf=0
    for i in range(len(predictions_kmeans)):
        if predictions_kmeans[i]==j:
            sum_asy440=data_org[i,0]+sum_asy440
            sum_asy675=data_org[i,1]+sum_asy675
            sum_asy870=data_org[i,2]+sum_asy870
            sum_asy1020=data_org[i,3]+sum_asy1020
            sum_rri440=data_org[i,4]+sum_rri440
            sum_rri675=data_org[i,5]+sum_rri675
            sum_rri870=data_org[i,6]+sum_rri870
            sum_rri1020=data_org[i,7]+sum_rri1020
            sum_iri440=data_org[i,8]+sum_iri440
            sum_iri675=data_org[i,9]+sum_iri675
            sum_iri870=data_org[i,10]+sum_iri870
            sum_iri1020=data_org[i,11]+sum_iri1020
            sum_ssa440=data_org[i,12]+sum_ssa440
            sum_ssa675=data_org[i,13]+sum_ssa675
            sum_ssa870=data_org[i,14]+sum_ssa870
            sum_ssa1020=data_org[i,15]+sum_ssa1020
            sum_volcc=data_org[i,16]+sum_volcc
            sum_volcf=data_org[i,17]+sum_volcf
            sum_vmrc=data_org[i,18]+sum_vmrc
            sum_vmrf=data_org[i,19]+sum_vmrf
            sum_stdc=data_org[i,20]+sum_stdc
            sum_stdf=data_org[i,21]+sum_stdf
            n=n+1
    clust = [sum_asy440/n,sum_asy675/n,sum_asy870/n,sum_asy1020/n,sum_rri440/n,sum_rri675/n,
          sum_rri870/n,sum_rri1020/n,sum_iri440/n,sum_iri675/n,sum_iri870/n,sum_iri1020/n,
          sum_ssa440/n,sum_ssa675/n,sum_ssa870/n,sum_ssa1020/n,sum_volcc/n,sum_volcf/n,sum_vmrc/n,
          sum_vmrf/n,sum_stdc/n,sum_stdf/n]   
    if len(k_means_cluster_centres)==0:
        k_means_cluster_centres=clust
    else:
        k_means_cluster_centres = np.vstack((k_means_cluster_centres,clust))


som_cluster_centres =[]
for j in range(no):
    n=0
    sum_asy440=0;sum_asy675=0;sum_asy870=0;sum_asy1020=0
    sum_rri440=0;sum_rri675=0;sum_rri870=0;sum_rri1020=0
    sum_iri440=0;sum_iri675=0;sum_iri870=0;sum_iri1020=0
    sum_ssa440=0;sum_ssa675=0;sum_ssa870=0;sum_ssa1020=0
    sum_volcc=0;sum_volcf=0
    sum_vmrc=0;sum_vmrf=0
    sum_stdc=0;sum_stdf=0
    for i in range(len(predictions_som)):
        if predictions_som[i]==j:
            sum_asy440=data_org[i,0]+sum_asy440
            sum_asy675=data_org[i,1]+sum_asy675
            sum_asy870=data_org[i,2]+sum_asy870
            sum_asy1020=data_org[i,3]+sum_asy1020
            sum_rri440=data_org[i,4]+sum_rri440
            sum_rri675=data_org[i,5]+sum_rri675
            sum_rri870=data_org[i,6]+sum_rri870
            sum_rri1020=data_org[i,7]+sum_rri1020
            sum_iri440=data_org[i,8]+sum_iri440
            sum_iri675=data_org[i,9]+sum_iri675
            sum_iri870=data_org[i,10]+sum_iri870
            sum_iri1020=data_org[i,11]+sum_iri1020
            sum_ssa440=data_org[i,12]+sum_ssa440
            sum_ssa675=data_org[i,13]+sum_ssa675
            sum_ssa870=data_org[i,14]+sum_ssa870
            sum_ssa1020=data_org[i,15]+sum_ssa1020
            sum_volcc=data_org[i,16]+sum_volcc
            sum_volcf=data_org[i,17]+sum_volcf
            sum_vmrc=data_org[i,18]+sum_vmrc
            sum_vmrf=data_org[i,19]+sum_vmrf
            sum_stdc=data_org[i,20]+sum_stdc
            sum_stdf=data_org[i,21]+sum_stdf
            n=n+1
    clust = [sum_asy440/n,sum_asy675/n,sum_asy870/n,sum_asy1020/n,sum_rri440/n,sum_rri675/n,
              sum_rri870/n,sum_rri1020/n,sum_iri440/n,sum_iri675/n,sum_iri870/n,sum_iri1020/n,
              sum_ssa440/n,sum_ssa675/n,sum_ssa870/n,sum_ssa1020/n,sum_volcc/n,sum_volcf/n,sum_vmrc/n,
              sum_vmrf/n,sum_stdc/n,sum_stdf/n]   
    if len(som_cluster_centres)==0:
        som_cluster_centres=clust
    else:
        som_cluster_centres = np.vstack((som_cluster_centres,clust))

if __name__ =='__main__':
    ########## Comparision of K-Means analysis and SOM analysis ################
    # lol=[]
    # for i in range(len(predictions_kmeans)):
    #     lol = np.append(lol,(10*predictions_kmeans[i]+predictions_som[i]))


    # jk = plt.hist(lol,23)
    # plt.xlabel('Combined Cluster Label from SOM and KMeans')
    # plt.ylabel('Frequency of Occurence')
    # # plt.xticks(range(22))
    # # plt.xlim([-1, 22])
    # plt.show()

    ######### Confusion Matrix ##################################################

    confusion_matrix = metrics.confusion_matrix(predictions_som,predictions_kmeans)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2])
    cm_display.plot()
    plt.ylabel('SOM Algorithm')
    plt.xlabel('KMeans Algorithm')
    plt.show()

