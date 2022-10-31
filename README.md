# Cluster_analysis_over_different_AERONET_stations_in_India

##### Data from five different AERONET stations in India are used. Stations are Kanpur, Jaipur, Pune, Gandhi College, Gual Pahari.

##### AERONET data downloaded from https://aeronet.gsfc.nasa.gov/cgi-bin/webtool_aod_v3?stage=2&region=Asia&state=India and Used data is level 2 Version 3 

##### All five stations data is appended and then screened and filtered using different techniques mentioned in Omar,A.H.et.al.,2005 

##### Data is standardized by transforming [min,max] to [0,1] for each feature of aerosol.

##### We performed different evaluation parametrics to optimize our no.of clusters. Silhouette Score, Calinski Harabasz Score, Davies–Bouldin index, and Distortion score.

##### Data is then clustered using K-Means algorithm and then validated with SOM clustering algorithm.

##### All clusters are then checked if they are clustered clearly by visualizing them in 2-D and 3-D using Principal Component Analysis (PCA).

##### We got 3 clusters named Absorbing mixed, Scattering mixed and Dust type aerosols.