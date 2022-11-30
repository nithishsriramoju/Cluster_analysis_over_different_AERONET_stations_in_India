# Cluster_analysis_over_different_AERONET_stations_in_India

### This project aims to perform aerosol cluster analysis using machine learning techniques 

##### Data from five different AERONET stations in India are used. Stations are Kanpur, Jaipur, Pune, Gandhi College, Gual Pahari.

![image](https://user-images.githubusercontent.com/116994415/204876053-bd0cb5f5-3f74-4320-8d8a-c0b3f175d15b.png=500x500)

<img src="[https://your-image-url.type](https://user-images.githubusercontent.com/116994415/204876053-bd0cb5f5-3f74-4320-8d8a-c0b3f175d15b.png)" width="100" height="100">


##### AERONET data downloaded from [AERONET webiste](https://aeronet.gsfc.nasa.gov/cgi-bin/webtool_aod_v3?stage=2&region=Asia&state=India) and Used data is level 2 Version 3 

##### All five stations data is appended and then screened and filtered using different techniques mentioned in [Omar,A.H.et al.,2005](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004JD004874) 

##### Data is standardized by transforming [min,max] to [0,1] for each feature of aerosol.

##### We performed different evaluation parametrics to optimize our no.of clusters. Silhouette Score, Calinski Harabasz Score, Davies–Bouldin index, and Distortion score.
![optimum_clusters](https://user-images.githubusercontent.com/116994415/204875502-769468ab-0947-4536-a899-7ba8bb625d6e.png=500x500)

##### Data is then clustered using K-Means algorithm and then compared with SOM clustering algorithm.

##### All clusters are then checked if they are clustered clearly by visualizing them in 2-D and 3-D using Principal Component Analysis (PCA).

##### We got 3 clusters named Absorbing mixed, Scattering mixed and Dust type aerosols.
