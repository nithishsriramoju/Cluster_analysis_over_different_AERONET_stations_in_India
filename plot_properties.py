################### Plotting Data ##################
from cluster_analysis_main import *
import math as m
import pandas as pd 
import numpy as np
   
def concentration(a,r):
    C_c=a[0];C_f=a[1];r_c=a[2];r_f=a[3];std_c=a[4];std_f=a[5]
    L=[]
    for i in range(len(r)):
        L = np.append(L,(C_f/(m.sqrt(2*m.pi)*m.log(std_f)))*m.exp(-((m.log(r[i])-m.log(r_f))/(2*m.log(std_f)))**2)+(C_c/(m.sqrt(2*m.pi)*m.log(std_c)))*m.exp(-((m.log(r[i])-m.log(r_c))/(2*m.log(std_c)))**2))
    return L
   
def plot_data(som_cluster_centres):
    r = np.linspace(0.01,10,1000)
    x = [440,640,870,1020]
    plt.figure()
    plt.subplot(2,2,1)
    for i in range(no):
        
        plt.plot(x,som_cluster_centres[i,0:4],label='%s data' % i)
        plt.ylabel('g')
        plt.legend()
        plt.xlabel('Wavelength')
    plt.subplot(2,2,2)   
    for i in range(no):
        plt.plot(x,som_cluster_centres[i,4:8],label='%s data' % i)
        plt.ylabel('RRI')
        plt.xlabel('Wavelength')
        plt.legend()
    plt.subplot(2,2,3)   
    for i in range(no):
        plt.plot(x,som_cluster_centres[i,8:12],label='%s data' % i)
        plt.ylabel('IRI')
        plt.xlabel('Wavelength')
        plt.legend()
    plt.subplot(2,2,4)   
    for i in range(no):
        plt.plot(x,som_cluster_centres[i,12:16],label='%s data' % i)
        plt.ylabel('SSA')
        plt.xlabel('Wavelength')
        plt.legend()
    plt.show()
    plt.figure()
    for i in range(no):
        L = concentration(som_cluster_centres[i,16:22],r)
        plt.plot(r,-1*L,label='%s data' % i)
        plt.xscale('log')
        plt.xlabel('Radius in micrometer')
        plt.ylabel('Volume Concentration')
        plt.title('Particle Size Distribution')
        plt.legend()    
    plt.show()
# plot_data(som_cluster_centres)  

df = pd.read_excel('Total_Data_with_dates.xlsx')
data_dates = df.to_numpy()  


plot_data(k_means_cluster_centres)
    