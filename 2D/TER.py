# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:04:05 2019

@author: Moli Oujia
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi#, voronoi_plot_2d
from numpy import log10

#from Tools import voronoi_cell_area, lis_point, density, moyenne, ecartype, histogramme#, met_log10, variance
from Tools import *


from numpy import random as npr  
from scipy import stats as st  

import time

#fichier="pcl-021000_z-4eta.txt"
#fichier="N512_st1\pcl-030000_z-4eta.txt"
#fichier="N512_st02\pcl-030000_z-4eta.txt"
#fichier="points_2d.txt"
#couleur='r'
#"N512_st005\pcl-030000_z-4eta.txt"

#fichier="N1024_st05\pcl-0115000_z-4eta.txt"

def Fonction(fichier, couleur, legende, nom_Vor):

    point=lis_point(fichier)

    m1=np.transpose(create_border(point))[0]
    m2=np.transpose(point)[0]
    l1=m1.tolist()
    l2=m2.tolist()
    
    Q1=l2+l1
    
    m3=np.transpose(create_border(point))[1]
    m4=np.transpose(point)[1]
    l3=m3.tolist()
    l4=m4.tolist()
    
    Q2=l4+l3
    
    Po=[]
    for i in range(len(Q1)):
        Po.append([Q1[i],Q2[i]])
    
    vor_2d_0 = Voronoi(Po)
    #voronoi_plot_2d(vor_2d_0)

    AR=[]
    AR=voronoi_cell_area_test(vor_2d_0,len(point))
    
    #My_hist_vor(AR, legende, couleur,2)
    #ecrit_data_binaire(nom_Vor, AR)
    
    return AR
    
#point = np.random.random_sample((700000,2))
#My_hist_vor( np.array(AR) / moyenne(np.array(AR)  ), 'Random', 'black',2, ',')

#My_hist_vor(lis_data_binaire("2D_N512-st1-030000-vor"),'1','r',2)


Area204=Fonction("N512_st5\pcl-030000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
My_hist_vor(Area204,'Re=204','b',2)
Area328=Fonction("N1024_st5\pcl-0115000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")
My_hist_vor(Area328,'Re=328','r',2)



AR_5_1=Fonction("N512_st5\pcl-030000_z-4eta.txt", 'pink', 'St=5', "2D_N512-st5-030000-vor")
AR_2_1=Fonction("N512_st2\pcl-030000_z-4eta.txt", 'y', 'St=2', "2D_N512-st2-030000-vor")
AR_1_1=Fonction("N512_st1\pcl-030000_z-4eta.txt", 'r', 'St=1', "2D_N512-st1-030000-vor")
AR_05_1=Fonction("N512_st05\pcl-030000_z-4eta.txt", 'c', 'St=0.5', "2D_N512-st05-030000-vor")
AR_02_1=Fonction("N512_st02\pcl-030000_z-4eta.txt", 'g', 'St=0.2', "2D_N512-st02-030000-vor")
AR_01_1=Fonction("N512_st01\pcl-030000_z-4eta.txt", 'b', 'St=0.1', "2D_N512-st01-030000-vor")
AR_005_1=Fonction("N512_st005\pcl-030000_z-4eta.txt", 'purple', 'St=0.05', "2D_N512-st005-030000-vor")

AR_5_2=Fonction("N512_st5\pcl-028000_z-4eta.txt", 'pink', 'St=5', "2D_N512-st5-030000-vor")
AR_2_2=Fonction("N512_st2\pcl-028000_z-4eta.txt", 'y', 'St=2', "2D_N512-st2-030000-vor")
AR_1_2=Fonction("N512_st1\pcl-028000_z-4eta.txt", 'r', 'St=1', "2D_N512-st1-030000-vor")
AR_05_2=Fonction("N512_st05\pcl-028000_z-4eta.txt", 'c', 'St=0.5', "2D_N512-st05-030000-vor")
AR_02_2=Fonction("N512_st02\pcl-028000_z-4eta.txt", 'g', 'St=0.2', "2D_N512-st02-030000-vor")
AR_01_2=Fonction("N512_st01\pcl-028000_z-4eta.txt", 'b', 'St=0.1', "2D_N512-st01-030000-vor")
AR_005_2=Fonction("N512_st005\pcl-028000_z-4eta.txt", 'purple', 'St=0.05', "2D_N512-st005-030000-vor")

AR_5_3=Fonction("N512_st5\pcl-025000_z-4eta.txt", 'pink', 'St=5', "2D_N512-st5-030000-vor")
AR_2_3=Fonction("N512_st2\pcl-025000_z-4eta.txt", 'y', 'St=2', "2D_N512-st2-030000-vor")
AR_1_3=Fonction("N512_st1\pcl-025000_z-4eta.txt", 'r', 'St=1', "2D_N512-st1-030000-vor")
AR_05_3=Fonction("N512_st05\pcl-025000_z-4eta.txt", 'c', 'St=0.5', "2D_N512-st05-030000-vor")
AR_02_3=Fonction("N512_st02\pcl-025000_z-4eta.txt", 'g', 'St=0.2', "2D_N512-st02-030000-vor")
AR_01_3=Fonction("N512_st01\pcl-025000_z-4eta.txt", 'b', 'St=0.1', "2D_N512-st01-030000-vor")
AR_005_3=Fonction("N512_st005\pcl-025000_z-4eta.txt", 'purple', 'St=0.05', "2D_N512-st005-030000-vor")



My_hist_vor( np.array(AR_5_1) / moyenne(np.array(AR_5_1)  ), 'St=5', 'g',2)
My_hist_vor( np.array(AR_5_2) / moyenne(np.array(AR_5_2)  ), 'St=5', 'r',2)
My_hist_vor( np.array(AR_5_3) / moyenne(np.array(AR_5_3)  ), 'St=5', 'b',2)
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St5_Vor.eps', bbox_inches='tight', quality=100, format='eps')

My_hist_vor( np.array(AR_1_1) / moyenne(np.array(AR_1_1)  ), 'St=1', 'g',3)
My_hist_vor( np.array(AR_1_2) / moyenne(np.array(AR_1_2)  ), 'St=1', 'r',3)
My_hist_vor( np.array(AR_1_3) / moyenne(np.array(AR_1_3)  ), 'St=1', 'b',3)
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St1_Vor.eps', bbox_inches='tight', quality=100, format='eps')

My_hist_vor( np.array(AR_02_1) / moyenne(np.array(AR_02_1)  ), 'St=0.2', 'g',4)
My_hist_vor( np.array(AR_02_2) / moyenne(np.array(AR_02_2)  ), 'St=0.2', 'r',4)
My_hist_vor( np.array(AR_02_3) / moyenne(np.array(AR_02_3)  ), 'St=0.2', 'b',4)
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St02_Vor.eps', bbox_inches='tight', quality=100, format='eps')

My_hist_vor( np.array(AR_005_1) / moyenne(np.array(AR_005_1)  ), 'St=0.05', 'g',5)
My_hist_vor( np.array(AR_005_2) / moyenne(np.array(AR_005_2)  ), 'St=0.05', 'r',5)
My_hist_vor( np.array(AR_005_3) / moyenne(np.array(AR_005_3)  ), 'St=0.05', 'b',5)
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St005_Vor.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')



My_hist_vor( np.array(AR_5_1.tolist()+AR_5_2.tolist()+AR_5_3.tolist()) / moyenne(np.array(AR_5_1.tolist()+AR_5_2.tolist()+AR_5_3.tolist())  ), 'St=5', 'pink',2, 'x')
My_hist_vor( np.array(AR_2_1.tolist()+AR_2_2.tolist()+AR_2_3.tolist()) / moyenne( np.array(AR_2_1.tolist()+AR_2_2.tolist()+AR_2_3.tolist()) ), 'St=2', 'y',2,'D')
My_hist_vor( np.array(AR_1_1.tolist()+AR_1_2.tolist()+AR_1_3.tolist()) / moyenne( np.array(AR_1_1.tolist()+AR_1_2.tolist()+AR_1_3.tolist()) ), 'St=1', 'r',2,'+')
My_hist_vor( np.array(AR_05_1.tolist()+AR_05_2.tolist()+AR_05_3.tolist()) / moyenne( np.array(AR_05_1.tolist()+AR_05_2.tolist()+AR_05_3.tolist()) ), 'St=0.5', 'c',2,',')
My_hist_vor( np.array(AR_02_1.tolist()+AR_02_2.tolist()+AR_02_3.tolist()) / moyenne( np.array(AR_02_1.tolist()+AR_02_2.tolist()+AR_02_3.tolist()) ), 'St=0.2', 'g',2,'o')
My_hist_vor( np.array(AR_01_1.tolist()+AR_01_2.tolist()+AR_01_3.tolist()) / moyenne( np.array(AR_01_1.tolist()+AR_01_2.tolist()+AR_01_3.tolist()) ), 'St=0.1', 'b',2,'^')
My_hist_vor( np.array(AR_005_1.tolist()+AR_005_2.tolist()+AR_005_3.tolist()) / moyenne( np.array(AR_005_1.tolist()+AR_005_2.tolist()+AR_005_3.tolist()) ), 'St=0.05', 'purple',2,'s')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('StAll_Vor.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')


"""
My_hist_vor( np.array(AR_5_1.tolist()+AR_5_2.tolist()+AR_5_3.tolist()) / ( 2*np.pi/np.array(len(AR_5_1.tolist())+len(AR_5_2.tolist())+len(AR_5_3.tolist()))  ), 'St=5', 'pink',2, 'x')
My_hist_vor( np.array(AR_2_1.tolist()+AR_2_2.tolist()+AR_2_3.tolist()) / ( 2*np.pi/np.array(len(AR_2_1.tolist())+len(AR_2_2.tolist())+len(AR_2_3.tolist())) ), 'St=2', 'y',2,'*')
My_hist_vor( np.array(AR_1_1.tolist()+AR_1_2.tolist()+AR_1_3.tolist()) / ( 2*np.pi/np.array(len(AR_1_1.tolist())+len(AR_1_2.tolist())+len(AR_1_3.tolist())) ), 'St=1', 'r',2,'+')
My_hist_vor( np.array(AR_05_1.tolist()+AR_05_2.tolist()+AR_05_3.tolist()) / ( 2*np.pi/np.array(len(AR_05_1.tolist())+len(AR_05_2.tolist())+len(AR_05_3.tolist())) ), 'St=0.5', 'c',2,',')
My_hist_vor( np.array(AR_02_1.tolist()+AR_02_2.tolist()+AR_02_3.tolist()) / ( 2*np.pi/np.array(len(AR_02_1.tolist())+len(AR_02_2.tolist())+len(AR_02_3.tolist())) ), 'St=0.2', 'g',2,'o')
My_hist_vor( np.array(AR_01_1.tolist()+AR_01_2.tolist()+AR_01_3.tolist()) / ( 2*np.pi/np.array(len(AR_01_1.tolist())+len(AR_01_2.tolist())+len(AR_01_3.tolist())) ), 'St=0.1', 'b',2,'^')
My_hist_vor( np.array(AR_005_1.tolist()+AR_005_2.tolist()+AR_005_3.tolist()) / ( 2*np.pi/np.array(len(AR_005_1.tolist())+len(AR_005_2.tolist())+len(AR_005_3.tolist())) ), 'St=0.05', 'purple',2,'3')
plt.xlabel('Density')
plt.ylabel('Frequency')

My_hist_vor( np.array(AR_5_1.tolist()+AR_5_2.tolist()+AR_5_3.tolist()) , 'St=5', 'pink',2, 'x')
My_hist_vor( np.array(AR_2_1.tolist()+AR_2_2.tolist()+AR_2_3.tolist()), 'St=2', 'y',2,'*')
My_hist_vor( np.array(AR_1_1.tolist()+AR_1_2.tolist()+AR_1_3.tolist()) , 'St=1', 'r',2,'+')
My_hist_vor( np.array(AR_05_1.tolist()+AR_05_2.tolist()+AR_05_3.tolist()) , 'St=0.5', 'c',2,',')
My_hist_vor( np.array(AR_02_1.tolist()+AR_02_2.tolist()+AR_02_3.tolist()) , 'St=0.2', 'g',2,'o')
My_hist_vor( np.array(AR_01_1.tolist()+AR_01_2.tolist()+AR_01_3.tolist()) , 'St=0.1', 'b',2,'^')
My_hist_vor( np.array(AR_005_1.tolist()+AR_005_2.tolist()+AR_005_3.tolist()) , 'St=0.05', 'purple',2,'3')
plt.xlabel('Density')
plt.ylabel('Frequency')
"""




My_hist_vor( np.array(AR_5_1.tolist()+AR_5_2.tolist()+AR_5_3.tolist()) / moyenne(np.array(AR_5_1.tolist()+AR_5_2.tolist()+AR_5_3.tolist())  ), 'St=5', 'pink',3)
data=np.array(AR_5_1.tolist()+AR_5_2.tolist()+AR_5_3.tolist()) / moyenne(np.array(AR_5_1.tolist()+AR_5_2.tolist()+AR_5_3.tolist()))
mu, sigma = moyenne(np.log(data))+ ecartype(np.log(data))**2, ecartype(np.log(data))
x = np.logspace(np.log10(0.1),np.log10(6),10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, label='Log-normal law', c='pink')
plt.legend()
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St5_Vor_normal.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

My_hist_vor( np.array(AR_1_1.tolist()+AR_1_2.tolist()+AR_1_3.tolist()) / moyenne( np.array(AR_1_1.tolist()+AR_1_2.tolist()+AR_1_3.tolist()) ), 'St=1', 'r',4)
data=np.array(AR_1_1.tolist()+AR_1_2.tolist()+AR_1_3.tolist()) / moyenne( np.array(AR_1_1.tolist()+AR_1_2.tolist()+AR_1_3.tolist())  )
mu, sigma = moyenne(np.log(data))+ ecartype(np.log(data))**2, ecartype(np.log(data))
x = np.logspace(np.log10(0.05),np.log10(10),10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, label='Log-normal law', c='r')
plt.legend()
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St1_Vor_normal.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

My_hist_vor( np.array(AR_02_1.tolist()+AR_02_2.tolist()+AR_02_3.tolist()) / moyenne( np.array(AR_02_1.tolist()+AR_02_2.tolist()+AR_02_3.tolist()) ), 'St=0.2', 'g',5)
data=np.array(AR_02_1.tolist()+AR_02_2.tolist()+AR_02_3.tolist()) / moyenne( np.array(AR_02_1.tolist()+AR_02_2.tolist()+AR_02_3.tolist()) )
mu, sigma = moyenne(np.log(data))+ ecartype(np.log(data))**2, ecartype(np.log(data))
x = np.logspace(np.log10(0.15),np.log10(5),10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, label='Log-normal law', c='g')
plt.legend()
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St02_Vor_normal.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

My_hist_vor( np.array(AR_005_1.tolist()+AR_005_2.tolist()+AR_005_3.tolist()) / moyenne( np.array(AR_005_1.tolist()+AR_005_2.tolist()+AR_005_3.tolist()) ), 'St=0.05', 'purple',6)
data=np.array(AR_005_1.tolist()+AR_005_2.tolist()+AR_005_3.tolist()) / moyenne( np.array(AR_005_1.tolist()+AR_005_2.tolist()+AR_005_3.tolist())  )
mu, sigma = moyenne(np.log(data))+ ecartype(np.log(data))**2, ecartype(np.log(data))
x = np.logspace(np.log10(0.3),np.log10(3),10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, label='Log-normal law', c='purple')
plt.legend()
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St005_Vor_normal.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')










Area2041=Fonction("N512_st05\pcl-025000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area2042=Fonction("N512_st05\pcl-028000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area2043=Fonction("N512_st05\pcl-030000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area3281=Fonction("N1024_st05\pcl-0115000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")
Area3282=Fonction("N1024_st05\pcl-0125000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")
Area3283=Fonction("N1024_st05\pcl-0135000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")

My_hist_vor( np.array(Area2041.tolist()+Area2042.tolist()+Area2043.tolist()) / moyenne( np.array(Area2041.tolist()+Area2042.tolist()+Area2043.tolist()) ), 'Re=204', 'b',5)
My_hist_vor( np.array(Area3281.tolist()+Area3282.tolist()+Area3283.tolist()) / moyenne( np.array(Area3281.tolist()+Area3282.tolist()+Area3283.tolist()) ), 'Re=328', 'r',5)
plt.title('St = 0.5')
plt.savefig('St05_Vor_Re.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')


Area2041=Fonction("N512_st1\pcl-025000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area2042=Fonction("N512_st1\pcl-028000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area2043=Fonction("N512_st1\pcl-030000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area3281=Fonction("N1024_st1\pcl-0115000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")
Area3282=Fonction("N1024_st1\pcl-0125000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")
Area3283=Fonction("N1024_st1\pcl-0135000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")

My_hist_vor( np.array(Area2041.tolist()+Area2042.tolist()+Area2043.tolist()) / moyenne( np.array(Area2041.tolist()+Area2042.tolist()+Area2043.tolist()) ), 'Re=204', 'b',5)
My_hist_vor( np.array(Area3281.tolist()+Area3282.tolist()+Area3283.tolist()) / moyenne( np.array(Area3281.tolist()+Area3282.tolist()+Area3283.tolist()) ), 'Re=328', 'r',5)
plt.title('St = 1')
plt.savefig('St1_Vor_Re.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')


Area2041=Fonction("N512_st2\pcl-025000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area2042=Fonction("N512_st2\pcl-028000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area2043=Fonction("N512_st2\pcl-030000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area3281=Fonction("N1024_st2\pcl-0115000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")
Area3282=Fonction("N1024_st2\pcl-0125000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")
Area3283=Fonction("N1024_st2\pcl-0135000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")

My_hist_vor( np.array(Area2041.tolist()+Area2042.tolist()+Area2043.tolist()) / moyenne( np.array(Area2041.tolist()+Area2042.tolist()+Area2043.tolist()) ), 'Re=204', 'b',5)
My_hist_vor( np.array(Area3281.tolist()+Area3282.tolist()+Area3283.tolist()) / moyenne( np.array(Area3281.tolist()+Area3282.tolist()+Area3283.tolist()) ), 'Re=328', 'r',5)
plt.title('St = 2')
plt.savefig('St2_Vor_Re.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')


Area2041=Fonction("N512_st5\pcl-025000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area2042=Fonction("N512_st5\pcl-028000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area2043=Fonction("N512_st5\pcl-030000_z-4eta.txt", 'b', 'Re=204', "2D_N512-st1-030000-vor")
Area3281=Fonction("N1024_st5\pcl-0115000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")
Area3282=Fonction("N1024_st5\pcl-0125000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")
Area3283=Fonction("N1024_st5\pcl-0135000_z-4eta.txt", 'r', 'Re=328', "2D_N512-st5-030000-vor")

My_hist_vor( np.array(Area2041.tolist()+Area2042.tolist()+Area2043.tolist()) / moyenne( np.array(Area2041.tolist()+Area2042.tolist()+Area2043.tolist()) ), 'Re=204', 'b',5)
My_hist_vor( np.array(Area3281.tolist()+Area3282.tolist()+Area3283.tolist()) / moyenne( np.array(Area3281.tolist()+Area3282.tolist()+Area3283.tolist()) ), 'Re=328', 'r',5)
plt.title('St = 5')
plt.savefig('St5_Vor_Re.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')













P1 = np.random.random_sample((10000000,2))*2*np.pi
D1=density(P1,1000)
My_hist_dens_poisson(D1, 'Random', 'r', 1)
loi_de_Poisson(10,24)
plt.savefig('Rand_Poisson_10.eps', bbox_inches='tight', quality=100, format='eps')


P2 = np.random.random_sample((12150000,2))*2*np.pi
D2=density(P2,900)
My_hist_dens_poisson(D2, 'Random', 'r', 3)
loi_de_Poisson(15,37)
plt.savefig('Rand_Poisson_20.eps', bbox_inches='tight', quality=100, format='eps')













lis_point("N512_st05\pcl-030000_z-4eta.txt")
lis_point("N1024_st05\pcl-0115000_z-4eta.txt")

lis_point("N512_st1\pcl-030000_z-4eta.txt")
lis_point("N1024_st1\pcl-0115000_z-4eta.txt")

lis_point("N512_st2\pcl-030000_z-4eta.txt")
lis_point("N1024_st2\pcl-0115000_z-4eta.txt")

lis_point("N512_st5\pcl-030000_z-4eta.txt")
lis_point("N1024_st5\pcl-0115000_z-4eta.txt")



#My_hist_dens(density(lis_point("N512_st05\pcl-030000_z-4eta.txt").tolist()+lis_point("N512_st05\pcl-028000_z-4eta.txt").tolist()+lis_point("N512_st05\pcl-025000_z-4eta.txt").tolist(),100), 'Re=204', 'b', 2)


#!!!!!!!!!!!!!!!!!!!!!!!!!! a refaire !!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
My_hist_dens(density(lis_point("N512_st5\pcl-030000_z-4eta.txt"),100), 'Re=204', 'b', 2,'*')
My_hist_dens(density(lis_point("N1024_st5\pcl-0115000_z-4eta.txt"),200), 'Re=328', 'r', 2,'^')
plt.title('St = 5')

My_hist_dens(density(lis_point("N512_st2\pcl-030000_z-4eta.txt"),100), 'Re=204', 'b', 3,'*')
My_hist_dens(density(lis_point("N1024_st2\pcl-0115000_z-4eta.txt"),200), 'Re=328', 'r', 3,'^')
plt.title('St = 2')

My_hist_dens(density(lis_point("N512_st1\pcl-030000_z-4eta.txt"),100), 'Re=204', 'b', 4,'*')
My_hist_dens(density(lis_point("N1024_st1\pcl-0115000_z-4eta.txt"),200), 'Re=328', 'r', 4,'^')
plt.title('St = 1')

My_hist_dens(density(lis_point("N512_st05\pcl-030000_z-4eta.txt"),100), 'Re=204', 'b', 5,'*')
My_hist_dens(density(lis_point("N1024_st05\pcl-0115000_z-4eta.txt"),200), 'Re=328', 'r', 5,'^')
plt.title('St = 0.5')

plt.xscale('log')
plt.yscale('log')
"""
#!!!!!!!!!!!!!!!!!!!!!!!!!! a refaire !!!!!!!!!!!!!!!!!!!!!!!!!!!


plt.scatter(point[0:int(len(p)/2)][:,0],point[0:int(len(p)/2)][:,1],s=0.05,c='black')


My_hist_dens(density(lis_point("N512_st5\pcl-030000_z-4eta.txt")[0:int(len(lis_point("N512_st5\pcl-030000_z-4eta.txt"))/2)],100), 'Re=204', 'b', 2,'*')
My_hist_dens(density(lis_point("N1024_st5\pcl-0115000_z-4eta.txt"),200), 'Re=328', 'r', 2,'^')
plt.title('St = 5')
plt.savefig('St5_dens_Re.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

My_hist_dens(density(lis_point("N512_st2\pcl-030000_z-4eta.txt")[0:int(len(lis_point("N512_st2\pcl-030000_z-4eta.txt"))/2)],100), 'Re=204', 'b', 3,'*')
My_hist_dens(density(lis_point("N1024_st2\pcl-0115000_z-4eta.txt"),200), 'Re=328', 'r', 3,'^')
plt.title('St = 2')
plt.savefig('St2_dens_Re.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

My_hist_dens(density(lis_point("N512_st1\pcl-030000_z-4eta.txt")[0:int(len(lis_point("N512_st1\pcl-030000_z-4eta.txt"))/2)],100), 'Re=204', 'b', 4,'*')
My_hist_dens(density(lis_point("N1024_st1\pcl-0115000_z-4eta.txt"),200), 'Re=328', 'r', 4,'^')
plt.title('St = 1')
plt.savefig('St1_dens_Re.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

My_hist_dens(density(lis_point("N512_st05\pcl-030000_z-4eta.txt")[0:int(len(lis_point("N512_st05\pcl-030000_z-4eta.txt"))/2)],100), 'Re=204', 'b', 5,'*')
My_hist_dens(density(lis_point("N1024_st05\pcl-0115000_z-4eta.txt"),200), 'Re=328', 'r', 5,'^')
plt.title('St = 0.5')
plt.savefig('St05_dens_Re.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

plt.xscale('log')
plt.yscale('log')



My_hist_dens(density(lis_point("N512_st5\pcl-030000_z-4eta.txt"),100) + density(lis_point("N512_st5\pcl-028000_z-4eta.txt"),100) + density(lis_point("N512_st5\pcl-025000_z-4eta.txt"),100) + density(lis_point("N512_st5\pcl-023000_z-4eta.txt"),100) + density(lis_point("N512_st5\pcl-021000_z-4eta.txt"),100), 'St = 5', 'pink', 2,'x')
My_hist_dens(density(lis_point("N512_st2\pcl-030000_z-4eta.txt"),100) + density(lis_point("N512_st2\pcl-028000_z-4eta.txt"),100) + density(lis_point("N512_st2\pcl-025000_z-4eta.txt"),100) + density(lis_point("N512_st2\pcl-023000_z-4eta.txt"),100) + density(lis_point("N512_st2\pcl-021000_z-4eta.txt"),100), 'St = 2', 'y', 2, '*')
My_hist_dens(density(lis_point("N512_st1\pcl-030000_z-4eta.txt"),100) + density(lis_point("N512_st1\pcl-028000_z-4eta.txt"),100) + density(lis_point("N512_st1\pcl-025000_z-4eta.txt"),100) + density(lis_point("N512_st1\pcl-023000_z-4eta.txt"),100) + density(lis_point("N512_st1\pcl-021000_z-4eta.txt"),100), 'St = 1', 'r', 2, '+')
My_hist_dens(density(lis_point("N512_st05\pcl-030000_z-4eta.txt"),100) + density(lis_point("N512_st05\pcl-028000_z-4eta.txt"),100) + density(lis_point("N512_st05\pcl-025000_z-4eta.txt"),100) + density(lis_point("N512_st05\pcl-023000_z-4eta.txt"),100) + density(lis_point("N512_st05\pcl-021000_z-4eta.txt"),100), 'St = 0.5', 'c', 2, ',')
My_hist_dens(density(lis_point("N512_st02\pcl-030000_z-4eta.txt"),100) + density(lis_point("N512_st02\pcl-028000_z-4eta.txt"),100) + density(lis_point("N512_st02\pcl-025000_z-4eta.txt"),100) + density(lis_point("N512_st02\pcl-023000_z-4eta.txt"),100) + density(lis_point("N512_st02\pcl-021000_z-4eta.txt"),100), 'St = 0.2', 'g', 2,'o')
My_hist_dens(density(lis_point("N512_st01\pcl-030000_z-4eta.txt"),100) + density(lis_point("N512_st01\pcl-028000_z-4eta.txt"),100) + density(lis_point("N512_st01\pcl-025000_z-4eta.txt"),100) + density(lis_point("N512_st01\pcl-023000_z-4eta.txt"),100) + density(lis_point("N512_st01\pcl-021000_z-4eta.txt"),100), 'St = 0.1', 'b', 2,'^')
My_hist_dens(density(lis_point("N512_st005\pcl-030000_z-4eta.txt"),100) + density(lis_point("N512_st005\pcl-028000_z-4eta.txt"),100) + density(lis_point("N512_st005\pcl-025000_z-4eta.txt"),100) + density(lis_point("N512_st005\pcl-023000_z-4eta.txt"),100) + density(lis_point("N512_st005\pcl-021000_z-4eta.txt"),100), 'St = 0.05', 'purple', 2, '|')
loi_de_Poisson_conti(7,19)
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('StAll_dens.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')




My_hist_dens(density(lis_point("N512_st5\pcl-030000_z-4eta.txt"),100), 'St = 5', 'b', 3,'x')
My_hist_dens(density(lis_point("N512_st5\pcl-028000_z-4eta.txt"),100), 'St = 5', 'r', 3,'+')
My_hist_dens(density(lis_point("N512_st5\pcl-025000_z-4eta.txt"),100), 'St = 5', 'g', 3,'^')
My_hist_dens(density(lis_point("N512_st5\pcl-023000_z-4eta.txt"),100), 'St = 5', 'purple', 3,'|')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St5_dens.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')


My_hist_dens(density(lis_point("N512_st1\pcl-030000_z-4eta.txt"),100), 'St = 1', 'b', 4,'x')
My_hist_dens(density(lis_point("N512_st1\pcl-028000_z-4eta.txt"),100), 'St = 1', 'r', 4,'+')
My_hist_dens(density(lis_point("N512_st1\pcl-025000_z-4eta.txt"),100), 'St = 1', 'g', 4,'^')
My_hist_dens(density(lis_point("N512_st1\pcl-023000_z-4eta.txt"),100), 'St = 1', 'purple', 4,'|')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St1_dens.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')


My_hist_dens(density(lis_point("N512_st02\pcl-030000_z-4eta.txt"),100), 'St = 0.2', 'b', 5,'x')
My_hist_dens(density(lis_point("N512_st02\pcl-028000_z-4eta.txt"),100), 'St = 0.2', 'r', 5,'+')
My_hist_dens(density(lis_point("N512_st02\pcl-025000_z-4eta.txt"),100), 'St = 0.2', 'g', 5,'^')
My_hist_dens(density(lis_point("N512_st02\pcl-023000_z-4eta.txt"),100), 'St = 0.2', 'purple', 5,'|')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St02_dens.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')


My_hist_dens(density(lis_point("N512_st005\pcl-030000_z-4eta.txt"),100), 'St = 0.05', 'b', 6,'x')
My_hist_dens(density(lis_point("N512_st005\pcl-028000_z-4eta.txt"),100), 'St = 0.05', 'r', 6,'+')
My_hist_dens(density(lis_point("N512_st005\pcl-025000_z-4eta.txt"),100), 'St = 0.05', 'g', 6,'^')
My_hist_dens(density(lis_point("N512_st005\pcl-023000_z-4eta.txt"),100), 'St = 0.05', 'purple', 6,'|')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig('St005_dens.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')








lis_point("N512_st005\pcl-030000_z-4eta.txt")
plt.title('St = 0.05')
plt.savefig('St005_image.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

lis_point("N512_st02\pcl-030000_z-4eta.txt")
plt.title('St = 0.2')
plt.savefig('St02_image.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

lis_point("N512_st1\pcl-030000_z-4eta.txt")
plt.title('St = 1')
plt.savefig('St1_image.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

lis_point("N512_st5\pcl-030000_z-4eta.txt")
plt.title('St = 5')
plt.savefig('St5_image.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')


lis_point("N512_st1\pcl-030000_z-4eta.txt")
plt.title('Re = 204')
plt.savefig('St204_image.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')

lis_point("N1024_st1\pcl-0115000_z-4eta.txt")
plt.title('Re = 328')
plt.savefig('Re328_image.eps', bbox_inches='tight', quality=100, format='eps')
plt.close('all')





lis_point("N1024_st1\pcl-0115000_z-4eta.txt")
plt.title('Re = 328')
plt.savefig('Re328_image.png', bbox_inches='tight', quality=100)
plt.close('all')







density(lis_point("N512_st5\pcl-030000_z-4eta.txt"),100) + density(lis_point("N512_st5\pcl-028000_z-4eta.txt"),100) + density(lis_point("N512_st5\pcl-025000_z-4eta.txt"),100) + density(lis_point("N512_st5\pcl-023000_z-4eta.txt"),100) + density(lis_point("N512_st5\pcl-021000_z-4eta.txt"),100)



x*+,o^|
pink y r c g b purple




My_hist_dens(density_density(lis_point("N512_st5\pcl-030000_z-4eta.txt"),100) + density_density(lis_point("N512_st5\pcl-029000_z-4eta.txt"),100) + density_density(lis_point("N512_st5\pcl-028000_z-4eta.txt"),100) + density_density(lis_point("N512_st5\pcl-027000_z-4eta.txt"),100) + density_density(lis_point("N512_st5\pcl-026000_z-4eta.txt"),100) + density_density(lis_point("N512_st5\pcl-025000_z-4eta.txt"),100) + density_density(lis_point("N512_st5\pcl-024000_z-4eta.txt"),100) + density_density(lis_point("N512_st5\pcl-023000_z-4eta.txt"),100) + density_density(lis_point("N512_st5\pcl-022000_z-4eta.txt"),100) + density_density(lis_point("N512_st5\pcl-021000_z-4eta.txt"),100), 'St = 5', 'pink', 2,'x')
My_hist_dens(density_density(lis_point("N512_st2\pcl-030000_z-4eta.txt"),100) + density_density(lis_point("N512_st2\pcl-029000_z-4eta.txt"),100) + density_density(lis_point("N512_st2\pcl-028000_z-4eta.txt"),100) + density_density(lis_point("N512_st2\pcl-027000_z-4eta.txt"),100) + density_density(lis_point("N512_st2\pcl-026000_z-4eta.txt"),100) + density_density(lis_point("N512_st2\pcl-025000_z-4eta.txt"),100) + density_density(lis_point("N512_st2\pcl-024000_z-4eta.txt"),100) + density_density(lis_point("N512_st2\pcl-023000_z-4eta.txt"),100) + density_density(lis_point("N512_st2\pcl-022000_z-4eta.txt"),100) + density_density(lis_point("N512_st2\pcl-021000_z-4eta.txt"),100), 'St = 2', 'y', 2, '*')
My_hist_dens(density_density(lis_point("N512_st1\pcl-030000_z-4eta.txt"),100) + density_density(lis_point("N512_st1\pcl-029000_z-4eta.txt"),100) + density_density(lis_point("N512_st1\pcl-028000_z-4eta.txt"),100) + density_density(lis_point("N512_st1\pcl-027000_z-4eta.txt"),100) + density_density(lis_point("N512_st1\pcl-026000_z-4eta.txt"),100) + density_density(lis_point("N512_st1\pcl-025000_z-4eta.txt"),100) + density_density(lis_point("N512_st1\pcl-024000_z-4eta.txt"),100) + density_density(lis_point("N512_st1\pcl-023000_z-4eta.txt"),100) + density_density(lis_point("N512_st1\pcl-022000_z-4eta.txt"),100) + density_density(lis_point("N512_st1\pcl-021000_z-4eta.txt"),100), 'St = 1', 'r', 2, '+')
My_hist_dens(density_density(lis_point("N512_st05\pcl-030000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-029000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-028000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-027000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-026000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-025000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-024000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-023000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-022000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-021000_z-4eta.txt"),100), 'St = 0.5', 'c', 2, ',')
plt.ylabel('Frequency')
plt.xlabel('Density of void areas')
plt.savefig('St_dens_dens_big.eps', bbox_inches='tight', quality=100, format='eps')

My_hist_dens(density_density(lis_point("N512_st05\pcl-030000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-029000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-028000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-027000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-026000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-025000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-024000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-023000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-022000_z-4eta.txt"),100) + density_density(lis_point("N512_st05\pcl-021000_z-4eta.txt"),100), 'St = 0.5', 'c', 2, ',')
My_hist_dens(density_density(lis_point("N512_st02\pcl-030000_z-4eta.txt"),100) + density_density(lis_point("N512_st02\pcl-028000_z-4eta.txt"),100) + density_density(lis_point("N512_st02\pcl-025000_z-4eta.txt"),100) + density_density(lis_point("N512_st02\pcl-023000_z-4eta.txt"),100) + density_density(lis_point("N512_st02\pcl-021000_z-4eta.txt"),100), 'St = 0.2', 'g', 2,'o')
My_hist_dens(density_density(lis_point("N512_st01\pcl-030000_z-4eta.txt"),100) + density_density(lis_point("N512_st01\pcl-028000_z-4eta.txt"),100) + density_density(lis_point("N512_st01\pcl-025000_z-4eta.txt"),100) + density_density(lis_point("N512_st01\pcl-023000_z-4eta.txt"),100) + density_density(lis_point("N512_st01\pcl-021000_z-4eta.txt"),100), 'St = 0.1', 'b', 2,'^')
My_hist_dens(density_density(lis_point("N512_st005\pcl-030000_z-4eta.txt"),100) + density_density(lis_point("N512_st005\pcl-028000_z-4eta.txt"),100) + density_density(lis_point("N512_st005\pcl-025000_z-4eta.txt"),100) + density_density(lis_point("N512_st005\pcl-023000_z-4eta.txt"),100) + density_density(lis_point("N512_st005\pcl-021000_z-4eta.txt"),100), 'St = 0.05', 'purple', 2, '|')
plt.ylabel('Frequency')
plt.xlabel('Density of void areas')
plt.savefig('St_dens_dens_small.eps', bbox_inches='tight', quality=100, format='eps')

plt.ylabel('Frequency')
plt.xlabel('Density of void areas')
plt.savefig('St_dens_dens.eps', bbox_inches='tight', quality=100, format='eps')





    plt.xscale('linear')
    plt.yscale('linear')


    plt.xscale('log')
    plt.yscale('log')






"""

print(moyenne(np.log10(AR_5_1)))
print(moyenne(np.log10(AR_2_1)))
print(moyenne(np.log10(AR_1_1)))
print(moyenne(np.log10(AR_05_1)))
print(moyenne(np.log10(AR_02_1)))
print(moyenne(np.log10(AR_01_1)))
print(moyenne(np.log10(AR_005_1)))


print(moyenne((AR_5_1)))
print(moyenne((AR_2_1)))
print(moyenne((AR_1_1)))
print(moyenne((AR_05_1)))
print(moyenne((AR_02_1)))
print(moyenne((AR_01_1)))
print(moyenne((AR_005_1)))


\begin{center}
	\begin{tabular}{ | c | c | }
		\hline
	   St & $Log_{10}(Area)$  \\ \hline
	   0.05 & -1.059  \\ \hline
	   0.1 & -1.069  \\ \hline
	   0.2 & -1.092  \\ \hline
	   0.5 & -1.153  \\ \hline
	   1 & -1.197  \\ \hline
	   2 & -1.204  \\ \hline
	   5 & -1.129  \\
		\hline
	\end{tabular}
\end{center}
"""






def Fonction2(fichier, couleur):

    
    point=lis_point(fichier)
    
    
    vor_2d = Voronoi(point)
    
    t1=time.time()
    data=[]
    data=voronoi_cell_area(vor_2d)
    
    t2=time.time()
    print(t2-t1)
    
    vor_2d = Voronoi(point+create_border(point))
    
    daaa=voronoi_cell_area_test(vor_2d)
    t3=time.time()
    print(t3-t2)
    """   """
    
    histogramme(data[0], 200,1,1,0.1,1000, fichier, couleur, 2)
    #plt.legend()
    """
    plt.figure(3)
    plt.hist(log10(data[0]), bins=200, normed=1, color = couleur, log=True)
    #plt.hist(log10(data), bins=200, normed=1, color = 'b')
    plt.xlabel('log10(x)')
    plt.savefig('Hist_log.png')
    
    
    #plt.figure(2)
    plt.figure(4)
    plt.yscale('log')
    n = 200
    m, sig = moyenne(log10(data[0])), ecartype(log10(data[0]))
    X = npr.normal(m, sig, n)
    x = np.arange(X.min(), X.max() + 0.01, 0.01)
    plt.plot(x, st.norm.pdf(x, m, sig), couleur, linewidth=1, label=fichier[5:10])
    plt.title('Log(Aire) des cellules de Voronoi')
    plt.legend()
    plt.savefig('Loi_log.png')
    """   
    
    dens=density(point, 100)
    """
    histogramme(dens, int(max(dens)),0,0,0,40, "density", couleur, 4)
    loi_de_Poisson(moyenne(dens))
    """
    """
    for i in range(len(dens)):
        dens[i]=dens[i]/(len(data[0]))
    """
    """
    histogramme(dens, 200,1,1,0,40, fichier, couleur, 5)
    
    histogramme(dens, int(max(dens)),0,1,0,1000, "Density", couleur, 7)
    loi_de_Poisson(moyenne(dens), int(max(dens)))
    plt.legend()
    
    
    plt.figure(6)
    rang=int(6*ecartype(dens))
    plt.hist(log10(dens), bins=rang, normed=1, color = couleur, log=True)
    plt.title('Density_log')
    plt.savefig('Density_log.png')
    """   
    
    print("------------------------------------------")
    
    return dens, data

"""
#Fonction("pcl-021000_z-4eta.txt", 'b')
   
#------------------------------------------------------------------------------


x1=Fonction("N512_st5\pcl-030000_z-4eta.txt",'pink')


plt.hist(x1[1][0], bins=200, normed=1, color = 'c')#, range=(0,60))#, log=True)


histogramme(x1[1][0], 200,1,1,0,60, "ggg", 'c', 7)


#print(voronoi_cell_area(vor_2d))
#voronoi_plot_2d(vor_2d)                      #affiche voronoi

"""




"""



np.transpose(create_border(point))



m1=np.transpose(create_border(point))[0]
m2=np.transpose(point)[0]
l1=m1.tolist()
l2=m2.tolist()

Q1=l2+l1

m3=np.transpose(create_border(point))[1]
m4=np.transpose(point)[1]
l3=m3.tolist()
l4=m4.tolist()

Q2=l4+l3

Po=[]
for i in range(len(Q1)):
    Po.append([Q1[i],Q2[i]])


vor_2d_0 = Voronoi(Po)
#voronoi_plot_2d(vor_2d_0)


t5=time.time()
AR=[]
AR=voronoi_cell_area_test(vor_2d_0,len(point))
t6=time.time()
print(t6-t5)


plot_loghist(AR, 200, 'r')
histogramme(AR, 200,1,1,0,60, "ggg", 'c', 6)




T=[]
G=gamma.ppf(0.9999, a)
m=max(AR)
for i in range(len(AR)):
    T.append(AR[i]/m*28)

plot_loghist(T, 200, 'c')
plot_loghist(T, 200, 'fffffffff', 'c',2)


plt.figure(1,figsize=(6,6), dpi=80)
plt.scatter(Q1,Q2,s=0.05,c='black')


vor_2d = Voronoi(point)

data=[]
data=voronoi_cell_area(vor_2d)


histogramme(data[0], 200,1,0,0,60, "ggg", 'c', 7)
plot_loghist(data[0], 200, 'test', 'r', 3)


vor_2d_1=Voronoi(Po)
t5=time.time()
DA=[]
DA=voronoi_cell_area(vor_2d_1)
t6=time.time()
print(t6-t5)

histogramme(DA[0], 200,1,0,0,60, "ggg", 'c', 7)
plot_loghist(DA[0], 200, 'test', 'r', 3)


maxi=0
for i in range(len(AR[0])):
    if AR[0][i]>maxi:
        maxi=AR[0][i]
        print(i,maxi,AR[1][i],AR[0][i])

for i in range(len(vor_2d_0.regions)-1):
    if (int(vor_2d_0.point_region[i]-1)==37529):
        print(vor_2d_0.regions[i+1], i+1)
        for j in range(len(vor_2d_0.regions[i+1])):
            print(vor_2d_0.vertices[vor_2d_0.regions[i+1][j]])


"""







#fichier="N512_st1\pcl-030000_z-4eta.txt"

def Fonction266(fichier, couleur, legende, nom):

    point=lis_point(fichier)

#    np.transpose(create_border(point))
    
    m1=np.transpose(create_border(point))[0]
    m2=np.transpose(point)[0]
    l1=m1.tolist()
    l2=m2.tolist()
    
    Q1=l2+l1
    
    m3=np.transpose(create_border(point))[1]
    m4=np.transpose(point)[1]
    l3=m3.tolist()
    l4=m4.tolist()
    
    Q2=l4+l3
    
    Po=[]
    for i in range(len(Q1)):
        Po.append([Q1[i],Q2[i]])
    
    vor_2d_0 = Voronoi(Po)
    #voronoi_plot_2d(vor_2d_0)

    AR=[]
    AR=voronoi_cell_area_test(vor_2d_0,len(point))
    
    My_hist_vor(AR, legende, couleur,2)
    




Fonction2("N512_st1\pcl-030000_z-4eta.txt", 'r', '1', "2D_N512-st1-030000-vor")












