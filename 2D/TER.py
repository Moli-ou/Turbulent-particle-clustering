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



#fichier="pcl-021000_z-4eta.txt"
#fichier="N512_st1\pcl-030000_z-4eta.txt"
#fichier="N512_st02\pcl-030000_z-4eta.txt"
#fichier="points_2d.txt"
#couleur='r'
#"N512_st005\pcl-030000_z-4eta.txt"


def Fonction(fichier, couleur):

    
    point=lis_point(fichier)
    
    vor_2d = Voronoi(point)
    
    
    data=[]
    data=voronoi_cell_area(vor_2d)
    
    histogramme(data[0], 200,1,1,0.1,1000, fichier, couleur, 2)
    #plt.legend()
    
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
    """   """
    
    dens=density(point, 100)
    """
    histogramme(dens, int(max(dens)),0,0,0,40, "density", couleur, 4)
    loi_de_Poisson(moyenne(dens))
    """
    """
    for i in range(len(dens)):
        dens[i]=dens[i]/(len(data[0]))
    """
    
    histogramme(dens, 200,1,1,0,40, fichier, couleur, 5)
    
    histogramme(dens, int(max(dens)),0,1,0,1000, "Density", couleur, 7)
    loi_de_Poisson(moyenne(dens), int(max(dens)))
    plt.legend()
    
    """
    plt.figure(6)
    rang=int(6*ecartype(dens))
    plt.hist(log10(dens), bins=rang, normed=1, color = couleur, log=True)
    plt.title('Density_log')
    plt.savefig('Density_log.png')
    """   
    
    print("------------------------------------------")
    
    return dens, data
    
#    return moyenne(log10(data)), ecartype(log10(data))


#Fonction("pcl-021000_z-4eta.txt", 'b')
   
    
#------------------------------------------------------------------------------


x1=Fonction("N512_st5\pcl-030000_z-4eta.txt",'pink')
x2=Fonction("N512_st2\pcl-030000_z-4eta.txt",'y')
x3=Fonction("N512_st1\pcl-030000_z-4eta.txt",'r')
x4=Fonction("N512_st05\pcl-030000_z-4eta.txt",'c')
x5=Fonction("N512_st02\pcl-030000_z-4eta.txt",'g')
x6=Fonction("N512_st01\pcl-030000_z-4eta.txt",'b')
x7=Fonction("N512_st005\pcl-030000_z-4eta.txt",'purple')


x1=Fonction("N512_st5\pcl-021000_z-4eta.txt",'b')
x2=Fonction("N512_st5\pcl-024000_z-4eta.txt",'g')
x1=Fonction("N512_st5\pcl-027000_z-4eta.txt",'r')
x2=Fonction("N512_st5\pcl-030000_z-4eta.txt",'c')

x1=Fonction("N512_st1\pcl-021000_z-4eta.txt",'b')
x2=Fonction("N512_st1\pcl-024000_z-4eta.txt",'g')
x1=Fonction("N512_st1\pcl-027000_z-4eta.txt",'r')
x2=Fonction("N512_st1\pcl-030000_z-4eta.txt",'c')

x1=Fonction("N512_st02\pcl-021000_z-4eta.txt",'b')
x2=Fonction("N512_st02\pcl-024000_z-4eta.txt",'g')
x1=Fonction("N512_st02\pcl-027000_z-4eta.txt",'r')
x2=Fonction("N512_st02\pcl-030000_z-4eta.txt",'c')

x1=Fonction("N512_st005\pcl-021000_z-4eta.txt",'b')
x2=Fonction("N512_st005\pcl-024000_z-4eta.txt",'g')
x1=Fonction("N512_st005\pcl-027000_z-4eta.txt",'r')
x2=Fonction("N512_st005\pcl-030000_z-4eta.txt",'c')



x1=Fonction("N1024_st5\pcl-0150000_z-4eta.txt",'r')
x1=Fonction("N512_st5\pcl-030000_z-4eta.txt",'b')
x2=Fonction("N1024_st2\pcl-0150000_z-4eta.txt",'r')
x2=Fonction("N512_st2\pcl-030000_z-4eta.txt",'b')
x3=Fonction("N1024_st1\pcl-0150000_z-4eta.txt",'r')
x3=Fonction("N512_st1\pcl-030000_z-4eta.txt",'b')
x4=Fonction("N1024_st05\pcl-0150000_z-4eta.txt",'r')
x4=Fonction("N512_st05\pcl-030000_z-4eta.txt",'b')



histogramme(x1[1], int(max(x1[0])),1,1,0.0,60, "ggg", 'pink', 7)
histogramme(x2[1], int(max(x1[0])),1,1,0.0,60, "ggg", 'y', 7)
histogramme(x3[1], int(max(x1[0])),1,1,0.0,60, "ggg", 'r', 7)
histogramme(x4[1], int(max(x1[0])),1,1,0.0,60, "ggg", 'c', 7)




x1=Fonction("N1024_st5\pcl-0150000_z-4eta.txt",'pink')
x2=Fonction("N1024_st2\pcl-0150000_z-4eta.txt",'y')
x3=Fonction("N1024_st1\pcl-0150000_z-4eta.txt",'r')
x4=Fonction("N1024_st05\pcl-0150000_z-4eta.txt",'c')


plt.hist(x1[0], bins=60, normed=1, color = 'pink', range=(0,60))#, log=True)
plt.hist(x2[0], bins=60, normed=1, color = 'y', range=(0,60))#, log=True)
plt.hist(x3[0], bins=60, normed=1, color = 'r', range=(0,60))#, log=True)
plt.hist(x4[0], bins=60, normed=1, color = 'c', range=(0,60))#, log=True)


histogramme(x1[0], 60,0,0,0.0,60, "ggg", 'pink', 7)
histogramme(x2[0], 60,0,0,0.0,60, "ggg", 'y', 7)
histogramme(x3[0], 60,0,0,0.0,60, "ggg", 'r', 7)
histogramme(x4[0], 60,0,0,0.0,60, "ggg", 'c', 7)


x1=Fonction("N1024_st5\pcl-0150000_z-4eta.txt",'b')
x2=Fonction("N1024_st5\pcl-0135000_z-4eta.txt",'g')
x1=Fonction("N1024_st5\pcl-0120000_z-4eta.txt",'r')
x2=Fonction("N1024_st5\pcl-0105000_z-4eta.txt",'c')





"""
stat0=[]
stat1=[]
stat0.append(moyenne(met_log10(x7[0])))
stat1.append(ecartype(met_log10(x7[0])))
stat0.append(moyenne(met_log10(x6[0])))
stat1.append(ecartype(met_log10(x6[0])))
stat0.append(moyenne(met_log10(x5[0])))
stat1.append(ecartype(met_log10(x5[0])))
stat0.append(moyenne(met_log10(x4[0])))
stat1.append(ecartype(met_log10(x4[0])))
stat0.append(moyenne(met_log10(x3[0])))
stat1.append(ecartype(met_log10(x3[0])))
stat0.append(moyenne(met_log10(x2[0])))
stat1.append(ecartype(met_log10(x2[0])))
stat0.append(moyenne(met_log10(x1[0])))
stat1.append(ecartype(met_log10(x1[0])))

plt.figure(10)
#St=[0.05,0.1,0.2,0.5,1,2,5]
St=[0.05,0.1,0.2,0.5,1,2,5]
pl=plt.scatter(St,stat1,s=5)
#pl=plt.plot(St,stat1,"b",marker=".")
#plt.legend(pl, ['ecartype'],bbox_to_anchor = (0.2, 1))
ax2 = plt.gca().twinx()
pl2=ax2.scatter(St,stat0,s=5, c='red')
#pl2 = ax2.plot(St,stat0,"r",marker=".")
#plt.legend(pl2, ['moyenne'])
plt.savefig('ecartype_moyenne.png')



stat3=[]
stat4=[]
stat3.append(moyenne(met_log10(x7[1])))
stat4.append(ecartype(met_log10(x7[1])))
stat3.append(moyenne(met_log10(x6[1])))
stat4.append(ecartype(met_log10(x6[1])))
stat3.append(moyenne(met_log10(x5[1])))
stat4.append(ecartype(met_log10(x5[1])))
stat3.append(moyenne(met_log10(x4[1])))
stat4.append(ecartype(met_log10(x4[1])))
stat3.append(moyenne(met_log10(x3[1])))
stat4.append(ecartype(met_log10(x3[1])))
stat3.append(moyenne(met_log10(x2[1])))
stat4.append(ecartype(met_log10(x2[1])))
stat3.append(moyenne(met_log10(x1[1])))
stat4.append(ecartype(met_log10(x1[1])))


plt.figure(11)
#St=[0.05,0.1,0.2,0.5,1,2,5]
St=[0.05,0.1,0.2,0.5,1,2,5]
pl=plt.scatter(St,stat4,s=5)
#pl=plt.plot(St,stat1,"b",marker=".")
#plt.legend(pl, ['ecartype'],bbox_to_anchor = (0.2, 1))
ax2 = plt.gca().twinx()
pl2=ax2.scatter(St,stat3,s=5, c='red')
#pl2 = ax2.plot(St,stat0,"r",marker=".")
#plt.legend(pl2, ['moyenne'])
plt.savefig('ecartype_moyenne.png')

"""







"""

histogramme(x1[1], 200,1,1,0.001,40, fichier, 'pink', 7)
histogramme(x2[1], 200,1,1,0.001,40, fichier, 'y', 7)
histogramme(x3[1], 200,1,1,0.001,40, fichier, 'r', 7)
histogramme(x4[1], 200,1,1,0.001,40, fichier, 'c', 7)
histogramme(x5[1], 200,1,1,0.001,40, fichier, 'g', 7)
histogramme(x6[1], 200,1,1,0.001,40, fichier, 'b', 7)
histogramme(x7[1], 200,1,1,0.001,40, "N512_st005\pcl-030000_z-4eta.txt", 'purple', 7)





plt.figure(6)
plt.hist(log10(x1[1]), bins=200, normed=1, color = 'pink', log=True)
plt.hist(log10(x2[1]), bins=200, normed=1, color = 'y', log=True)
plt.hist(log10(x3[1]), bins=200, normed=1, color = 'r', log=True)
plt.hist(log10(x4[1]), bins=200, normed=1, color = 'c', log=True)
plt.hist(log10(x5[1]), bins=200, normed=1, color = 'g', log=True)
plt.hist(log10(x6[1]), bins=200, normed=1, color = 'b', log=True)
plt.hist(log10(x7[1]), bins=200, normed=1, color = 'purple', log=True)



histogramme(x1[0], int(max(x1[0])),0,0,0.0,40, "ggg", 'pink', 7)
histogramme(x2[0], int(max(x2[0])),0,0,0.0,40, "ggg", 'y', 7)
histogramme(x3[0], int(max(x3[0])),0,0,0.0,40, "ggg", 'r', 7)
histogramme(x4[0], int(max(x4[0])),0,0,0.0,40, "ggg", 'c', 7)
histogramme(x5[0], int(max(x5[0])),0,0,0.0,40, "ggg", 'g', 7)
histogramme(x6[0], int(max(x6[0])),0,0,0.0,40, "ggg", 'b', 7)
histogramme(x7[0], int(max(x7[0])),0,0,0.0,40, "ggg", 'purple', 7)

histogramme(x1[0], int(max(x1[0])),1,1,0.0,40, fichier, 'pink', 7)
histogramme(x2[0], int(max(x2[0])),1,1,0.0,40, fichier, 'y', 7)
histogramme(x3[0], int(max(x3[0])),1,1,0.0,40, fichier, 'r', 7)
histogramme(x4[0], int(max(x4[0])),1,1,0.0,40, fichier, 'c', 7)
histogramme(x5[0], int(max(x5[0])),1,1,0.0,40, fichier, 'g', 7)
histogramme(x6[0], int(max(x6[0])),1,1,0.0,40, fichier, 'b', 7)
histogramme(x7[0], int(max(x7[0])),1,1,0.0,40, fichier, 'purple', 7)

histogramme(x1[0], 200,1,1,0.0,40, fichier, 'pink', 7)
histogramme(x2[0], 200,1,1,0.0,40, fichier, 'y', 7)
histogramme(x3[0], 200,1,1,0.0,40, fichier, 'r', 7)
histogramme(x4[0], 200,1,1,0.0,40, fichier, 'c', 7)
histogramme(x5[0], 200,1,1,0.0,40, fichier, 'g', 7)
histogramme(x6[0], 200,1,1,0.0,40, fichier, 'b', 7)
histogramme(x7[0], 200,1,1,0.0,40, fichier, 'purple', 7)
"""


"""   


x1=Fonction("N1024_st5\pcl-0150000_z-4eta.txt",'pink')
x2=Fonction("N1024_st2\pcl-0150000_z-4eta.txt",'y')
x3=Fonction("N1024_st1\pcl-0150000_z-4eta.txt",'r')
x4=Fonction("N1024_st05\pcl-0150000_z-4eta.txt",'c')




x1005=Fonction("N512_st005\pcl-021000_z-4eta.txt",'pink')
x2005=Fonction("N512_st005\pcl-023000_z-4eta.txt",'y')
x3005=Fonction("N512_st005\pcl-025000_z-4eta.txt",'r')
x4005=Fonction("N512_st005\pcl-027000_z-4eta.txt",'c')
x5005=Fonction("N512_st005\pcl-028000_z-4eta.txt",'g')
x6005=Fonction("N512_st005\pcl-029000_z-4eta.txt",'b')
x7005=Fonction("N512_st005\pcl-030000_z-4eta.txt",'purple')

"""


"""

histogramme(x1005[0], 200,1,1,0.001,40, fichier, 'pink', 7)
histogramme(x2005[0], 200,1,1,0.001,40, fichier, 'y', 7)
histogramme(x3005[0], 200,1,1,0.001,40, fichier, 'r', 7)
histogramme(x4005[0], 200,1,1,0.001,40, fichier, 'c', 7)
histogramme(x5005[0], 200,1,1,0.001,40, fichier, 'g', 7)
histogramme(x6005[0], 200,1,1,0.001,40, fichier, 'b', 7)
histogramme(x7005[0], 200,1,1,0.001,40, fichier, 'purple', 7)



histogramme(x1005[0], 25,0,0,0.0,40, fichier, 'pink', 7)
histogramme(x2005[0], 25,0,0,0.0,40, fichier, 'y', 7)
histogramme(x3005[0], 25,0,0,0.0,40, fichier, 'r', 7)
histogramme(x4005[0], 25,0,0,0.0,40, fichier, 'c', 7)
histogramme(x5005[0], 25,0,0,0.0,40, fichier, 'g', 7)
#histogramme(x6005[0], 200,0,0,0.0,40, fichier, 'b', 7)
histogramme(x7005[0], 25,0,0,0.0,40, fichier, 'purple', 7)





plt.hist(x2005[0], bins=25, normed=1, color = couleur, range=(0,25))#, log=True)


"""









"""
histogramme(x1005[1], 200,0,0,0.001,40, fichier, 'pink', 7)
histogramme(x2005[1], 200,0,0,0.001,40, fichier, 'y', 7)
histogramme(x3005[1], 200,0,0,0.001,40, fichier, 'r', 7)
histogramme(x4005[1], 200,0,0,0.001,40, fichier, 'c', 7)
histogramme(x5005[1], 200,0,0,0.001,40, fichier, 'g', 7)
histogramme(x6005[1], 200,0,0,0.001,40, fichier, 'b', 7)
histogramme(x7005[1], 200,0,0,0.001,40, fichier, 'purple', 7)


histogramme(x1005, 200,1,1,0.001,40, fichier, 'pink', 7)
histogramme(x2005, 200,1,1,0.001,40, fichier, 'y', 7)
histogramme(x3005, 200,1,1,0.001,40, fichier, 'r', 7)
histogramme(x4005, 200,1,1,0.001,40, fichier, 'c', 7)
histogramme(x5005, 200,1,1,0.001,40, fichier, 'g', 7)
histogramme(x6005, 200,1,1,0.001,40, fichier, 'b', 7)
histogramme(x7005, 200,1,1,0.001,40, fichier, 'purple', 7)


plt.figure(6)
plt.hist(x6005, bins=25, normed=1, color = couleur, range=(0,25))#, log=True)




"""




"""

Stat=[]

Stat.append(Fonction("N512_st005\pcl-024000_z-4eta.txt",'purple'))
Stat.append(Fonction("N512_st01\pcl-024000_z-4eta.txt",'b'))
Stat.append(Fonction("N512_st02\pcl-024000_z-4eta.txt",'g'))
Stat.append(Fonction("N512_st05\pcl-024000_z-4eta.txt",'c'))
Stat.append(Fonction("N512_st1\pcl-024000_z-4eta.txt",'r'))
Stat.append(Fonction("N512_st2\pcl-024000_z-4eta.txt",'y'))
Stat.append(Fonction("N512_st5\pcl-024000_z-4eta.txt",'pink'))



stat0=[]
stat1=[]
for i in range(len(Stat)):
    stat0.append(Stat[i][0])
    stat1.append(Stat[i][1])

plt.figure(5)
#St=[0.05,0.1,0.2,0.5,1,2,5]
St=[0.05,0.1,0.2,0.5,1,2,5]
pl=plt.scatter(St,stat1,s=5)
#pl=plt.plot(St,stat1,"b",marker=".")
#plt.legend(pl, ['ecartype'],bbox_to_anchor = (0.2, 1))
ax2 = plt.gca().twinx()
pl2=ax2.scatter(St,stat0,s=5, c='red')
#pl2 = ax2.plot(St,stat0,"r",marker=".")
#plt.legend(pl2, ['moyenne'])
plt.savefig('ecartype_moyenne.png')



"""











"""


x1=Fonction("N512_st5\pcl-030000_z-4eta.txt",'pink')
x2=Fonction("N512_st2\pcl-030000_z-4eta.txt",'y')
x3=Fonction("N512_st1\pcl-030000_z-4eta.txt",'r')
x4=Fonction("N512_st05\pcl-030000_z-4eta.txt",'c')
x5=Fonction("N512_st02\pcl-030000_z-4eta.txt",'g')
x6=Fonction("N512_st01\pcl-030000_z-4eta.txt",'b')
x7=Fonction("N512_st005\pcl-030000_z-4eta.txt",'purple')


"""












#print(voronoi_cell_area(vor_2d))
#voronoi_plot_2d(vor_2d)                      #affiche voronoi


