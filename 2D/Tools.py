# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:27:01 2019

@author: Moli Oujia
"""


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from numpy import log10



def density(point, N):
    
    
    nombre=np.zeros((N,N))
    for i in range(len(point)):
        nombre[int(point[i][0]*N/(2*np.pi))][int(point[i][1]*N/(2*np.pi))]+=1
        
    tableau=[]
    for i in range(N):
        for j in range(N):
            if nombre[j][i]!=0:
                tableau.append(nombre[j][i])          #tableau.append(log10(nombre[j][i]))       #
            else:
                tableau.append(0)

    return tableau



#"N512_st005\pcl-030000_z-4eta.txt"
#lis_point("N512_st005\pcl-024000_z-4eta.txt")

def lis_point(fichier):
    
    x, y, z = [], [], []
    with open(fichier, "r") as f_read:
    #with open("N512_st05\pcl-030000_z-4eta.txt", "r") as f_read:
        for line in f_read:
            line = line.strip()  #suppression du retour charriot en fin de ligne
            if line:
                xi, yi, zi = [float(elt) for elt in line.split("  ")]  #conversion en nb
                x.append(xi), y.append(yi), z.append(zi)
    
    
    plt.figure(1,figsize=(6,6), dpi=80)
    
    plt.scatter(x,y,s=0.05,c='black')
    
    plt.title('Nuage de points')
#    plt.xlabel('x')
#    plt.ylabel('y')
    plt.savefig('Particles.png')
    
    """ """
    point=np.zeros((len(x),2))
    
    for i in range(len(x)):
        point[i][0]=x[i]
        point[i][1]=y[i]

    return point



def ecrit_point(fichier, point):
    fichier = open(fichier, "w")
    for i in range(len(point)):
        fichier.write(str(point[i][0])+ "   "+ str(point[i][1])+ "   "+str(point[i][1])+'\n')
    fichier.close()


def ecrit_data(fichier, data):
    fichier = open(fichier, "w")
    for i in range(len(data)):
        fichier.write(str(data[i])+'\n')
    fichier.close()
    

def voronoi_cell_area(vor_2d):
    Area=np.zeros((len(vor_2d.regions),1))
    
    for i in range(len(vor_2d.regions)):
        Coord_ind_points=[]
        test_espace=0
        if -1 not in vor_2d.regions[i] and vor_2d.regions[i]!=[]:
            for j in range(len(vor_2d.regions[i])):
                if vor_2d.vertices[vor_2d.regions[i][j]][0]<0 or 2*np.pi<vor_2d.vertices[vor_2d.regions[i][j]][0] or vor_2d.vertices[vor_2d.regions[i][j]][1]<0 or 2*np.pi<vor_2d.vertices[vor_2d.regions[i][j]][1]:
                    test_espace=1
            if test_espace==0:
                for j in range(len(vor_2d.regions[i])):
                    Coord_ind_points.append(vor_2d.vertices[vor_2d.regions[i][j]])
                Area[i]=ConvexHull(Coord_ind_points).area
    
    Areanozero=[]
    for i in range(len(vor_2d.regions)):
        if Area[i]!=0:
            Areanozero.append(Area[i])
        
    return Areanozero, Area



def moyenne(tableau):
    somme=0
    for i in range(len(tableau)):
        somme=somme+tableau[i]
    return somme / len(tableau)

#    print(moyenne(-log10(data)))
    
def variance(tableau):
    m=moyenne(tableau)
    return moyenne([(x-m)**2 for x in tableau])
    
#    print(variance(-log10(data)))
    
    
def ecartype(tableau):
    return variance(tableau)**0.5



def loi_de_Poisson(l,r):
    
    plt.yscale('log')
    #plt.xscale('log')
    
    p=[]
    px=[]
    for i in range(r+1):
        p.append(l**i*math.exp(-l)/(math.factorial(i)))
        px.append(i)
    
    plt.scatter(px,p,s=2, c='r',label="Poisson")


def loi_normal(mu, s, mini, maxi, ln):
    x = np.arange(mini, maxi + 0.01, 0.001)
    f=[]
    
    for i in range(len(x)):
        f.append( ( (1/(s*math.sqrt(2*np.pi)))*math.exp(-1/2*((x[i]-mu)/s)**2) ) )
        if ln==1:
            x[i]=10**x[i]#math.exp(x[i])#
            #plt.xscale('log')
    plt.plot(x,f)


def met_log10(liste):
    for i in range(len(liste)):
        if liste[i]!=0:
            liste[i]=log10(liste[i])
    return liste


def histogramme(tab, subdivise, logx, logy, bmin, bmax, legende, couleur, fig):

    #compte=moyenne(tab)*len(tab)
    
    liste=np.copy(tab)
    if logx==1:
        for i in range(len(liste)):
            if liste[i]!=0:
                liste[i]=log10(liste[i])
        
    
    maxi=max(abs(max(liste)),abs(min(liste)))
    dens=np.zeros((subdivise+1))
    moy=np.zeros((subdivise+1))
    hist=[]
    x=[]
    
    #moyenne(liste)

    for i in range(len(liste)):
        if bmin<=abs(liste[i])<=bmax:
            #print(tab[i],abs(int((liste[i])*subdivise/maxi)))
            dens[ abs(int(liste[i]*subdivise/maxi)) ]+=1
            moy[ abs(int(liste[i]*subdivise/maxi)) ]+=tab[i] #liste[i] #
    

    n=maxi/subdivise#subdivise
    compte=0
    for i in range(subdivise):
        compte=compte+dens[i]*n
    #print(compte)
    #print(n)
    
    for i in range(subdivise):
        if dens[i]!=0:
            x.append(abs(moy[i]/dens[i]))
            hist.append(abs(dens[i]/compte))
    
    #test=0
    #for i in range(subdivise):
    
    plt.figure(fig)
    if logx==1:
        plt.xscale('log')
    if logy==1:
        plt.yscale('log')
    
    if logy==0 and logy==1:
        plt.axis([0-0.0001, max(x)+max(x)/50, 0-0.0001, max(hist)+max(hist)/50])
    
    plt.scatter(x,hist,s=2, label=legende, c=couleur)
    plt.legend()
    
    
    #loi_normal( abs(moyenne(log10(tab))), abs(ecartype(log10(tab))), -3, 3, 1 )
    
    #return x,hist

"""

loi_normal(moyenne(x7[1]), ecartype(x7[1]), min(x7[1]), max(x7[1]), 1)
histogramme(x7[1], 200,1,1,0.001,40, "N512_st005\pcl-030000_z-4eta.txt", 'purple', 7)



histogramme(x6005, 200,0,0,0.001,40, fichier, 'b', 7)



histogramme(data, 200,1,1,0.1,1000, fichier, 'r', 2)
histogramme(data1, 200,1,1,0.1,1000, fichier, 'pink', 2)




lololl=histogramme(data, 200,1,1,-40,40)

plt.figure(3)
loi_normal( abs(moyenne(log10(data))), abs(ecartype(log10(data))), log10(lololl[0][0]), log10(lololl[0][178]), 1 )
plt.figure(4)
loi_normal( abs(moyenne(log10(data))), abs(ecartype(log10(data))), 0, 3, 0 )
"""

"""
plt.figure(3)
loi_normal( abs(moyenne(log10(data))), ecartype(log10(data)), 33, 3, 0) 
"""





"""
test=[]
mo=10**(moyenne(log10(data[0])) - ecartype(log10(data[0])))
for i in range(len(data[1])-1):
    if data[1][i]<mo:
        test.append(i)

point2=[]

for i in test:
    point2.append(point[i])

x=[]
y=[]

for i in range(len(point2)):
    x.append(point2[i][0])
    y.append(point2[i][1])

plt.figure(66,figsize=(6,6), dpi=80)
plt.scatter(x,y,s=0.05,c='black')

"""
