# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:27:01 2019

@author: Moli Oujia
"""


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from struct import *
import struct
from numpy import log10



def density_3d(point, N):
    
    
    nombre=np.zeros((N,N,N))
    for i in range(len(point)):
        nombre[int(point[i][0]*N/(2*np.pi))][int(point[i][1]*N/(2*np.pi))][int(point[i][2]*N/(2*np.pi))]+=1
    
    tableau=[]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if nombre[j][i][k]!=0:
                    tableau.append(nombre[j][i][k])          #tableau.append(log10(nombre[j][i]))       #
                    #print(nombre[j][i][k])
                else:
                    tableau.append(0)

    return tableau


def lis_point_3d(fichier):
    
    x, y, z = [], [], []
    with open(fichier, "r") as f_read:
    #with open("N512_st05\pcl-030000_z-4eta.txt", "r") as f_read:
        for line in f_read:
            line = line.strip()  #suppression du retour charriot en fin de ligne
            if line:
                xi, yi, zi = [float(elt) for elt in line.split("  ")]  #conversion en nb
                x.append(xi), y.append(yi), z.append(zi)
    
    """
    plt.figure(1,figsize=(6,6), dpi=80)
    
    plt.scatter(x,y,s=0.05,c='black')
    
    plt.title('Nuage de points')
#    plt.xlabel('x')
#    plt.ylabel('y')
    plt.savefig('Particles.png')
    """
    
    point=np.zeros((len(x),3))
    
    for i in range(len(x)):
        point[i][0]=x[i]
        point[i][1]=y[i]
        point[i][2]=z[i]

    return point


#"N512_st005\pcl-030000_z-4eta.txt"
#lis_point("N512_st005\pcl-024000_z-4eta.txt")

def lis_point_3d_binaire(fichier):
    
    Donner=[]
        
    f = open(fichier, "rb")
    
    test='12345678'
    
    try:
        while True and len(test)!=0:
            test=f.read(8)
            if len(test)==8:
                Donner.append( struct.unpack('d',test)[0] )
             
        #print(Donner) # Decode ("unpack")
     
    except IOError:
            # Your error handling here
            # Nothing for this example
            pass
    finally:
        f.close()
    
    points_3d=np.zeros((3,int(len(Donner)/3)))
    
    points_3d[0]=Donner[0:int(len(Donner)/3)]
    points_3d[1]=Donner[int(len(Donner)/3):int(2*len(Donner)/3)]
    points_3d[2]=Donner[int(2*len(Donner)/3):int(len(Donner))]
    
    points_3d=np.transpose(points_3d)

    return points_3d

def ecrit_point_3d(fichier, point):
    fichier = open(fichier, "w")
    for i in range(len(point)):
        fichier.write(str(point[i][0])+ "   "+ str(point[i][1])+ "   "+str(point[i][2])+'\n')
    fichier.close()

def ecrit_data(fichier, data):
    fichier = open(fichier, "w")
    for i in range(len(data)):
        fichier.write(str(data[i])+'\n')
    fichier.close()

def lis_data(fichier):
    
    x = []
    with open(fichier, "r") as f_read:
    #with open("N512_st05\pcl-030000_z-4eta.txt", "r") as f_read:
        for line in f_read:
            line = line.strip()  #suppression du retour charriot en fin de ligne
            if line:
                xi = [float(elt) for elt in line.split("  ")]  #conversion en nb
                x.append(xi)
    
    return x

def voronoi_cell_volume(vor_3d):
    Volume=[]
    for i in range(len(vor_3d.regions)):
        Coord_ind_points=[]
        test_espace=0
        if -1 not in vor_3d.regions[i] and vor_3d.regions[i]!=[]:
            for j in range(len(vor_3d.regions[i])):
                if vor_3d.vertices[vor_3d.regions[i][j]][0]<0 or 2*np.pi<vor_3d.vertices[vor_3d.regions[i][j]][0] or vor_3d.vertices[vor_3d.regions[i][j]][1]<0 or 2*np.pi<vor_3d.vertices[vor_3d.regions[i][j]][1] or vor_3d.vertices[vor_3d.regions[i][j]][2]<0 or 2*np.pi<vor_3d.vertices[vor_3d.regions[i][j]][2]:
                    test_espace=1
            if test_espace==0:
                for j in range(len(vor_3d.regions[i])):
                    Coord_ind_points.append(vor_3d.vertices[vor_3d.regions[i][j]])
                Volume.append(ConvexHull(Coord_ind_points).volume)
    return Volume

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



def histogramme(tab, subdivise, logx, logy, bmin, bmax, legende, couleur, fig, chocolat):

    #compte=moyenne(tab)*len(tab)
    
    liste=np.copy(tab)
    if logx==1:
        for i in range(len(liste)):
            if liste[i]!=0:
                liste[i]=log10(liste[i])
        
    
    maxi=max(abs(max(liste)),abs(min(liste)))
    densi=np.zeros((subdivise+1))
    moy=np.zeros((subdivise+1))
    hist=[]
    x=[]
    
    #moyenne(liste)

    for i in range(len(liste)):
        if bmin<=abs(liste[i])<=bmax:
            #print(tab[i],abs(int((liste[i])*subdivise/maxi)))
            densi[ abs(int(liste[i]*subdivise/maxi)) ]+=1
            moy[ abs(int(liste[i]*subdivise/maxi)) ]+=tab[i] #liste[i] #
    

    n=maxi/subdivise#subdivise
    compte=0
    for i in range(subdivise):
        compte=compte+densi[i]*n
        
    if chocolat==1:
        print("sdfqsdf")
        print(histogramme(tab, subdivise,0,0,0,40, "XD LOL", couleur, -1, -1), "lolil")
        compte=histogramme(tab, subdivise,0,0,0,40, "XD LOL", couleur, -1, -1)
        print("0000000000000")
        
    for i in range(subdivise):
        if densi[i]!=0:
            x.append(abs(moy[i]/densi[i]))
            hist.append(abs(densi[i]/compte))
            
    if fig!=-1:
        plt.figure(fig)
        if logx==1:
            plt.xscale('log')
        if logy==1:
            plt.yscale('log')
        
        if logy==0 and logy==1:
            plt.axis([0-0.0001, max(x)+max(x)/50, 0-0.0001, max(hist)+max(hist)/50])
        
        plt.scatter(x,hist,s=2, label=legende, c=couleur)
        plt.legend()
    
    if chocolat==-1:
        return compte
    


def loi_de_Poisson(l, r):
    
    #plt.figure(60)
    #plt.yscale('log')
    #plt.xscale('log')
    
    p=[]
    px=[]
    for i in range(r+1):
        p.append(l**i*math.exp(-l)/(math.factorial(i)))
        px.append(i)
    
    plt.scatter(px, p, s=2, c='r',label="Poisson")
    plt.legend()










