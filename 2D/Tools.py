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
import struct



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



#fichier="N512_st1\pcl-024000_z-4eta.txt"
#fichier="N1024_st1\pcl-0115000_z-4eta.txt"
#lis_point("N512_st1\pcl-024000_z-4eta.txt")
#lis_point("N1024_st1\pcl-0115000_z-4eta.txt")

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
    
    #plt.scatter(x,y,s=0.05,c='black')
    #plt.scatter(point[...][:,0],point[...][:,1],s=0.05,c='black')
    
    #plt.title('Nuage de points')
#    plt.xlabel('x')
#    plt.ylabel('y')
    plt.savefig('Particles.png')
    
    """ """
    point=[]
    
    if len(x)>80000:
        mz=max(z)
        for i in range(len(x)):
            if z[i]<mz/2:
                point.append([x[i],y[i]])
    else:
        for i in range(len(x)):
                point.append([x[i],y[i]])
    point=np.array(point)
    
    plt.scatter(point[...][:,0],point[...][:,1],s=0.05,c='black')

    return point


def plot_loghist(x, bins, couleur):
    plt.figure(1)
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins, log=True, density=True, color = couleur)
    plt.xscale('log')


def ecrit_point(fichier, point):
    fichier = open(fichier, "w")
    for i in range(len(point)):
        fichier.write(str(point[i][0])+ "   "+ str(point[i][1])+ "   "+str(point[i][1])+'\n')
    fichier.close()


def lis_data(fichier):
    x = []
    file = open(fichier)
    for line in file:
        x.append(float(line.rstrip()))
    return x


def ecrit_data(fichier, data):
    fichier = open(fichier, "w")
    for i in range(len(data)):
        fichier.write(str(data[i])+'\n')
    fichier.close()



def ecrit_data_binaire(fichier, data):
    try:
        with open(fichier, 'wb') as file:
            file.write(struct.pack('H', len(data)))
            for elem in data:
                file.write(struct.pack('d', elem))
    except IOError:
        print('Erreur d\'écriture.')


def lis_data_binaire(fichier):
    try:
        with open(fichier, 'rb') as file:
            n = struct.unpack('H', file.read(2))[0]
            data = []
            for i in range(n):
                raw = struct.unpack('d', file.read(8))
                data.append(raw[0])
        print(data, sep='\n')
    except IOError:
        print('Erreur de lecture.')


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
            Areanozero.append(Area[i][0])
        
    return Areanozero, Area


def voronoi_cell_area_test(vor_2d,max_ind_point):
    Area=np.zeros((max_ind_point,1))
    
    for i in range(max_ind_point):
        Coord_ind_points=[]
        for j in range(len(vor_2d.regions[vor_2d.point_region[i]])):
            Coord_ind_points.append(vor_2d.vertices[vor_2d.regions[vor_2d.point_region[i]][j]])
        Area[i]=ConvexHull(Coord_ind_points).area
    
    return Area[:,0]


def create_border(points):
    B=2*np.pi#1
    P=[]
    for i in range(len(points)):
        if points[i][0]<B:
            P.append([2*np.pi+points[i][0],points[i][1]])
        if points[i][0]>2*np.pi-B:
            P.append([-2*np.pi+points[i][0],points[i][1]])
        if points[i][1]<B:
            P.append([points[i][0],2*np.pi+points[i][1]])
        if points[i][1]>2*np.pi-B:
            P.append([points[i][0],-2*np.pi+points[i][1]])
        if points[i][0]<B and points[i][1]<B:
             P.append([2*np.pi+points[i][0],2*np.pi+points[i][1]])
        if points[i][0]<B and points[i][1]>2*np.pi-B:
             P.append([2*np.pi+points[i][0],-2*np.pi+points[i][1]])
        if points[i][0]>2*np.pi-B and points[i][1]>2*np.pi-B:
             P.append([-2*np.pi+points[i][0],-2*np.pi+points[i][1]])
        if points[i][0]>2*np.pi-B and points[i][1]<B:
             P.append([-2*np.pi+points[i][0],2*np.pi+points[i][1]])
        
    return P


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
    
    #plt.yscale('log')
    #plt.xscale('log')
    
    p=[]
    px=[]
    for i in range(r+1):
        p.append(l**i*math.exp(-l)/(math.factorial(i)))
        px.append(i)
    
    plt.scatter(px,p,s=2, c='black',label="Poisson")
    plt.legend()

import scipy.special as sps


def loi_de_Poisson_conti(l,r):
    
    #plt.yscale('log')
    #plt.xscale('log')
    x=np.arange(0.5, r + 0.01, 0.01)
    p=l**x*math.exp(-l)/(sps.gamma(x+1))
    
    plt.plot(x, p , c='b',label="Poisson")
    plt.legend()
    
    
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

    plt.scatter(np.array(x),np.array(hist)/4,s=2, label=legende, c=couleur)
    plt.legend()
    
    
    #loi_normal( abs(moyenne(log10(tab))), abs(ecartype(log10(tab))), -3, 3, 1 )
    
    #return x,hist
    


def My_hist_dens(data, legende, couleur, fig, si):
    
    h = np.histogram(data, bins=int(max(data)))
    m1=[]
    m2=[]
    for i in range(len(h[1])-1):
        #m2.append(h[1][i])
        m2.append((h[1][i]+h[1][i+1])/2)
        m1.append(h[0][i])

    M1=[]
    M2=[]
    for i in range(len(m2)):
        if m1[i] != 0:
            M1.append(m1[i])
            M2.append(m2[i])
    
    som=0
    for i in range(len(M1)-1):
        som=som+abs((M2[i+1]-M2[i])*M1[i])
    
    m1=np.array(m1)
    m2=np.array(m2)
    M1=np.array(M1)
    M2=np.array(M2)
    
    plt.figure(fig)
    #plt.plot(M2,M1/som, label=legende, c=couleur, marker=si)
    plt.scatter(M2,M1/som,c=couleur, label=legende, marker=si, s=15)
    
    """   """
    plt.xscale('log')
    plt.yscale('log')
    
    plt.legend()




def My_hist_vor(data, legende, couleur, fig, si):
    
    h = np.histogram(np.log10(data), bins=200)
    m1=[]
    m2=[]
    for i in range(len(h[1])-1):
            m2.append(10**((h[1][i]+h[1][i+1])/2))
            m1.append(h[0][i])

    som=0
    for i in range(len(m1)-1):
        som=som+abs((m2[i+1]-m2[i])*m1[i])
    
    m1=np.array(m1)
    m2=np.array(m2)
    
    plt.figure(fig)
    plt.plot(m2,m1/som, label=legende, c=couleur)
    plt.legend()
    plt.scatter(m2[::20],m1[::20]/som, c=couleur, marker=si)
    plt.xscale('log')
    plt.yscale('log')




def My_hist_dens_poisson(data, legende, couleur, fig):
    
    h = np.histogram(data, bins=int(max(data)))
    m1=[]
    m2=[]
    for i in range(len(h[1])-1):
        #m2.append(h[1][i])
        m2.append(h[1][i])
        m1.append(h[0][i])

    
    plt.figure(fig)
    #plt.plot(M2,M1/som, label=legende, c=couleur, marker=si)
    plt.scatter(m2,np.array(m1)/(1000*1000),c=couleur, label=legende, s=0.5)
    
    plt.legend()




def density_density(point, N):
    
    
    nombre=np.zeros((N,N))
    for i in range(len(point)):
        nombre[int(point[i][0]*N/(2*np.pi))][int(point[i][1]*N/(2*np.pi))]+=1
    
    dens=np.zeros((10,10))
    for i in range(N):
        for j in range(N):
            if nombre[i][j]<4:
                dens[i//10][j//10]+=1
            
    
    tableau=[]
    for i in range(10):
        for j in range(10):
            if dens[j][i]!=0:
                tableau.append(dens[j][i])        
            else:
                tableau.append(0)



    return tableau










