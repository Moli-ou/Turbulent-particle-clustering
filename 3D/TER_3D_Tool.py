# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:04:05 2019

@author: Moli Oujia
"""


from scipy.spatial import Voronoi#, voronoi_plot_2d
import numpy as np
from scipy.spatial import ConvexHull
import struct


def ecrit_data(fichier, data):
    fichier = open(fichier, "w")
    for i in range(len(data)):
        fichier.write(str(data[i])+'\n')
    fichier.close()


def density_3d(point, N):
    
    nombre=np.zeros((N,N,N))
    for i in range(len(point)):
        nombre[int(point[i][0]%(2*np.pi)*N/(2*np.pi))][int(point[i][1]%(2*np.pi)*N/(2*np.pi))][int(point[i][2]%(2*np.pi)*N/(2*np.pi))]+=1
    
    tableau=[]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if nombre[j][i][k]!=0:
                    tableau.append(nombre[j][i][k])
                else:
                    tableau.append(0)

    return tableau


def lis_point_3d_binaire(fichier):
    
    Donner=[]
        
    f = open(fichier, "rb")
    
    test='12345678'
    
    try:
        while len(test)!=0:
            test=f.read(8)
            if len(test)==8:
                Donner.append( struct.unpack('<d',test)[0] )
             
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


def ecrit_data_binaire(fichier, data):
    try:
        with open(fichier, 'wb') as file:
            for elem in data:
                file.write(struct.pack('<d', elem))
    except IOError:
        print('Erreur d\'Ã©criture.')


def lis_data_binaire(fichier):
    R='12345678'
    data = []
    try:
        with open(fichier, 'rb') as file:
            while len(R)!=0:
                R=file.read(8)
                if len(R)==8:
                    data.append( struct.unpack('<d', R)[0] )
                else:
                    return data
    except IOError:
        print('Erreur de lecture.')
    return data


def voronoi_cell_volume(vor_3d,max_ind_point):
    Volume=np.zeros((max_ind_point,1))
    
    for i in range(max_ind_point):
        Coord_ind_points=[]
        for j in range(len(vor_3d.regions[vor_3d.point_region[i]])):
            Coord_ind_points.append(vor_3d.vertices[vor_3d.regions[vor_3d.point_region[i]][j]])
        Volume[i]=ConvexHull(Coord_ind_points).volume
    
    return Volume[:,0]


def create_border(points):
    B=0.5
    P=[]
    for i in range(len(points)):
        for j in range(3):
            #face
            if points[i][j]<B:
                l=np.copy(points[i])
                l[j]=2*np.pi+points[i][j]
                P.append(l)
            if points[i][j]>2*np.pi-B:
                l=np.copy(points[i])
                l[j]=-2*np.pi+points[i][j]
                P.append(l)
            #arete
            if points[i][j]<B and points[i][(j+1)%3]<B:
                l=np.copy(points[i])
                l[j]=2*np.pi+points[i][j]
                l[(j+1)%3]=2*np.pi+points[i][(j+1)%3]
                P.append(l)
            if points[i][j]>2*np.pi-B and points[i][(j+1)%3]>2*np.pi-B:
                l=np.copy(points[i])
                l[j]=-2*np.pi+points[i][j]
                l[(j+1)%3]=-2*np.pi+points[i][(j+1)%3]
                P.append(l)
            if points[i][j]>2*np.pi-B and points[i][(j+1)%3]<B:
                l=np.copy(points[i])
                l[j]=-2*np.pi+points[i][j]
                l[(j+1)%3]=2*np.pi+points[i][(j+1)%3]
                P.append(l)
            if points[i][j]<B and points[i][(j+1)%3]>2*np.pi-B:
                l=np.copy(points[i])
                l[j]=2*np.pi+points[i][j]
                l[(j+1)%3]=-2*np.pi+points[i][(j+1)%3]
                P.append(l)
            #sommet
        if points[i][0]<B and points[i][1]<B and points[i][2]<B :
            P.append([ 2*np.pi+points[i][0], 2*np.pi+points[i][1], 2*np.pi+points[i][2]])
        if points[i][0]<B and points[i][1]<B and points[i][2]>2*np.pi-B :
            P.append([ 2*np.pi+points[i][0], 2*np.pi+points[i][1], -2*np.pi+points[i][2]])
        if points[i][0]<B and points[i][1]>2*np.pi-B and points[i][2]<B :
            P.append([ 2*np.pi+points[i][0], -2*np.pi+points[i][1], 2*np.pi+points[i][2]])
        if points[i][0]<B and points[i][1]>2*np.pi-B and points[i][2]>2*np.pi-B :
            P.append([ 2*np.pi+points[i][0], -2*np.pi+points[i][1], -2*np.pi+points[i][2]])
            
        if points[i][0]>2*np.pi-B and points[i][1]<B and points[i][2]<B :
            P.append([ -2*np.pi+points[i][0], 2*np.pi+points[i][1], 2*np.pi+points[i][2]])
        if points[i][0]>2*np.pi-B and points[i][1]<B and points[i][2]>2*np.pi-B :
            P.append([ -2*np.pi+points[i][0], 2*np.pi+points[i][1], -2*np.pi+points[i][2]])
        if points[i][0]>2*np.pi-B and points[i][1]>2*np.pi-B and points[i][2]<B :
            P.append([ -2*np.pi+points[i][0], -2*np.pi+points[i][1], 2*np.pi+points[i][2]])
        if points[i][0]>2*np.pi-B and points[i][1]>2*np.pi-B and points[i][2]>2*np.pi-B :
            P.append([ -2*np.pi+points[i][0], -2*np.pi+points[i][1], -2*np.pi+points[i][2]])
                      
    return P


def Fonction(fichier, nom_Vor, nom_Den):
    
    #point=lis_point_3d(fichier)
    #point=lis_point_3d_binaire(fichier)

    point=np.random.random_sample((15000000,3))	
    
    for i in range(len(point)):
        point[i][0]=point[i][0]*(2*np.pi)
        point[i][1]=point[i][1]*(2*np.pi)
        point[i][2]=point[i][2]*(2*np.pi)

    """   """
    dens=density_3d(point, 100)
    ecrit_data_binaire(nom_Den, dens)
    dens=[]
    
    
    C=create_border(point)
    
    m1=np.transpose(C)[0]
    m2=np.transpose(point)[0]
    l1=m1.tolist()
    l2=m2.tolist()
    
    Q1=l2+l1
    
    m3=np.transpose(C)[1]
    m4=np.transpose(point)[1]
    l3=m3.tolist()
    l4=m4.tolist()
    
    Q2=l4+l3
    
    m5=np.transpose(C)[2]
    m6=np.transpose(point)[2]
    l5=m5.tolist()
    l6=m6.tolist()
    
    Q3=l6+l5
    
    Po=[]
    for i in range(len(Q1)):#len(point),
        Po.append([Q1[i],Q2[i],Q3[i]])

    vor_3d = Voronoi(Po)
    data=voronoi_cell_volume(vor_3d,len(point))
    ecrit_data_binaire(nom_Vor, data)


 


Fonction("Data_particules/N512_st1/pcl-024000.pos", "rand-vor", "rand-dens")




"""
Fonction("Data_particules/N512_st005/pcl-024000.pos", "3D_N512-st005-024000-512cube-vor", "3D_N512-st005-024000-512cube-dens")
Fonction("Data_particules/N512_st01/pcl-024000.pos", "3D_N512-st01-024000-512cube-vor", "3D_N512-st01-024000-512cube-dens")
Fonction("Data_particules/N512_st02/pcl-024000.pos", "3D_N512-st02-024000-512cube-vor", "3D_N512-st02-024000-512cube-dens")
Fonction("Data_particules/N512_st05/pcl-024000.pos", "3D_N512-st05-024000-512cube-vor", "3D_N512-st05-024000-512cube-dens")
Fonction("Data_particules/N512_st1/pcl-024000.pos", "3D_N512-st1-024000-512cube-vor", "3D_N512-st1-024000-512cube-dens")
Fonction("Data_particules/N512_st2/pcl-024000.pos", "3D_N512-st2-024000-512cube-vor", "3D_N512-st2-024000-512cube-dens")
Fonction("Data_particules/N512_st5/pcl-024000.pos", "3D_N512-st5-024000-512cube-vor", "3D_N512-st5-024000-512cube-dens")
"""


"""

Fonction("Data_particules/N512_st005/pcl-030000.pos", "3D_N512-st005-030000-vor", "3D_N512-st005-030000-dens")
Fonction("Data_particules/N512_st005/pcl-029000.pos", "3D_N512-st005-029000-vor", "3D_N512-st005-029000-dens")
Fonction("Data_particules/N512_st005/pcl-028000.pos", "3D_N512-st005-028000-vor", "3D_N512-st005-028000-dens")
Fonction("Data_particules/N512_st005/pcl-027000.pos", "3D_N512-st005-027000-vor", "3D_N512-st005-027000-dens")
Fonction("Data_particules/N512_st005/pcl-026000.pos", "3D_N512-st005-026000-vor", "3D_N512-st005-026000-dens")
Fonction("Data_particules/N512_st005/pcl-025000.pos", "3D_N512-st005-025000-vor", "3D_N512-st005-025000-dens")
Fonction("Data_particules/N512_st005/pcl-024000.pos", "3D_N512-st005-024000-vor", "3D_N512-st005-024000-dens")
Fonction("Data_particules/N512_st005/pcl-023000.pos", "3D_N512-st005-023000-vor", "3D_N512-st005-023000-dens")
Fonction("Data_particules/N512_st005/pcl-022000.pos", "3D_N512-st005-022000-vor", "3D_N512-st005-022000-dens")
Fonction("Data_particules/N512_st005/pcl-021000.pos", "3D_N512-st005-021000-vor", "3D_N512-st005-021000-dens")



Fonction("Data_particules/N512_st01/pcl-030000.pos", "3D_N512-st01-030000-vor", "3D_N512-st01-030000-dens")
Fonction("Data_particules/N512_st01/pcl-029000.pos", "3D_N512-st01-029000-vor", "3D_N512-st01-029000-dens")
Fonction("Data_particules/N512_st01/pcl-028000.pos", "3D_N512-st01-028000-vor", "3D_N512-st01-028000-dens")
Fonction("Data_particules/N512_st01/pcl-027000.pos", "3D_N512-st01-027000-vor", "3D_N512-st01-027000-dens")
Fonction("Data_particules/N512_st01/pcl-026000.pos", "3D_N512-st01-026000-vor", "3D_N512-st01-026000-dens")
Fonction("Data_particules/N512_st01/pcl-025000.pos", "3D_N512-st01-025000-vor", "3D_N512-st01-025000-dens")
Fonction("Data_particules/N512_st01/pcl-024000.pos", "3D_N512-st01-024000-vor", "3D_N512-st01-024000-dens")
Fonction("Data_particules/N512_st01/pcl-023000.pos", "3D_N512-st01-023000-vor", "3D_N512-st01-023000-dens")
Fonction("Data_particules/N512_st01/pcl-022000.pos", "3D_N512-st01-022000-vor", "3D_N512-st01-022000-dens")
Fonction("Data_particules/N512_st01/pcl-021000.pos", "3D_N512-st01-021000-vor", "3D_N512-st01-021000-dens")



Fonction("Data_particules/N512_st02/pcl-030000.pos", "3D_N512-st02-030000-vor", "3D_N512-st02-030000-dens")
Fonction("Data_particules/N512_st02/pcl-029000.pos", "3D_N512-st02-029000-vor", "3D_N512-st02-029000-dens")
Fonction("Data_particules/N512_st02/pcl-028000.pos", "3D_N512-st02-028000-vor", "3D_N512-st02-028000-dens")
Fonction("Data_particules/N512_st02/pcl-027000.pos", "3D_N512-st02-027000-vor", "3D_N512-st02-027000-dens")
Fonction("Data_particules/N512_st02/pcl-026000.pos", "3D_N512-st02-026000-vor", "3D_N512-st02-026000-dens")
Fonction("Data_particules/N512_st02/pcl-025000.pos", "3D_N512-st02-025000-vor", "3D_N512-st02-025000-dens")
Fonction("Data_particules/N512_st02/pcl-024000.pos", "3D_N512-st02-024000-vor", "3D_N512-st02-024000-dens")
Fonction("Data_particules/N512_st02/pcl-023000.pos", "3D_N512-st02-023000-vor", "3D_N512-st02-023000-dens")
Fonction("Data_particules/N512_st02/pcl-022000.pos", "3D_N512-st02-022000-vor", "3D_N512-st02-022000-dens")
Fonction("Data_particules/N512_st02/pcl-021000.pos", "3D_N512-st02-021000-vor", "3D_N512-st02-021000-dens")



Fonction("Data_particules/N512_st05/pcl-030000.pos", "3D_N512-st05-030000-vor", "3D_N512-st05-030000-dens")
Fonction("Data_particules/N512_st05/pcl-029000.pos", "3D_N512-st05-029000-vor", "3D_N512-st05-029000-dens")
Fonction("Data_particules/N512_st05/pcl-028000.pos", "3D_N512-st05-028000-vor", "3D_N512-st05-028000-dens")
Fonction("Data_particules/N512_st05/pcl-027000.pos", "3D_N512-st05-027000-vor", "3D_N512-st05-027000-dens")
Fonction("Data_particules/N512_st05/pcl-026000.pos", "3D_N512-st05-026000-vor", "3D_N512-st05-026000-dens")
Fonction("Data_particules/N512_st05/pcl-025000.pos", "3D_N512-st05-025000-vor", "3D_N512-st05-025000-dens")
Fonction("Data_particules/N512_st05/pcl-024000.pos", "3D_N512-st05-024000-vor", "3D_N512-st05-024000-dens")
Fonction("Data_particules/N512_st05/pcl-023000.pos", "3D_N512-st05-023000-vor", "3D_N512-st05-023000-dens")
Fonction("Data_particules/N512_st05/pcl-022000.pos", "3D_N512-st05-022000-vor", "3D_N512-st05-022000-dens")
Fonction("Data_particules/N512_st05/pcl-021000.pos", "3D_N512-st05-021000-vor", "3D_N512-st05-021000-dens")



Fonction("Data_particules/N512_st1/pcl-030000.pos", "3D_N512-st1-030000-vor", "3D_N512-st1-030000-dens")
Fonction("Data_particules/N512_st1/pcl-029000.pos", "3D_N512-st1-029000-vor", "3D_N512-st1-029000-dens")
Fonction("Data_particules/N512_st1/pcl-028000.pos", "3D_N512-st1-028000-vor", "3D_N512-st1-028000-dens")
Fonction("Data_particules/N512_st1/pcl-027000.pos", "3D_N512-st1-027000-vor", "3D_N512-st1-027000-dens")
Fonction("Data_particules/N512_st1/pcl-026000.pos", "3D_N512-st1-026000-vor", "3D_N512-st1-026000-dens")
Fonction("Data_particules/N512_st1/pcl-025000.pos", "3D_N512-st1-025000-vor", "3D_N512-st1-025000-dens")
Fonction("Data_particules/N512_st1/pcl-024000.pos", "3D_N512-st1-024000-vor", "3D_N512-st1-024000-dens")
Fonction("Data_particules/N512_st1/pcl-023000.pos", "3D_N512-st1-023000-vor", "3D_N512-st1-023000-dens")
Fonction("Data_particules/N512_st1/pcl-022000.pos", "3D_N512-st1-022000-vor", "3D_N512-st1-022000-dens")
Fonction("Data_particules/N512_st1/pcl-021000.pos", "3D_N512-st1-021000-vor", "3D_N512-st1-021000-dens")



Fonction("Data_particules/N512_st2/pcl-030000.pos", "3D_N512-st2-030000-vor", "3D_N512-st2-030000-dens")
Fonction("Data_particules/N512_st2/pcl-029000.pos", "3D_N512-st2-029000-vor", "3D_N512-st2-029000-dens")
Fonction("Data_particules/N512_st2/pcl-028000.pos", "3D_N512-st2-028000-vor", "3D_N512-st2-028000-dens")
Fonction("Data_particules/N512_st2/pcl-027000.pos", "3D_N512-st2-027000-vor", "3D_N512-st2-027000-dens")
Fonction("Data_particules/N512_st2/pcl-026000.pos", "3D_N512-st2-026000-vor", "3D_N512-st2-026000-dens")
Fonction("Data_particules/N512_st2/pcl-025000.pos", "3D_N512-st2-025000-vor", "3D_N512-st2-025000-dens")
Fonction("Data_particules/N512_st2/pcl-024000.pos", "3D_N512-st2-024000-vor", "3D_N512-st2-024000-dens")
Fonction("Data_particules/N512_st2/pcl-023000.pos", "3D_N512-st2-023000-vor", "3D_N512-st2-023000-dens")
Fonction("Data_particules/N512_st2/pcl-022000.pos", "3D_N512-st2-022000-vor", "3D_N512-st2-022000-dens")
Fonction("Data_particules/N512_st2/pcl-021000.pos", "3D_N512-st2-021000-vor", "3D_N512-st2-021000-dens")



Fonction("Data_particules/N512_st5/pcl-030000.pos", "3D_N512-st5-030000-vor", "3D_N512-st5-030000-dens")
Fonction("Data_particules/N512_st5/pcl-029000.pos", "3D_N512-st5-029000-vor", "3D_N512-st5-029000-dens")
Fonction("Data_particules/N512_st5/pcl-028000.pos", "3D_N512-st5-028000-vor", "3D_N512-st5-028000-dens")
Fonction("Data_particules/N512_st5/pcl-027000.pos", "3D_N512-st5-027000-vor", "3D_N512-st5-027000-dens")
Fonction("Data_particules/N512_st5/pcl-026000.pos", "3D_N512-st5-026000-vor", "3D_N512-st5-026000-dens")
Fonction("Data_particules/N512_st5/pcl-025000.pos", "3D_N512-st5-025000-vor", "3D_N512-st5-025000-dens")
Fonction("Data_particules/N512_st5/pcl-024000.pos", "3D_N512-st5-024000-vor", "3D_N512-st5-024000-dens")
Fonction("Data_particules/N512_st5/pcl-023000.pos", "3D_N512-st5-023000-vor", "3D_N512-st5-023000-dens")
Fonction("Data_particules/N512_st5/pcl-022000.pos", "3D_N512-st5-022000-vor", "3D_N512-st5-022000-dens")
Fonction("Data_particules/N512_st5/pcl-021000.pos", "3D_N512-st5-021000-vor", "3D_N512-st5-021000-dens")



"""








