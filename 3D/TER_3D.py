# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:04:05 2019

@author: Moli Oujia
"""

#import numpy as np
#import matplotlib.pyplot as plt
from scipy.spatial import Voronoi#, voronoi_plot_2d

from Tools_3D import voronoi_cell_volume, lis_point_3d, density_3d, ecrit_data, histogramme, moyenne, loi_de_Poisson#, lis_data, lis_point_3d_binaire
#from Tools import *


#fichier="random3d\pcl-ran000"
#fichier="points_3d.txt"
#couleur='c'
#"N512_st005\pcl-030000_z-4eta.txt"


def Fonction(fichier, couleur):
    
    #point=lis_point_3d_binaire(fichier)
    point=lis_point_3d(fichier)

    vor_3d = Voronoi(point)

    data=voronoi_cell_volume(vor_3d)
    ecrit_data("Vor_Vol.txt", data)
    
    histogramme(data, 200,1,1,0.1,1000, fichier, couleur, 2,0)
    histogramme(data, 200,0,0,0,1000, fichier, couleur, 3,0)
    plt.savefig('Hist_log.png')
    """   """
    
    dens=density_3d(point, 20)
    ecrit_data("Dens.txt", dens)
    
    histogramme(dens, int(max(dens)),0,1,0,40, "density", couleur, 6,0) 
    loi_de_Poisson(moyenne(dens), int(max(dens)))
    """
    histogramme(dens, int(max(dens)),1,1,0,40, "density", couleur, 4,1)
    loi_de_Poisson(moyenne(dens))
    
    histogramme(dens, int(max(dens)),0,0,0,40, "density", couleur, 5,0) 
    loi_de_Poisson(moyenne(dens))
    plt.savefig('Hist_log.png') 
    """   

    
Fonction("points_3d.txt",'c')
#Fonction("random3d\pcl-ran000",'c')
#loi_de_test(ecartype(data))

#print(voronoi_cell_area(vor_2d))
#voronoi_plot_2d(vor_2d)                      #affiche voronoi
