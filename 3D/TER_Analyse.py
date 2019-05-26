# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:30:17 2019

@author: Moli Oujia
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
from numpy import random as npr  
from scipy import stats as st  


fig=1


def moyenne(tableau):
    somme=0
    for i in range(len(tableau)):
        somme=somme+tableau[i]
    return somme / len(tableau)

    
def variance(tableau):
    m=moyenne(tableau)
    return moyenne([(x-m)**2 for x in tableau])
    
    
def ecartype(tableau):
    return variance(tableau)**0.5


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


def My_hist_vor(data, legende, couleur, fig):
    
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
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()


def My_hist_vor_pos_neg(data, legende, couleur, fig):
    
    datapos=[]
    dataneg=[]
    
    for i in range(len(data)):
        if data[i]>0:
            datapos.append(data[i])
        else:
            dataneg.append(data[i])
        
    
    h = np.histogram(np.log10(datapos), bins=200)
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
    plt.plot(m2,m1/som, label='positif', c='r')
    plt.legend()
    
    
    
    h = np.histogram(np.log10(-np.array(dataneg)), bins=200)
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
    plt.plot(m2,m1/som, label='negatif', c='b')
    plt.legend()
    
    plt.xscale('log')
    plt.yscale('log')


def My_hist_dens(data, legende, couleur, fig):
    
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
    #plt.plot(M2,M1/som, label=legende, c=couleur)
    plt.scatter(M2,M1/som,s=0.5,c=couleur, label=legende)
    
    """   """
    plt.xscale('log')
    plt.yscale('log')
    
    plt.legend()


def aff_tout_st(cha, fig, logx, logy):

    
    data1=lis_data_binaire("Densité_3D/3D_N512-st5-02"+str(cha)+"000-dens")
    data2=lis_data_binaire("Densité_3D/3D_N512-st2-02"+str(cha)+"000-dens")
    data3=lis_data_binaire("Densité_3D/3D_N512-st1-02"+str(cha)+"000-dens")
    data4=lis_data_binaire("Densité_3D/3D_N512-st05-02"+str(cha)+"000-dens")
    data5=lis_data_binaire("Densité_3D/3D_N512-st02-02"+str(cha)+"000-dens")
    data6=lis_data_binaire("Densité_3D/3D_N512-st01-02"+str(cha)+"000-dens")
    data7=lis_data_binaire("Densité_3D/3D_N512-st005-02"+str(cha)+"000-dens")
    
    """
    data1=lis_data_binaire("Densité_3D/3D_N512-st5-02"+str(cha)+"000-240cube-dens")
    data2=lis_data_binaire("Densité_3D/3D_N512-st2-02"+str(cha)+"000-240cube-dens")
    data3=lis_data_binaire("Densité_3D/3D_N512-st1-02"+str(cha)+"000-240cube-dens")
    data4=lis_data_binaire("Densité_3D/3D_N512-st05-02"+str(cha)+"000-240cube-dens")
    data5=lis_data_binaire("Densité_3D/3D_N512-st02-02"+str(cha)+"000-240cube-dens")
    data6=lis_data_binaire("Densité_3D/3D_N512-st01-02"+str(cha)+"000-240cube-dens")
    data7=lis_data_binaire("Densité_3D/3D_N512-st005-02"+str(cha)+"000-240cube-dens")
    
    
    data1=lis_data_binaire("Densité_3D/3D_N512-st5-02"+str(cha)+"000-512cube-dens")
    data2=lis_data_binaire("Densité_3D/3D_N512-st2-02"+str(cha)+"000-512cube-dens")
    data3=lis_data_binaire("Densité_3D/3D_N512-st1-02"+str(cha)+"000-512cube-dens")
    data4=lis_data_binaire("Densité_3D/3D_N512-st05-02"+str(cha)+"000-512cube-dens")
    data5=lis_data_binaire("Densité_3D/3D_N512-st02-02"+str(cha)+"000-512cube-dens")
    data6=lis_data_binaire("Densité_3D/3D_N512-st01-02"+str(cha)+"000-512cube-dens")
    data7=lis_data_binaire("Densité_3D/3D_N512-st005-02"+str(cha)+"000-512cube-dens")
    """
    
    My_hist_dens(data1,"st5",'pink',fig)
    My_hist_dens(data2,"st2",'y',fig)
    My_hist_dens(data3,"st1",'r',fig)
    My_hist_dens(data4,"st05",'c',fig)
    My_hist_dens(data5,"st02",'g',fig)
    My_hist_dens(data6,"st01",'b',fig)
    My_hist_dens(data7,"st005",'purple',fig)
    
    if logx == 1:
        plt.xscale('log')
    if logy == 1:
        plt.yscale('log')

#plt.yscale('linear')

def aff_un_st_dens(cha, fig):

    data1=lis_data_binaire("Densité_3D/3D_N512-st"+str(cha)+"-021000-dens")
    data2=lis_data_binaire("Densité_3D/3D_N512-st"+str(cha)+"-022000-dens")
    data3=lis_data_binaire("Densité_3D/3D_N512-st"+str(cha)+"-024000-dens")
    data4=lis_data_binaire("Densité_3D/3D_N512-st"+str(cha)+"-025000-dens")
    data5=lis_data_binaire("Densité_3D/3D_N512-st"+str(cha)+"-026000-dens")
    data6=lis_data_binaire("Densité_3D/3D_N512-st"+str(cha)+"-028000-dens")
    data7=lis_data_binaire("Densité_3D/3D_N512-st"+str(cha)+"-029000-dens")
    
    My_hist_dens(data1,cha,'pink',fig)
    My_hist_dens(data2,cha,'y',fig)
    My_hist_dens(data3,cha,'r',fig)
    My_hist_dens(data4,cha,'c',fig)
    My_hist_dens(data5,cha,'g',fig)
    My_hist_dens(data6,cha,'b',fig)
    My_hist_dens(data7,cha,'purple',fig)


def main_dens(logx, logy):
    
    aff_tout_st('4', 2, logx, logy)
    
    aff_un_st_dens('005',3)
    aff_un_st_dens('02',4)
    aff_un_st_dens('1',5)
    aff_un_st_dens('5',6)


def aff_tout_st_vor(fig):
    data=lis_data_binaire("Voronoi/3D_N512-st5-024000-vor")
    My_hist_vor(data, "5", 'pink', fig)
    data=lis_data_binaire("Voronoi/3D_N512-st2-024000-vor")
    My_hist_vor(data, "2", 'y', fig)
    data=lis_data_binaire("Voronoi/3D_N512-st1-024000-vor")
    My_hist_vor(data, "1", 'r', fig)
    data=lis_data_binaire("Voronoi/3D_N512-st05-024000-vor")
    My_hist_vor(data, "05", 'c', fig)
    data=lis_data_binaire("Voronoi/3D_N512-st02-024000-vor")
    My_hist_vor(data, "02", 'g', fig)
    data=lis_data_binaire("Voronoi/3D_N512-st01-024000-vor")
    My_hist_vor(data, "01", 'b', fig)
    data=lis_data_binaire("Voronoi/3D_N512-st005-024000-vor")
    My_hist_vor(data, "005", 'purple', fig)
    
    
    
    
    data1=lis_data_binaire("Voronoi/3D_N512-st1-024000-vor")
    My_hist_vor(data1, "1", 'r', fig)
    data2=lis_data_binaire("Voronoi/3D_N512-st1-024000-vor-vel")
    My_hist_vor(data2, "1 vel", 'black', fig)
    m1=moyenne(data1)
    m2=moyenne(data2)
    m3=(m1-m2)/m2
    
    D= 1/((np.array(data2)+np.array(data1))/2)*(np.array(data2)-np.array(data1))/0.001
    My_hist_vor(abs(D), "D", 'black', fig)
    My_hist_vor_pos_neg(D, "D", 'black', fig)
    
    
    
    data=lis_data_binaire("rand-vor")
    moy=moyenne(np.array(data))
    import scipy.special as sps
    m = moyenne(np.array(data)/moy)
    s = ecartype(np.array(data)/moy)
    bins = np.logspace(np.log10(0.04),np.log10(4.8),10000)
    shape, scale = 1/s**2, s**2
    #shape, scale = 6.7, s**2
    y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
    My_hist_vor(np.array(data)/moy, "rand", 'black', fig)
    plt.plot(bins, y, linewidth=2, label='Gamma')
    
    



#plt.yscale('log')
#plt.yscale('linear')


#fichier="Voronoi/3D_N512-st005-024000-vor"
def aff_un_st_vor_log(fichier):
        
    plt.figure(2)
    
    #data=lis_data_binaire("3D_N512-st1-024000-vor")
    data=lis_data_binaire(fichier)
    
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
    
    mu, sigma = moyenne(np.log(np.array(data)))+ ecartype(np.log(np.array(data)))**2, ecartype(np.log(np.array(data)))
    
    plt.plot(m2,m1/som, label="PDF")
    
    x = np.logspace(np.log10(1*10**(-6)),np.log10(2*10**(-4)),10000)
    #x = np.logspace(np.log10(10**(-7))+1,log10(0.0002),10000)
    pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
    plt.plot(x, pdf, linewidth=2, label='Normal law')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()


def variation():
    
    data1=lis_data_binaire("Voronoi/3D_N512-st1-024000-vor")
    data2=lis_data_binaire("Voronoi/3D_N512-st1-024000-vor-vel")
    
    data=np.array(data1)-np.array(data2)
    
    point=lis_point_3d_binaire('pcl-024000.pos')
    
    xp=[]
    yp=[]
    
    xn=[]
    yn=[]
    
    for i in range(len(point)):
        if point[i][0]<2*np.pi/200:
            if data[i]>0:
                xp.append(point[i][1])
                yp.append(point[i][2])
            elif data[i]<0:
                xn.append(point[i][1])
                yn.append(point[i][2])
                

    plt.figure(1,figsize=(6,6), dpi=80)
    plt.scatter(xp+xn,yp+yn,s=0.01,c='black')
    
    
    plt.figure(2,figsize=(10,10), dpi=80)
    plt.scatter(xp,yp,s=0.01,c='r')
    plt.scatter(xn,yn,s=0.01,c='b')



    
"""
def aff_un_st_vor():
    
    data=lis_data_binaire("3D_N512-st1-024000-vor")
    
    h = np.histogram(np.log10(data), bins=200)
    m1=[]
    m2=[]
    for i in range(len(h[1])-1):
            m2.append((h[1][i]+h[1][i+1])/2)
            m1.append(h[0][i])
    
    
    som=0
    for i in range(len(m1)-1):
        som=som+abs((m2[i+1]-m2[i])*m1[i])
    
    #m1.reverse()
    m1=np.array(m1)
    m2=np.array(m2)
        
    n = 2000000
    m, sig = moyenne(np.log10(data)), ecartype(np.log10(data))
    X = npr.normal(m, sig, n)
    x = np.arange(X.min(), X.max() + 0.01, 0.01)
    
    plt.plot(m2,m1/som, label="PDF")
    plt.plot(x, st.norm.pdf(x, m, sig), linewidth=1, label='Normal law')
    plt.yscale('log')
    plt.legend()
"""


"""
plot_loghist(data, 200, 'test', 'r', 3)

histogramme(data, 200,1,1,0.1,1000, fichier, couleur, 2)
#My_hist_dens(dataaa[0], "rrr", 'r', 6,'vor')
#My_hist_dens(dens, "rrr", 'r', 6,'dens')

plt.figure(5)
plt.hist(log10(data), bins=200, normed=1, color = 'r', log=True)
"""

"""   
    m1=np.array(m1)
    mm2=max(m2)
    mmx=max(x)
    m2=np.array(m2)/mm2*mmx
"""







"""


def plot_loghist(x, bins, legende, couleur,fig):
    plt.figure(1)
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    n=plt.hist(x, bins=logbins, log=True, density=True, color = couleur)
    plt.xscale('log')
    #plt.close()
    t=[]
    for i in range(len(n[1])-1):
        t.append((n[1][i]+n[1][i+1])/2)
    
    m1=[]
    m2=[]
    
    for i in range(len(t)):
        if n[0][i]!=0:
            m1.append(n[0][i])
            m2.append(t[i])
    legende='test'
    plt.figure(fig)
    
    plt.loglog(m2,m1, label=legende, c=couleur)
    plt.xscale('log')
    plt.legend()





def My_hist_dens(data, legende, couleur, fig):
    plt.figure(666)    
    logbin = np.logspace(0,np.log10(int(max(data))),200)
    n = plt.hist(data, bins=logbin, log=True, density=True, color = 'c')
    plt.xscale('log')
    plt.close()
    t=[]
    for i in range(len(n[1])-1):
        t.append((n[1][i]+n[1][i+1])/2)
    
    m1=[]
    m2=[]
    
    for i in range(len(t)):
        if n[0][i]!=0:
            m1.append(n[0][i])
            m2.append(t[i])
    
    plt.figure(fig)
    plt.loglog(m2,m1, label=legende, c=couleur)
    plt.legend()
"""

