# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 00:09:49 2019

@author: Moli Oujia
"""
#100000        2 sec
#1000000       16  sec
#10000000      3  min
#100000000     impossible   
n=1000


#import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from Tools import *
#from numpy import random as npr  
#from scipy import stats as st  
import time

from numpy import random as npr  
from scipy import stats as st  
import scipy as sp 
from scipy.stats import gamma

from scipy.stats import skewnorm

import matplotlib.pyplot as plt
#dataa = [0,7,2,5,1,3,5,6,3,4,1,8,9.5,1,4,2,6.5,3,8.5,4.5,5,3.5,0.5,]

t1=time.time()

#points_2d = np.random.random_sample((n,2))


#points_2d = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],[2, 0], [2, 1], [2, 2]])
#points_2d = np.array([[1, 1], [1, 5], [2, 5], [5, 2], [5, 1]])
#points_2d = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],[2, 0], [2, 1], [2, 2], [1, 5], [2, 5], [5, 2], [5, 1]])

"""
points_2d=[]
for i in range(n):
    points_2d.append([np.random.power(5),np.random.power(5)])
""" 

points_2d=lis_point("poisson.txt")


vor_2d = Voronoi(points_2d)

#voronoi_plot_2d(vor_2d)

#print(polygon_area(points_2d))
#print(polygon_area_2(points_2d))

#plt.figure(2)
print("---------------------------------------------")

dataa=voronoi_cell_area(vor_2d)
"""
"""
#print(dataa)

t2=time.time()
print(t2-t1)


plt.figure(3)


plt.hist(log(dataa), bins=100, normed=1, color = 'pink')#,range = (moyenne(-log(dataa))-3, moyenne(-log(dataa))+3))


print(moyenne(log(dataa)))


print(variance(log(dataa)))


print(ecartype(log(dataa)))


#plt.figure(4)

#plt.yscale('log')

n = 200
m, sig = moyenne(log(dataa)), ecartype(log(dataa))
X = npr.normal(m, sig, n)
x = np.arange(X.min(), X.max() + 0.01, 0.01)
plt.plot(x, st.norm.pdf(x, m, sig), 'b', linewidth=1, label='densité')

plt.savefig('Log_poisson.png')

"""

plt.figure(5)

plt.yscale('log')
plt.xscale('log')
   


mu, sigma = 4,1#log(m)-1/2*log(1+sig/m**2), (log(1+sig/m**2))**(1/2)        #3,1#exp(3.), exp(1.) # mean and standard deviation
s = np.random.lognormal(mu, sigma, 1000)
x = np.linspace(0, 500, 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
       / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()





"""


noisy_image = image + A*poisson(ones((len(image),len(image[0])))















x1 = [1, 2, 2, 3, 4, 4, 4, 4, 4, 5, 5]
x2 = [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 5, 5]
bins = [x + 0.5 for x in range(0, 6)]
pyplot.hist([x1, x2], bins = bins, label = ['x1', 'x2']) # bar est le defaut







f = open("random3d\pcl-ran000", "rb")
 
try:
    while True:
        bytes = f.read(1) # read the next /byte/
        if bytes == "":
            break;
        # Do stuff with byte
        # e.g.: print as hex
	 	  print("%02X " % ord(bytes[0]))
 
except IOError:
	# Your error handling here
	# Nothing for this example
	 pass
finally:
     f.close()



import struct
 
f = open("random3d\pcl-ran000", "rb")
 
try:
    while True:
        record = f.read(56)
        #print(unpack(record))
        if len(record) != 56:
            break;
        # Do stuff with record
 
except IOError:
        # Your error handling here
        # Nothing for this example
        pass
finally:
    f.close()






from struct import *
import struct


f = open("random3d\pcl-ran000", "rb")
 
try:
    #s = struct.Struct("<dffflfffffffl") # Binary data format
    while True:
	     record = f.read(56)[0:8]
	     if len(record) != 56:
	         break
 
    record=struct.unpack('d',record)
    print(record)
	 #print(s.unpack(record)) # Decode ("unpack")
 
except IOError:
        # Your error handling here
        # Nothing for this example
        pass
finally:
    f.close()




with open("random3d\pcl-ran000", "rb") as f:
    test=f.read()[0:8]
    test=struct.unpack('d',test)
    print(test)











with open("random3d\pcl-ran000", 'rb') as file:
    for line in file: # line est de type string contenant le caractère de saut de ligne qui sera toujours "\n"
        print(type(line), repr(line), line.strip())












import struct
# on enregistre un entier, un réel et 4 caractères
i = 10
x = 3.1415692
s = "ABCD"

# écriture
with open("info.txt", "wb") as fb:
    fb.write(struct.pack("i", i))
    fb.write(struct.pack("d", x))
    # il faut convertir les caractères en bytes
    octets = s.encode("ascii")
    fb.write(struct.pack("4s", octets))

# lecture
with open("info.txt", "rb") as fb:
    i = struct.unpack("i", fb.read(4))
    x = struct.unpack("d", fb.read(8))
    s = struct.unpack("4s", fb.read(4))

# affichage pour vérifier que les données ont été bien lues
print(i)
print(x)
print(s)



from struct import pack
print(len(pack('i', 0)))
print(len(pack('d', 0)))
print(len(pack('s', b'0')))









import pickle

dico = {'a': [1, 2.0, 3, "e"], 'b': ('string', 2), 'c': None}
lis  = [1, 2, 3]

with open ('data.bin', 'wb') as fb:
    pickle.dump(dico, fb)
    pickle.dump(lis, fb)



with open('data.bin', 'rb') as fb:
    dico = pickle.load(fb)
    lis  = pickle.load(fb)

print(dico)











from scipy.stats import gamma
import matplotlib.pyplot as plt





histogramme(data, 200,1,1,0.1,1000, fichier, couleur, 7)

plt.figure(7)
a = 1/ecartype(log10(data))
mean, var, skew, kurt = gamma.stats(a, moments='mvsk')

plt.xscale('log')
plt.yscale('log')
"""   """
x = np.linspace(gamma.ppf(0.001, a),gamma.ppf(0.999, a), 100)
plt.plot(x, gamma.pdf(x, a),'r',  label='gamma pdf')








def loi_de_test(s):
    
    plt.figure(660)
    
    #plt.yscale('log')
    #plt.xscale('log')
    x = np.arange(0, 1 + 0.01, 0.001)
    p=[]
    px=[]
    for i in range(len(x)):
        p.append(1/(  s**(2/s**2)*gamma(1/s**2)  )*(x[i])**(1/s**2-1)*exp(-x[i]/s**2))
        px.append(i)
    
    plt.scatter(px,p,s=2)


#1/(  s**(2/s**2)*gamma(1/s**2)  )*(x[i])**(1/s**2-1)*exp(-x[i]/s**2)



from scipy.special import gamma, factorial
gamma(0.5)**2  # gamma(0.5) = sqrt(pi)

x = np.linspace(-3.5, 5.5, 2251)
y=[]
for i in range(len(x)):
    y.append(gamma(x[i]))
#y = gamma(x)
import matplotlib.pyplot as plt
plt.plot(x, y, 'b', alpha=0.6, label='gamma(x)')
k = np.arange(1, 7)
plt.plot(k, factorial(k-1), 'k*', alpha=0.6,
         label='(x-1)!, x = 1, 2, ...')
plt.xlim(-3.5, 5.5)
plt.ylim(-10, 25)
plt.grid()
plt.xlabel('x')
plt.legend(loc='lower right')
plt.show()



def loi_de_test(s):
    
    plt.figure(660)
    
    plt.yscale('log')
    plt.xscale('log')
    x = np.arange(0, 10 + 0.01, 0.001)
    p=[]
    px=[]
    for i in range(len(x)):
        p.append(1/(  s**(2/s**2)*gamma(1/s**2)  )*(x[i])**(1/s**2-1)*exp(-x[i]/s**2))
        px.append(x[i])
    
    plt.plot(px,p)


#1/(  s**(2/s**2)*gamma(1/s**2)  )*(x[i])**(1/s**2-1)*exp(-x[i]/s**2)






def loi_de_data_2d(s):
    
    plt.figure(2)
    
    plt.yscale('log')
    plt.xscale('log')
    x = np.arange(0, 10 + 0.01, 0.001)
    p=[]
    px=[]
    for i in range(len(x)):
        p.append(343/15*sqrt(7/2*np.pi)*(x[i])**(5/2)*exp(-1*x[i]-1))
        px.append(x[i])
    
    plt.plot(px,p)




#343/15*sqrt(7/2*np.pi)*x[i]**(5/2)*exp(-7/2*x[i])







points_2d = np.random.random_sample((50000,3))
#the edges of a 3D Voronoi diagram will be the farthest from the obstacle coordinates
for i in range(len(points_2d)):
    points_2d[i][0]=points_2d[i][0]*2*np.pi
    points_2d[i][1]=points_2d[i][1]*2*np.pi
    points_2d[i][2]=points_2d[i][2]*2*np.pi
ecrit_point("points_2d.txt",points_2d)







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








Donner=[]



with open("random3d\pcl-ran000", "rb") as f:
    test=f.read(8)
    Donner.append( struct.unpack('d',test) )
    print(Donner)







Donner=[]


from struct import *
import struct


f = open("random3d\pcl-ran000", "rb")
 
try:
    while True:
        test=f.read(8)
        Donner.append( struct.unpack('d',test) )
         
    print(Donner) # Decode ("unpack")
 
except IOError:
        # Your error handling here
        # Nothing for this example
        pass
finally:
    f.close()


point_3d

for i in range(len(Donner)):
    if Donner[i]==4.41807387E-01:
        print(i)










#plt.figure(4)
#fig, ax = plt.subplots(1, 1)


"""
x = np.arange(-10, 10 + 0.01, 0.01)
s = ecartype(log(dataa))
sc = 1
a = 4
y = 5*x/sc
#x = np.linspace(gamma.ppf(0.01, a), gamma.ppf(0.99, a), 1000)
#rv = gamma(a, loc = 0, scale = variance(log(dataa))/abs(moyenne(log(dataa))))
plt.plot(x, gamma.pdf(y, a) / sc, 'k-', lw=2, label='frozen pdf')
"""


"""
plt.yscale('log')
plt.xscale('log')

x = np.arange(-10, 10 + 0.01, 0.01)
s = 1#ecartype(log(dataa))
sc = s**2
a = 1/s**2
y = (x/s**2)/sc
#x = np.linspace(gamma.ppf(0.01, a), gamma.ppf(0.99, a), 1000)
#rv = gamma(a, loc = 0, scale = variance(log(dataa))/abs(moyenne(log(dataa))))
plt.plot(x, gamma.pdf(y, a) / sc, 'k-', lw=2, label='frozen pdf')
"""



"""

from scipy.stats import gamma
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)


a = 1.99
mean, var, skew, kurt = gamma.stats(a, moments='mvsk')

x = np.linspace(gamma.ppf(0.01, a),
                gamma.ppf(0.99, a), 100)
ax.plot(x, gamma.pdf(x, a),
       'r-', lw=5, alpha=0.6, label='gamma pdf')


"""

"""
    x = np.arange(-10, 10 + 0.01, 0.01)
    s = ecartype(log(dataa))
    sc = s**2
    a = 1/s**2
    y = (x/s**2)/sc
    plt.plot(x, gamma.pdf(y, a) / sc, 'k-', lw=2, label='frozen pdf')
    
    
    #a=-moyenne(log10(data))/2
    #x = np.linspace(gamma.ppf(0.01, a), gamma.ppf(0.99, a), 1000)
    x = np.arange(log10(data).min(), log10(data).max() + 0.01, 0.01)
    #rv = gamma(a, loc = -5, scale = 1/variance(log10(data)))
    rv = gamma((abs(moyenne(log(data))))**2/variance(log(data)), loc = -11, scale = variance(log(dataa))/abs(moyenne(log(dataa))) )
    plt.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
"""


"""
        x = np.arange(0,20 + 0.01, 0.01)
        rv = gamma(9, loc = 0, scale = 0.5)
        plt.plot(x, rv.pdf(x))
        x = np.arange(0,20 + 0.01, 0.01)
        rv = gamma(5, loc = 0, scale = 1)
        plt.plot(x, rv.pdf(x))
        x = np.arange(0,20 + 0.01, 0.01)
        rv = gamma(3, loc = 0, scale = 2)
        plt.plot(x, rv.pdf(x))
        x = np.arange(0,20 + 0.01, 0.01)
        rv = gamma(2, loc = 0, scale = 2)
        plt.plot(x, rv.pdf(x))
        x = np.arange(0,20 + 0.01, 0.01)
        rv = gamma(1, loc = 0, scale = 2)
        plt.plot(x, rv.pdf(x))
        

"""

"""
a = 5
mean, var, skew, kurt = gamma.stats(a, moments='mvsk')
x = np.linspace(gamma.ppf(0.01, a), gamma.ppf(0.99, a), 100)
plt.plot(x, gamma.pdf(x, a), 'r-', label='gamma pdf')
"""

"""
import numpy as np
import mpl_scatter_density
import matplotlib.pyplot as plt

# Generate fake data

N = 10000000
x = np.random.normal(4, 2, N)
y = np.random.normal(3, 1, N)

# Make the plot - note that for the projection option to work, the
# mpl_scatter_density module has to be imported above.

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
ax.scatter_density(x, y)
ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)
fig.savefig('gaussian.png')

"""



"""

import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt 
x=sp.random.poisson(lam=1, size=100) 
#plt.plot(x,'o') 
a = 10. # shape  
s = np.random.power(a, n) 
count, bins, ignored = plt.hist(s, bins=150) 


x = np.linspace(0, 1, 100) 
y = a*x**(a-1.) 
normed_y = n*np.diff(bins)[0]*y 
plt.title("Poisson distribution")
plt.ylabel("y")
plt.xlabel("x")
plt.plot(x, normed_y) 



"""


"""

n = 200
m, sig = 9.2, 0.55
X = npr.normal(m, sig, n)
x = np.arange(X.min(), X.max() + 0.01, 0.01)
plt.plot(x, st.norm.pdf(x, m, sig), 'b', linewidth=1, label='densité')


#plt.figure(4)
a = 5+6
mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')

x = np.linspace(skewnorm.ppf(0.01, a), skewnorm.ppf(0.99, a), 1000)

plt.plot(x, skewnorm.pdf(x, a,1,1),'r-', lw=5, alpha=0.6, label='skewnorm pdf')

rv = skewnorm(a)
plt.plot(x, rv.pdf(x), 'k-', label='frozen pdf')













plt.figure(4)

plt.hist(dataa, bins=100, normed=1, log = True)
"""


"""
if float(vor_2d.vertices[0][0])>0:
    print(vor_2d.vertices[0][0])

"""



"""
for i in range(len(vor_2d.vertices)):
    test=vor_2d.vertices[i][0]
    if test>0:
        print(vor_2d.vertices[i][0])

print(vor_2d.vertices[10])
print(vor_2d.vertices[11])
print(vor_2d.vertices[2])
print(vor_2d.vertices[25])
print(vor_2d.vertices[29])
"""


"""
Coord_ind_points=[]
for i in range(len(vor_2d.regions)):
    if -1 not in vor_2d.regions[i] and vor_2d.regions[i]!=[]:
        for j in range(len(vor_2d.regions[i])):
            Coord_ind_points.append(vor_2d.vertices[vor_2d.regions[i][j]])
        print(polygon_area(Coord_ind_points))


print(vor_2d.vertices)
print(Coord_ind_points)



points = np.random.rand(4, 2)   # 30 random points in 2-D

points[0]=[1,1]
points[1]=[1,5]
points[2]=[2,5]
points[3]=[5,1]

print(polygon_area(points))


"""





#------------------------------------------------------------------------------







Donner=[]


from struct import *
import struct


f = open("random3d\pcl-ran000", "rb")

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

   

points_3d_bis=np.zeros((150000,3))
for i in range(150000):
    points_3d_bis[i]=np.copy(points_3d[i])

lol=points_3d

points_3d=points_3d_bis
"""
"""

#points_3d = np.random.random_sample((n,3))
#the edges of a 3D Voronoi diagram will be the farthest from the obstacle coordinates

t1=time.time()
vor = Voronoi(points_3d)
   
t2=time.time()  
   
print(t2-t1)

fig_drone_3d = plt.figure()
fig_drone_3d.set_size_inches(8,8)
ax = fig_drone_3d.add_subplot(111, projection = '3d')

for ridge_indices in vor.ridge_vertices:
    voronoi_ridge_coords = vor.vertices[ridge_indices]
    ax.plot(voronoi_ridge_coords[...,0], voronoi_ridge_coords[...,1], voronoi_ridge_coords[...,2], lw=2, c = 'green', alpha = 0.05)
    
vor_vertex_coords = vor.vertices

ax.scatter(points_3d[...,0], points_3d[...,1], points_3d[...,2], c= 'k',s=0.05, label='obstacles', edgecolor='none')
ax.scatter(vor_vertex_coords[...,0], vor_vertex_coords[...,1], vor_vertex_coords[...,2], c= 'orange', label='Voronoi vertices',edgecolors='white', marker = 'o', alpha = 0.9)

ax.legend()
ax.set_xlim3d(points_3d[...,0].min(), points_3d[...,0].max())
ax.set_ylim3d(points_3d[...,1].min(), points_3d[...,1].max())
ax.set_zlim3d(points_3d[...,2].min(), points_3d[...,2].max())

"""
"""