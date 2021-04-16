from array import array
import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

width = 600
height = 600
depth = 1100

#data = array('B')

#with open('attenuation.bin', 'rb') as f:

f = open("attenuation.bin", "rb")
data = np.fromfile(f, count=-1, dtype=np.float32)
f.close()

data3D = data.reshape((depth,height,width)).transpose()

#print(data3D[129,94,565])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data3D[:,0]
y = data3D[:,1]
z = data3D[:,2]

ax.scatter(x,y,z)
plt.show()

