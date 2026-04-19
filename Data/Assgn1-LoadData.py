import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

data = np.load("../Data/isotropic1024_slice.npz" )
u=data["u"]
v=data["v"]
w=data["w"]

urms = np.mean(u**2.0)**0.5
L = 1.346
nu = 0.000185
nx = 1024
dx = 2.0*np.pi/nx

print(u.shape)

skip = 2
plt.pcolor( u[::skip,::skip], vmin=-2, vmax=2, cmap='RdBu')
plt.title(r'$u$')

plt.show()

