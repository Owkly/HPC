import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import numpy as np

if len(sys.argv) != 5:
    print("USAGE: python3 {} [filename.txt] [n] [m] [o]")
    sys.exit(1)

_, filename, n, m, o = sys.argv
n = int(n)
m = int(m)
o = int(o)

L = 15
l = 12
E = 0.8
dx = L / n
dy = E / m
dz = l / o

print("Loading data from {}".format(filename))
A = np.loadtxt(filename) 
T = A.reshape([o, m, n]) 

levels = MaxNLocator(nbins=512, integer=True).tick_values(20, 90)
y, x = np.mgrid[slice(0, l, dy), slice(0, L, dx)]

xz_bot = T[:,  0, :]
xz_top = T[:, -1, :]

cmap = plt.get_cmap('magma')

fig, (ax0, ax1) = plt.subplots(ncols=2)
fig.set_size_inches(25, 6)

ax0.set_title('Top face')
ax0.set_ylabel('z (cm)')
ax0.set_xlabel('x (cm)')
ax0.set_aspect('equal')

ax1.set_title('Bottom face')
ax1.set_aspect('equal')
ax1.set_xlabel('x (cm)')

cf = ax0.contourf(x + dx/2, y + dy/2, xz_top, levels=levels, cmap=cmap)
ax1.contourf(x + dx/2, y + dy/2, xz_bot, levels=levels, cmap=cmap)
fig.colorbar(cf, ax=fig.get_axes())

cpu = Rectangle( ((L-7.54) / 2, ((l-5.85) / 2)), 7.54, 5.85, fill=False)
ax0.add_patch(cpu)

cpu = Rectangle( ((L-7.54) / 2, ((l-5.85) / 2)), 7.54, 5.85, fill=False)
ax1.add_patch(cpu)

plt.show()
#plt.savefig("steady_state.png")
