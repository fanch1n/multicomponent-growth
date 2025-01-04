import numpy as np
import os
import matplotlib.pyplot as plt
import sys

list_eps = sys.argv[1]
prefix = sys.argv[2]
eps = [float(_) for _ in list_eps.split(' ')]
#eps = [-8., -10, -12., -20.]
pat=['black', 'blue', 'green', 'orange', 'magenta', 'cyan', 'red', 'violet']

color_map ={2:'orange', 1:'green', 0:'red', 3:'black', 4:'grey'}

for i, e in enumerate(eps):
    dat = np.loadtxt('%s_%.3f_scaling.dat' % (prefix, e))
    list_code = []
    for status in dat[:, -1]:
        if np.isnan(status):
            list_code.append('cyan')
        else:
            list_code.append(color_map[int(status)])

#    plt.scatter(dat[:, 0], dat[:, 1],  color=list_code)#, label=r'$u_{11} = %.1f$'% e)
    plt.scatter(dat[:, 0], dat[:, 1],  color=pat[i], label=r'$u_{11} = %.1f$'% e)



def murugan_capacity(N, M, z=4):

    return N * pow(N, (z-2.)/2.) / M

Ns = np.arange(2, 50, 2)
plt.plot(Ns, murugan_capacity(Ns, 16), '--', color='grey')

plt.xlabel(r'$N$', fontsize=14)
plt.ylabel(r'capacity $K_*$', fontsize=14)
plt.legend()
plt.show()

