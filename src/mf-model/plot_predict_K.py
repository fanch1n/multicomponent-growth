import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
#from plot_model import *
from predict_mf_model import *

def get_factor(K, eps, dof=4, replacement=True, q=64):
    Z = 0
    top = 0
    if replacement:
        multiplicty = lambda K, q, m: comb(dof, m, exact=True) * pow(K, m) * pow(q-K, dof-m)
    else:
        multiplicty = lambda K, q, m: comb(K, m, exact=True) * comb(q-K, dof-m, exact=True)
    for m in range(dof+1):
        mult = multiplicty(K, q, m)
        Z += mult * np.exp(-eps * m)
        top += m * mult * np.exp(-eps * m)
    factors = top/Z
    return factors

def get_u_00(K, bond_eng, dof=2, q=64.):
    top, Z = 0., 0.
    mult = np.array([(q-K) / q, K / q]) # m = 0, m = 1
    boltz = np.array([1, np.exp(-bond_eng)])
    Z = sum(mult * boltz)
    top = sum(bond_eng * np.array([0, 1]) * mult * boltz)

    return top / Z


def plot_K_parametrization(q, u_11, z, axs):
    be = u_11 / z
    for K in range(1, 11, 1):
        f = (K-1)/q
        # g = get_factor(K, u_11/z, dof=z)/z
        g = get_u_00(K, be) / u_11
        u_01 = u_11 * f
        u_00 = u_11 * g
        print('%d %.3f %.3f ' %(K, f, g))
        axs.scatter(f, g, marker='x', s=K*10, c='black')
    return

if __name__ == '__main__':
    q = 64
    z = 2
    u_11 = -8.

    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    predict(u_11, q, axs=axs)
    plot_K_parametrization(q, u_11, z, axs)
    axs.set_xlabel(r'$u_{01}/u_{11}$', fontsize=14)
    axs.set_ylabel(r'$u_{00}/u_{11}$', fontsize=14)
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)
    fig.tight_layout()
    plt.show()
    #plt.savefig('predict_eps11_%.1f.svg' %u_11)

