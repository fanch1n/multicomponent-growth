import numpy as np
import math
from scipy.optimize import fsolve, minimize, least_squares, minimize_scalar
from predict_mf_model import *
from scipy.special import comb, lambertw
import sys
from color_func import * 
#def get_u_00(K, bond_eng, dof=2, q=64.):
#    top, Z = 0., 0.
#    mult = np.array([(q-K) / q, K / q]) # m = 0, m = 1
#    boltz = np.array([1, np.exp(-bond_eng)])
#    Z = np.sum(mult * boltz)
#    top = np.sum(bond_eng * np.array([0, 1]) * mult * boltz)
#
#    return top / Z


def get_u_00(K, bond, q):
    top = np.exp(-bond) * K / q 
    Z = np.exp(-bond) * K / q  + (q*1. - K) / q
    return top / Z 
 

def get_K_parametrization(K, q, u_11, z=2):
    be = u_11 / z
    f = (K - 1.) / q
    g = get_u_00(K, be, q) / z # Current parametrization
    #u_01 = u_11 * f
    #u_00 = u_11 * g
    return f, g

def get_boundary(u_11, q, K=1):
    line_a = lambda x, b: x * (np.exp(x + b) + 1)
    bgrid = np.linspace(u_11,  -u_11, 200)
    line_upper = np.array([line_a(lambertw(-np.exp(-b) * np.exp(1.), -1) - 1., b) for b in bgrid])
    line_lower = np.array([line_a(lambertw(-np.exp(-b) * np.exp(1.), 0) - 1., b) for b in bgrid])
    xdata, ylow_ = [], []
    for i in range(len(bgrid)):
        u_01, u_00 = solve_for_int(u_11, np.real(line_lower[i]), np.real(bgrid[i]), q, K=K)
        a, b = solve_for_ab(u_11, u_01, u_00, q, K=K)
        if b >= 2:
            xdata.append(u_01)
            ylow_.append(u_00)
    xxdata, yupper_ = [], []
    for i in range(len(bgrid)):
        u_01, u_00 = solve_for_int(u_11, np.real(line_upper[i]), np.real(bgrid[i]), q, K=K)
        a, b = solve_for_ab(u_11, u_01, u_00, q, K=K)
        if b >= 2:
            xxdata.append(u_01)
            yupper_.append(u_00)

    xdata, xxdata, ylow_, yupper_ = np.array(xdata), np.array(xxdata), np.array(ylow_), np.array(yupper_)
   
    return {'lower': (xdata/u_11, ylow_/u_11), 'upper':(xxdata/u_11, yupper_/u_11)}


def find_mindist_point(curve, pt):
    x, y = pt
    xdata, ydata = curve
    ans = None
    min_dist = float('inf')
    for x_ref, y_ref in zip(xdata, ydata):
        dis = (x - x_ref)**2 + (y - y_ref)**2
        if dis < min_dist:
            min_dist = dis
            ans = (x_ref, y_ref)

    return ans

def get_min_dist(curve1, curve2, pt):
    pt1 = find_mindist_point(curve1, pt)
    pt2 = find_mindist_point(curve2, pt)
    x, y = pt
    dist1 = (pt1[0] - x)**2 + (pt1[1] - y)**2
    dist2 = (pt2[0] - x)**2 + (pt2[1] - y)**2
    if dist1 < dist2:
        return dist1, pt1
    else:
        return dist2, pt2


def solve_capacity(N, u_11, rot_z=4, ax=None):
    q = rot_z * N
    line_a = lambda x, b: x * (np.exp(x + b) + 1)
   
    cusp_point_ab_plane = (-4., 2.) # (a*, b*)
    cusp_point = solve_for_int(u_11, *cusp_point_ab_plane, q, K=1)
    if ax:
        #ax.scatter(*np.array(cusp_point)/u_11, color='green')
        lines = get_boundary(u_11, q, K=1)
        ax.plot(lines['lower'][0], lines['lower'][1], color='green', linestyle='dashed', label='lowered bound, below 1 sol')
        ax.plot(lines['upper'][0], lines['upper'][1], color='green', label='upper bound, above 1 sol')

    #print('cusp point')
    def func(K):
        #print('input: ', q, K, u_11, np.log(q-K))
        #u_01, u_00 = get_K_parametrization(K, q, u_11)
        x0, y0 = get_K_parametrization(K, q, u_11)
        u_01, u_00 = x0 * u_11, y0 * u_11
        #x0, y0 = u_01/u_11, u_00/u_11

        p_a, p_b = solve_for_ab(u_11, u_01, u_00, q, K=K)
        b_ref = p_b
        a_ref_upper = line_a(lambertw(-np.exp(-b_ref) * np.exp(1.), -1) - 1., b_ref)
        a_ref_lower = line_a(lambertw(-np.exp(-b_ref) * np.exp(1.), 0) - 1., b_ref)


        x, y = solve_for_int(u_11, a_ref_upper, b_ref, q, K=K) # point with the same b val, on the top branch 
        x_, y_ = solve_for_int(u_11, a_ref_lower, b_ref, q, K=K) # point with the same b val, on the bottom branch 

        bdx = get_boundary(u_11, q, K=K)
        
        if_below = int(x0 < x/u_11)
        if_above = int(x0 > x_/u_11)
        
        diff, pt = get_min_dist(bdx['lower'], bdx['upper'], (x0, y0))

        #print(K, x0, y0, diff)
        #print('diff, pt: ', diff, pt)
        origin = (x0, y0)

        if (if_below and if_above):
            return diff, pt, origin 
        else:
            return float('inf'), (None, None), origin

    ans = None
    min_diff = None #float('inf')
    cp = None
    for K in range(1, q-1, 1):
        #print('K input = ', K)
        res = func(K)
        diff, curr, ref = res[0], res[1], res[2]
        if math.isinf(diff):
            break

        if not min_diff:
            min_diff = diff

        if diff < min_diff:
            min_diff = diff
            ans = K
            cp = curr
            x0, y0 = ref
    #TODO: determine the color of the cp point
    color='None'
    if cp:
        print('point = ', *cp)
        color = determine_color(u_11, cp[0], cp[1], q, K)

    if ax and ans:
        line = ['--', 'solid']
        #print('plt: ', x0, y0)
        ax.scatter(x0, y0, color='black')
        ax.scatter(cp[0], cp[1], color='blue')
        #print('cp: ', cp)
        ax.text(0.25, 0.5, 'N = %d, K = %d' % (N, ans))

        #ax.plot([x0, cp[0]], [y0, cp[1]], linestyle='solid')

        lines = get_boundary(u_11, q, K=ans)
        ax.plot(lines['lower'][0], lines['lower'][1], color='black', linestyle='dashed')
        ax.plot(lines['upper'][0], lines['upper'][1], color='black')


    return ans, color

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bze', type=float, help="order-order interaction")
    parser.add_argument('--Nmax', type=int, default=16, help="max number of components [16]")
    parser.add_argument('--plot', action='store_true', help="plot Kdep points")
    parser.add_argument('--output', type=str, default=None, help="file name for the output .dat file")
    clargs = parser.parse_args()
      
    rot_z = 4
    u_11 = clargs.bze
    dat = []
    dN = 5
    Ns = np.arange(2, clargs.Nmax + dN, dN)
    Nplot = np.arange(2, clargs.Nmax + dN, dN)


    if clargs.plot:
        fig, axs = plt.subplots(len(Ns), 1, figsize=(3, len(Ns)*3), sharex=True, sharey=True)
        
        axs[1].set_xlim(0, 1)
        axs[1].set_ylim(0, 1)
        axs[0].set_xlabel(r'$u_{01}/u_{11}$', fontsize=14)
        axs[0].set_ylabel(r'$u_{00}/u_{11}$', fontsize=14)

    cnt = 0
    for N in Ns:
        ax_ = None
        if clargs.plot: #and cnt % 5 == 0:
            ax_ = axs[int(cnt)]

        capacity, color_code = solve_capacity(N, u_11, rot_z=rot_z, ax=ax_)
        res = [N, capacity, color_code] #FIXME
        print(*res)
        dat.append(res)
        cnt += 1

    if clargs.output:
        print(dat)
        np.savetxt('%s_%.3f_scaling.dat' % (clargs.output, u_11), np.array(dat, dtype=np.float64))
        if clargs.plot:
            fig.tight_layout()
            plt.savefig('%s_%.3f_fig.svg' % (clargs.output, u_11))

    #if clargs.plot and not clargs.output:
    #    plt.show() 
