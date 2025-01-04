import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.special import comb, lambertw
from scipy.optimize import fsolve, minimize, least_squares
from plot_predict_K import *
import subprocess

def solve_for_int(u_11, a, b, q, K=1):
    u_01 = u_11 + np.log(q-K) - a - b
    u_00 = np.log(q-K) - b + u_01

    return u_01, u_00

def solve_for_ab(u_11, u_01, u_00, q, K=1):
    a = u_11 + u_00 - 2. * u_01
    b = np.log(q - K) - u_00 + u_01

    return a, b


def solve_for_PT_boundary(u_11, u_01, q, K=1):
    '''
      Given u_11 and u_01, simultanously solve for p_bifr and u_00 such that the disordered coexistence concentration equals the bifurcation concentration
    '''
    def polyfunc(p, a1, a2):
        return q*p*(1-p) + (1-p*q)*(p*(1-p)*(a1-a2) + (1-2*p))

    def gen_nonlinear_eqns(x, u_11, u_01):
        eqs = np.zeros(2)
        p, u_00 = x[0], x[1]
        eqs[0] = u_00 - np.log(q - K) - np.log(1./(1. - q*p) * p * (1.-p) * (np.exp((u_11 - u_01)*p+u_01) - np.exp((u_01 - u_00)*p+u_00)))
        eqs[1] = polyfunc(p, u_11, u_01) * np.exp((u_11 - u_01)*p + u_01) - \
                                     polyfunc(p, u_01, u_00) * np.exp((u_01 - u_00)*p + u_00)
        return eqs

    p_, e_, resid = None, None, None
    for t in range(100):
        if t <= 50:
            p_guess = 0.1
        else:
            p_guess = 0.9
        u_00_guess = np.random.random() * u_11
        x0 = np.array([p_guess, u_00_guess])
        init_eq = gen_nonlinear_eqns(x0, u_11, u_01)
        if np.any(np.isnan(init_eq)):
            continue
        else:
            try:
                res = least_squares(gen_nonlinear_eqns, x0, args=(u_11, u_01))
                if 0 < res.x[0] < 1. and res.success:
                    if not e_:
                        p_, e_ = res.x
                        resid = res.fun
                    else:
                        if np.linalg.norm(res.fun) < np.linalg.norm(resid):
                            p_, e_ = res.x
                            resid = res.fun
            except:
                continue

    return p_, e_, resid


def get_pcoex_sols(u_11, u_01, u_00, q, K=1):
    a, b = solve_for_ab(u_11, u_01, u_00, q, K=K)
    a = float(a)
    b = float(b)
    output = subprocess.run(["./rLamb", str(a), str(b)], stdout = subprocess.PIPE,\
          universal_newlines = True).stdout

    lines = output.split('\n')
    p_sols = []
    for ln in lines:
        if 'W' in ln and '=' in ln:
            p_ = float(ln.split('=')[-1]) / a
            p_sols.append(p_)

    return sorted(p_sols)

def solve_for_disorder_boundary(u_11, u_01, q, K=1, init_guess=None):
    '''
    '''

    def func(u_00):
        # get p_sol from the generalized Lambert function
        p_sols = get_pcoex_sols(u_11, u_01, u_00, q, K)
        # check it is in the multi-solution region
        if len(p_sols) < 2:
            return float('inf')

        p = p_sols[-1]
        p_dis = p_sols[0]
        ln_ccoex_ord = np.log(p) + (u_11 - u_01) * p + u_01 * np.log(q)
        ln_ccoex_dis = np.log(p_dis) + (u_11 - u_01) * p_dis + u_01 * np.log(q)
        # ln_ccoex_dis = np.log(1.-p) + (u_01 - u_00) * p + u_00 + np.log(q/(q-K))
        diff = ln_ccoex_ord - ln_ccoex_dis
        #print(u_00, p, p_dis, diff)
        return diff**2

    if not init_guess:
        u_00_init = np.log(q - K) + u_11 # initial guess using the apprx with p goes to 0 or 1
    else:
        u_00_init = init_guess
    res = minimize(func, u_00_init, tol=1e-6)
    #print(res.x, res.success)

    return res




def predict(u_11, q, axs=None, K=1):
    if not axs:
        fig, axs = plt.subplots(1, 2, figsize=(4, 4))

    line_a = lambda x, b: x * (np.exp(x + b) + 1)
    bgrid = np.linspace(2*u_11,  -2*u_11, 100)
    line_upper = np.array([line_a(lambertw(-np.exp(-b) * np.exp(1.), -1) - 1., b) for b in bgrid])
    line_lower = np.array([line_a(lambertw(-np.exp(-b) * np.exp(1.), 0) - 1., b) for b in bgrid])

    boundary = {'upper':[], 'lower':[]} #TODO cleanup
    xdata, ylow_= [], []

    cusp_point_ab_plane = (-4., 2.) # (a*, b*)
    cusp_point = solve_for_int(u_11, *cusp_point_ab_plane, q)
    for i in range(len(bgrid)):
        u_01, u_00 = solve_for_int(u_11, np.real(line_lower[i]), np.real(bgrid[i]), q)
        a, b = solve_for_ab(u_11, u_01, u_00, q, K=K)
        if b >= 2:
            xdata.append(u_01)
            ylow_.append(u_00)

    xxdata,yupper_ = [], []
    for i in range(len(bgrid)):
        u_01, u_00 = solve_for_int(u_11, np.real(line_upper[i]), np.real(bgrid[i]), q)
        a, b = solve_for_ab(u_11, u_01, u_00, q, K=K)
        if b >= 2:
            xxdata.append(u_01)
            yupper_.append(u_00)

    xdata, xxdata, ylow_, yupper_ = np.array(xdata), np.array(xxdata), np.array(ylow_), np.array(yupper_)
    axs.plot(xdata/u_11, ylow_/u_11, color='green', linestyle='dashed', label='lowered bound, below 1 sol')
    axs.plot(xxdata/u_11, yupper_/u_11, color='green', label='upper bound, above 1 sol')


    xgrid = np.linspace(0.005, 0.75, 40)

    # disordered boundary
    ypred = np.ones(len(xgrid)) * (np.log(q - K) / u_11 + 1.)
    mask_region = np.argwhere(xgrid < np.interp(ypred, yupper_/u_11, xxdata/u_11))
    axs.plot(xgrid[mask_region], ypred[mask_region], color='black')

    # numerically solve for the disordered boundary
    x_, y_ = [], []
    u00_guess = None
    for x in xgrid:
        res = solve_for_disorder_boundary(u_11, x * u_11, q, K=K, init_guess=u00_guess)
        if res.success:
            if not u00_guess:
                u00_guess = res.x
            x_.append(x)
            y_.append(res.x / u_11)
    axs.plot(x_, y_, color='red')
    print('# u_01/u_11 u_00/u_11')
    for i in range(len(x_)):
        print(x_[i], y_[i])
    print()

    # DPT
    xsol, ysol = [], []
    for f in xgrid:
        u_01 = f * u_11
        p_b, u_00_sol, _ = solve_for_PT_boundary(u_11, u_01, q)
        if u_00_sol:
            # check if below the three-solution boundary
            xsol.append(f)
            ysol.append(u_00_sol/u_11)

    xsol, ysol = np.array(xsol), np.array(ysol)
    mask_region = np.argwhere(xsol < np.interp(ysol, yupper_/u_11, xxdata/u_11))
    axs.plot(xsol[mask_region], ysol[mask_region], '-.', color='blue')

    axs.set_xlim(0., 0.3)
    axs.set_ylim(0., 0.6)
    #axs.set_xlim(0., 1.)
    #axs.set_ylim(0., 1.)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bze', type=float, help="order-order interaction")
    parser.add_argument('--qval', type=int, default=64, help="q = number of states for particles on lattice [64]")
    parser.add_argument('--addK', action='store_true', help="plot Kdep points")
    parser.add_argument('--plotname', type=str, default=None, help="file name for the output .svg file")
    clargs = parser.parse_args()

    q = int(clargs.qval)
    u_11 = clargs.bze
    K = 1 # this is
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    axs.set_xlabel(r'$u_{01}/u_{11}$', fontsize=14)
    axs.set_ylabel(r'$u_{00}/u_{11}$', fontsize=14)


    predict(u_11, q, axs=axs)

    if clargs.addK:
        z = 2
        plot_K_parametrization(q, u_11, z, axs)


    if clargs.plotname:
        figname = clargs.plotname
        plt.savefig(figname)
    else:
        plt.show()

