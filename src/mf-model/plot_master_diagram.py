import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.special import comb, lambertw
from scipy.optimize import fsolve, minimize, least_squares
from predict_mf_model import *
from plot_predict_K import *

def dotN(p, c, eps_11, eps_01, eps_00):
    return c - p * np.exp((eps_11 - eps_01)*p + eps_01) - (1. - p) * np.exp((eps_01-eps_00)*p + eps_00)

def coex_func(p, a, b):
    r = np.exp(-b)
    return r * p + p * np.exp(a * p) - np.exp(-b)

def solve_for_int(eps_11, a, b, q, K=1):
    eps_01 = eps_11 + np.log(q-K) - a - b
    eps_00 = np.log(q-K) - b + eps_01

    return eps_01, eps_00

def solve_for_ab(eps_11, eps_01, eps_00, q, K=1):
    a = eps_11 + eps_00 - 2. * eps_01
    b = np.log(q - K) - eps_00 + eps_01

    return a, b

def order_coex_lnc(p, eps_11, eps_01, q, K=1):
    return np.log(p) + eps_11 * p + eps_01 * (1.-p) + np.log(q)

def disorder_coex_lnc(p, eps_01, eps_00, q, K=1):
    return np.log(1-p) + eps_00 * (1.-p) + eps_01 * p + np.log(q/(q-K))

def get_lnc_bifr(p, eps_11, eps_01, eps_00, q, K=1):
    c = q * p * (1.-p)/(1. - q*p) * (np.exp((eps_11-eps_01) * p + eps_01) - np.exp((eps_01-eps_00) * p + eps_00))
    return np.log(c)

def gen_nonlinear_eqns(x, eps_11, eps_01, q, K=1):
    '''
    '''
    eqs = np.zeros(2)
    p, eps_00 = x[0], x[1]
    eqs[0] = eps_00 - np.log(q - K) - np.log(1./(1. - q*p) * p * (1.-p) * (np.exp((eps_11 - eps_01)*p+eps_01) - np.exp((eps_01 - eps_00)*p+eps_00)))
    eqs[1] = polyfunc(p, eps_11, eps_01) * np.exp((eps_11 - eps_01)*p + eps_01) - \
                                 polyfunc(p, eps_01, eps_00) * np.exp((eps_01 - eps_00)*p + eps_00)
    return eqs

def nonlinear_solve_for_pbifr(eps_11, eps_01, eps_00, q):
    def gfunc(p, c): # function defining fixed points: y=0
        A = np.exp((eps_11 - eps_01) * p + eps_01)
        B = np.exp((eps_01 - eps_00) * p + eps_00)

        return c/q - p * (A - B + c) + p**2 * (A - B)


    def dg_dp(p, c): # partial derivative of g with respect to u
        A = np.exp((eps_11 - eps_01) * p + eps_01)
        B = np.exp((eps_01 - eps_00) * p + eps_00)
        factor3 = (eps_11 - eps_01) * A - (eps_01 - eps_00) * B

        return A * (2*p - 1) + B * (1 - 2*p) - c - p * factor3 + p**2 * factor3

    def gen_nonlinear_eqns(x):
        '''
            Given the dynamical system dot p = g(p, c), simultaneoulsy solve for dot p = 0 and dg/dp = 0
        '''
        eqs = np.zeros(2)
        p, c = x[0], x[1]
        eqs[0] = gfunc(p, c)
        eqs[1] = dg_dp(p, c)
        return eqs

    p_ans, c_ans, resid = None, None, None
    for t in range(100):
        p_guess = np.random.random()
        mu_guess = np.random.random() * (eps_11 - (-1.)) + eps_11
        x0 = np.array([p_guess, np.exp(mu_guess)])
        res = least_squares(gen_nonlinear_eqns, x0, ftol=1e-11, xtol=1e-11, gtol=1e-12)
        if 0 < res.x[0] < 1.:
            if not c_ans:
                p_ans, c_ans = res.x
                resid = res.fun
            else:
                if np.linalg.norm(res.fun) < np.linalg.norm(resid):
                    p_ans, c_ans = res.x
                    resid = res.fun

    return p_ans, c_ans, resid

def solve_for_PT_boundary(eps_11, eps_01, q, K=1):
    '''
      Given eps_11 and eps_01, simultanously solve for p_bifr and eps_00 such that the disordered coexistence concentration equals the bifurcation concentration
    '''
    def polyfunc(p, a1, a2):
        return q*p*(1-p) + (1-p*q)*(p*(1-p)*(a1-a2) + (1-2*p))

    def gen_nonlinear_eqns(x, eps_11, eps_01):
        eqs = np.zeros(2)
        p, eps_00 = x[0], x[1]
        eqs[0] = eps_00 - np.log(q - K) - np.log(1./(1. - q*p) * p * (1.-p) * (np.exp((eps_11 - eps_01)*p+eps_01) - np.exp((eps_01 - eps_00)*p+eps_00)))
        eqs[1] = polyfunc(p, eps_11, eps_01) * np.exp((eps_11 - eps_01)*p + eps_01) - \
                                     polyfunc(p, eps_01, eps_00) * np.exp((eps_01 - eps_00)*p + eps_00)
        return eqs

    p_, e_, resid = None, None, None
    for t in range(100):
        if t <= 50:
            p_guess = 0.1
        else:
            p_guess = 0.9
        eps_00_guess = np.random.random() * eps_11
        x0 = np.array([p_guess, eps_00_guess])
        init_eq = gen_nonlinear_eqns(x0, eps_11, eps_01)
        if np.any(np.isnan(init_eq)):
            continue
        else:
            try:
                res = least_squares(gen_nonlinear_eqns, x0, args=(eps_11, eps_01))
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



def solve_for_pdis_given_c(eps_11, eps_01, eps_00, c, q=64):
    def eqn(p):
        A = np.exp((eps_11 - eps_01) * p + eps_01)
        B = np.exp((eps_01 - eps_00) * p + eps_00)

        return (p * c + p * (A - B) + p**2 * (B - A) - c/q)**2

    p_, resid = None, None
    for t in range(100):
        p0 = np.random.random() * 0.01
        res = minimize(eqn, p0, bounds=[(0.,1.,)], tol=1e-10)
        if res.success:
            if not p_:
                 p_, = res.x
                 resid = eqn(p_)
            else:
                if eqn(p_) < resid:
                    p_ = res.x
                    resid = eqn(p_)

    return p_, resid


#def predict(eps_11, q, axs=None, K=1):
#    if not axs:
#        fig, axs = plt.subplots(1, 2, figsize=(4, 4))
#
#    line_a = lambda x, b: x * (np.exp(x + b) + 1)
#    bgrid = np.linspace(2*eps_11,  -2*eps_11, 100) # TODO check the q-dependency here
#    line_upper = np.array([line_a(lambertw(-np.exp(-b) * np.exp(1.), -1) - 1., b) for b in bgrid])
#    line_lower = np.array([line_a(lambertw(-np.exp(-b) * np.exp(1.), 0) - 1., b) for b in bgrid])
#
#    boundary = {'upper':[], 'lower':[]} #TODO cleanup
#    xdata, ylow_= [], []
#
#    cusp_point_ab_plane = (-4., 2.) # (a*, b*)
#    cusp_point = solve_for_int(eps_11, *cusp_point_ab_plane, q)
#    for i in range(len(bgrid)):
#        eps_01, eps_00 = solve_for_int(eps_11, np.real(line_lower[i]), np.real(bgrid[i]), q)
#        a, b = solve_for_ab(eps_11, eps_01, eps_00, q, K=K)
#        if b >= 2:
#            xdata.append(eps_01)
#            ylow_.append(eps_00)
#
#    xxdata,yupper_ = [], []
#    for i in range(len(bgrid)):
#        eps_01, eps_00 = solve_for_int(eps_11, np.real(line_upper[i]), np.real(bgrid[i]), q)
#        a, b = solve_for_ab(eps_11, eps_01, eps_00, q, K=K)
#        if b >= 2:
#            xxdata.append(eps_01)
#            yupper_.append(eps_00)
#
#    xdata, xxdata, ylow_, yupper_ = np.array(xdata), np.array(xxdata), np.array(ylow_), np.array(yupper_)
#    axs.plot(xdata/eps_11, ylow_/eps_11, color='green', linestyle='dashed', label='lowered bound, below 1 sol')
#    axs.plot(xxdata/eps_11, yupper_/eps_11, color='green', label='upper bound, above 1 sol')
#
#
#    xgrid = np.linspace(0.01, 1., 20)
#
#    # disordered boundary
#    ypred = np.ones(len(xgrid)) * (np.log(q - K) / eps_11 + 1.)
#    mask_region = np.argwhere(xgrid < np.interp(ypred, yupper_/eps_11, xxdata/eps_11))
#    axs.plot(xgrid[mask_region], ypred[mask_region], color='black')
#
#    # DPT
#    xsol, ysol = [], []
#    for f in xgrid:
#        eps_01 = f * eps_11
#        p_b, eps_00_sol, _ = solve_for_PT_boundary(eps_11, eps_01, q)
#        if eps_00_sol:
#            # check if below the three-solution boundary
#            xsol.append(f)
#            ysol.append(eps_00_sol/eps_11)
#
#    xsol, ysol = np.array(xsol), np.array(ysol)
#    mask_region = np.argwhere(xsol < np.interp(ysol, yupper_/eps_11, xxdata/eps_11))
#    axs.plot(xsol[mask_region], ysol[mask_region], '-.', color='blue')
#
#    #axs.plot(xsol, ysol, '-.', color='blue')
#    axs.set_xlim(0., 1.)
#    axs.set_ylim(0., 1.)
#    return
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bze', type=float, help="order-order interaction")
    parser.add_argument('--qval', type=int, default=64, help="q = number of states for particles on lattice [64]")
    parser.add_argument('--addK', action='store_true', help="plot Kdep points")
    parser.add_argument('--plotname', type=str, default=None, help="file name for the output .svg file")
    clargs = parser.parse_args()

    q = int(clargs.qval)
    eps_11 = clargs.bze
    K = 1 # this is a minimal model

    xlim = 1.
    ylim = 1.
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    m, n = 20, 20
    # xgrid = np.linspace(0.005, 0.3, m) #TODO add arg input specifying the range
    # ygrid = np.linspace(0.005, 0.6, n)
    xgrid = np.linspace(0.005, 1., m) #TODO add arg input specifying the range
    ygrid = np.linspace(0.005, 1., n)
    xv, yv = np.meshgrid(xgrid, ygrid)
    data = np.zeros(xv.shape)

    for i in range(m):
        for j in range(n):
            ratio_01, ratio_00 = xv[i, j], yv[i, j]
            eps_01 = eps_11 * ratio_01
            eps_00 = eps_11 * ratio_00
            a, b = solve_for_ab(eps_11, eps_01, eps_00, q)
            line_a = lambda x, b: x * (np.exp(x + b) + 1)
            upper_bound = lambda b: line_a(lambertw(-np.exp(-b) * np.exp(1), -1) - 1., b)
            lower_bound = lambda b: line_a(lambertw(-np.exp(-b) * np.exp(1), 0) - 1., b)
            a_upper = upper_bound(b)
            a_lower = lower_bound(b)
            # get type of the model
            p_order, p_disorder, p_bifr = -1, -1, -1
            order_coex, dis_coex, bifr_lnc, max_bifr_lnc = None, None, None, None

            # get p_coex for order and disorder coexistence
            ans = set()
            for t in range(100):
                g0 = np.random.rand()
                res = fsolve(coex_func, g0, args=(a, b), xtol=1e-12)
                p = res[0]
                if  abs(coex_func(p, a, b)) < 1e-9:
                    ans.add(np.round(p, 6))

            nsol = len(ans)
            if nsol >= 3:
                ans = sorted(ans)
                p_disorder = ans[0]
                dis_coex = disorder_coex_lnc(p_disorder, eps_01, eps_00, q)
                p_order = ans[-1]
                order_coex = order_coex_lnc(p_order, eps_11, eps_01, q)
                # Ndot_discoex = dotN(p_disorder, np.exp(dis_coex), eps_11, eps_01, eps_00)

            else:
                for sol in ans:
                    if sol < 0.9:
                        p_disorder = sol
                        dis_coex = disorder_coex_lnc(sol, eps_01, eps_00, q)
                    else:
                        p_order = sol
                        order_coex = order_coex_lnc(sol, eps_11, eps_01, q)

            if order_coex and dis_coex:
                # get p_bifr
                p_bifr, c_, residual = nonlinear_solve_for_pbifr(eps_11, eps_01, eps_00, q)
                if c_ < 0 or np.isnan(c_):
                    print(c_, eps_11, eps_01, eps_00, p_bifr)

                bifr_lnc = np.log(c_)
                if not np.isinf(bifr_lnc):
                    if not max_bifr_lnc:
                        max_bifr_lnc = bifr_lnc
                        p_bifr = p

            status = -10
            ocolor = 'blue'
            if order_coex and dis_coex and max_bifr_lnc:
                if order_coex < dis_coex < max_bifr_lnc: # DP
                    ocolor = 'orange'
                    status = 2
                elif order_coex  < max_bifr_lnc < dis_coex: # assembly
                    ocolor = 'green'
                    status = 1
                elif dis_coex < order_coex < max_bifr_lnc: # no-assembly
                    ocolor = 'red'
                    status = 0
                elif max_bifr_lnc < order_coex < dis_coex:
                    ocolor = 'cyan'
                    status = 4
                elif dis_coex < max_bifr_lnc < order_coex:
                    ocolor = 'magenta'
                    status = 5
                else:
                    ocolor = 'yellow'
                    status = 6
            else:
                if p_order == -1 and p_disorder == -1:
                    status = -4
                    ocolor = 'pink'
                else:
                    status = -1
                    ocolor = 'grey'

                    p_bifr, c_, residual = nonlinear_solve_for_pbifr(eps_11, eps_01, eps_00, q)
                    if c_ > 0 and (0 < p_bifr < 1) and np.all(residual < 1e-8):
                        print(eps_01/eps_11, eps_00/eps_11, p_bifr, c_, residual)
                        status = -2
                        ocolor = 'black'


            if ocolor == 'blue':
                print('blue: ', p_order, p_disorder, p_bifr, eps_01, eps_00, order_coex, dis_coex, max_bifr_lnc)
            data[i, j] = status
            axs.scatter(ratio_01, ratio_00, marker='.', color=ocolor)

            #print('%.6g %.6g %.6g %.6g %d %.3g %.3g %.3g' %(a, b, eps_01, eps_00, status, order_coex, dis_coex, max_bifr_lnc))
            #print(a, b, eps_01, eps_00, status, order_coex, dis_coex, max_bifr_lnc)

    # predict solution boundaries
    predict(eps_11, q, axs=axs)

    if clargs.addK:
        z = 2
        plot_K_parametrization(q, eps_11, z, axs)

    axs.set_xlabel(r'$u_{01}/u_{11}$', fontsize=14)
    axs.set_ylabel(r'$u_{00}/u_{11}$', fontsize=14)
    axs.set_xlim(0., xlim)
    axs.set_ylim(0., ylim)
    fig.tight_layout()

    if clargs.plotname:
        figname = clargs.plotname
    else:
        figname = 'scan_eps11_%.1f_q_%d.svg' %(eps_11, q)

    plt.savefig(figname)
    #np.save('status', data)
