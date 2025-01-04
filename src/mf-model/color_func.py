import numpy as np
from scipy.special import comb, lambertw
from scipy.optimize import fsolve, minimize, least_squares
from predict_mf_model import *
from dynamical_mf_model import *
from plot_master_diagram import *

def determine_color(eps_11, ratio_01, ratio_00, q, K):
    eps_01 = eps_11 * ratio_01
    eps_00 = eps_11 * ratio_00
    # get type of the model
    p_order, p_disorder, p_bifr = -1, -1, -1
    order_coex, dis_coex, bifr_lnc, max_bifr_lnc = None, None, None, None
    # model prediction
    model = MFmodel(eps_11, eps_01, eps_00, q, K)
    sol = {'p_ord':None, 'p_dis':None, 'p_bifr':None, \
            'c_coex_ord':None, 'c_coex_dis':None, 'c_coex_ord_est':None, 'c_coex_dis_est':None, 'c_bifr':None}

    is_multi_roots = model.predict_if_bifr()
    print('model prediction: if in the bifurcation region = ', is_multi_roots)
    if is_multi_roots:
        coex_points = model.solve_for_coexistence()
        #coex_points = sorted(coex_points)
        #sol['p_ord'] = coex_points[-1]
        #sol['p_dis'] = coex_points[0]
        #sol['c_coex_ord'] = model.get_coex()
        #sol['c_coex_dis'] = model.get_coex()
        sol['c_coex_ord_est'] = np.exp(model.est_order_coex_lnc())
        sol['c_coex_dis_est'] = np.exp(model.est_disorder_coex_lnc())
    
    sol['p_bifr'], sol['c_bifr'], bifr_sol_resid = model.solve_for_bifr()

    ocolor = None
    if is_multi_roots:
        # by definition should have bifr. solution
        # solve for coexistence and compare
        p_ord, order_coex = coex_points[0]
        p_dis, dis_coex = coex_points[1]

        if order_coex < dis_coex < np.log(sol['c_bifr']): # DP
            ocolor = 'orange'
            status = 2
        elif order_coex  < np.log(sol['c_bifr']) < dis_coex: # assembly
            ocolor = 'green'
            status = 1
        elif dis_coex < order_coex < np.log(sol['c_bifr']): # no-assembly
            ocolor = 'red'
            status = 0
    
    else:
        #
        if np.all(bifr_sol_resid) < 1e-10:
            if_bifr_pt = True
            status = 3
            ocolor = 'black'
        else:
            status = 4
            ocolor = 'grey'

    return status

color_map ={2:'orange', 1:'green', 0:'red', 3:'black', 4:'grey'
        }
