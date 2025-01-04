import math
#from test_bifr import *
import argparse
from dynamical_mf_model import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        arr = np.interp(value, x, y, left=-np.inf, right=np.inf)
        ma = np.ma.masked_array(arr)
        return ma
    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        arr = np.interp(value, x, y, left=-np.inf, right=np.inf)
        return arr

def color_mapping(lst):
    minima = min(lst)
    maxima = max(lst)
    norm = MidpointNormalize(vmin=minima, vcenter=(minima + maxima)/2., vmax=maxima)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
    return mapper


def gen_minimal_model(eps_11, eps_01, eps_00, c, q=64, K=1, dof=4):
    U = np.zeros((K + 1, K + 1))
    U[0, 0] = eps_00
    U[1, 1] = eps_11
    U[0, 1] = U[1, 0] = eps_01

    h = np.zeros(K + 1)
    h[0] = c * (q - K) / q
    h[1:] = c / q
    def Gamma(p):
        return h - p * np.exp(np.dot(U, p))

    def dotN(p):
        return Gamma(p).sum()

    def dotp(p):
        G = Gamma(p)
        return G - p * G.sum()

    def ddotpdp(p, full=False):
        expi = np.exp(np.dot(U, p))
        G = h - p * expi
        dGij = -(np.eye(K + 1) * expi).T - (U.T * p * expi).T
        ddotNj = dGij.sum(axis=0)
        ddotpij = dGij - np.outer(p, ddotNj) - np.eye(K + 1) * G.sum()
        if full: return ddotpij
        else: return ddotpij[1:,1:] - ddotpij[1:,0]

    def ddotpdp_finitediff(p, d=1.e-6, full=False): # For verification of Jacobian calculation above.
        ddotpij = np.zeros((K+1, K+1))
        for j in range(K+1):
            dp_j = np.zeros(K+1)
            dp_j[j] = d / 2.
            ddotpij[:,j] = (dotp(sp_j + dp_j) - dotp(sp_j - dp_j)) / d
        if full: return ddotpij
        else: return ddotpij[1:,1:] - ddotpij[1:,0]

    return dotp, dotN, ddotpdp

def search_sp(dotp, K, ntrials=10000, tol=1.e-6, tolcluster=1.e-2, prev_solution=[]):
    sp = []
    # initial guess based on solutions (stable branch) found at previous step
    for init_p in prev_solution:
        sp_candidate = fsolve(dotp, init_p)
        if np.linalg.norm(dotp(sp_candidate)) > tol or \
           math.fabs(sp_candidate.sum() - 1.) > tol or \
           np.any(sp_candidate < -tol):
            continue
        if not any(np.linalg.norm(sp_candidate - sp_j) < tolcluster for sp_j in sp):
            sp.append(sp_candidate)

    # initial guess: uniform random
    for trial in range(ntrials):
        #p = np.array([random.random() for i in range(K+1)])
        p = np.random.dirichlet(np.ones(K+1))
        p /= p.sum()
        sp_candidate = fsolve(dotp, p)
        if np.linalg.norm(dotp(sp_candidate)) > tol or \
           math.fabs(sp_candidate.sum() - 1.) > tol or \
           np.any(sp_candidate < -tol):
            continue

        if not any(np.linalg.norm(sp_candidate - sp_j) < tolcluster for sp_j in sp):
            sp.append(sp_candidate)

    # initial guess: dirichlet
    for trial in range(ntrials):
        p = np.random.dirichlet(np.ones(K+1))
        p /= p.sum()
        sp_candidate = fsolve(dotp, p)
        if np.linalg.norm(dotp(sp_candidate)) > tol or \
           math.fabs(sp_candidate.sum() - 1.) > tol or \
           np.any(sp_candidate < -tol):
            continue

        if not any(np.linalg.norm(sp_candidate - sp_j) < tolcluster for sp_j in sp):
            sp.append(sp_candidate)

    return sp

def get_solutions(datapath):
    '''
      locate points based on brute-force solution
    '''
    data = np.genfromtxt(datapath, dtype=np.complex_)
    filter_data = np.array(data[np.where(data[:, 4] > 0)], dtype=np.complex_)
    sol_data  = np.array(filter_data)
    sols_ordered = {}
    sols_disordered = {}
    sols = {}
    for i in range(len(sol_data[:, 0])):
        mu = float('%.4g' %np.real(sol_data[i, 0]))
        if np.real(sol_data[i, 2]) > 0.05:
            if mu not in sols_ordered:
                sols_ordered[mu] = []
            sols_ordered[mu].append([np.real(sol_data[i, 2]), np.real(sol_data[i, 5])])
        else:
            if mu not in sols_disordered:
                sols_disordered[mu] = []
            sols_disordered[mu].append([np.real(sol_data[i, 2]), np.real(sol_data[i, 5])])

    coex_point = min(sols_ordered.keys()) #sols_ordered[min(sols_ordered.keys())][0][0]
    bfr_left = max(sols_ordered.keys()) #sols_ordered[max(sols_ordered.keys())][0][0]

    diff = float('inf')
    for key in sorted(sols_disordered.keys()):
        if key < coex_point:
            continue
        else:
            bfr_right = key
            break

    # first disordered solution that is closest to the coexistence point

    return sols_ordered, sols_disordered, [np.log((coex_point)), sols_ordered[coex_point][0]], \
          [np.log((bfr_left)), sols_ordered[bfr_left][0]], \
          [np.log((bfr_right)), sols_disordered[bfr_right][0]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bze', type=float, help="order-order interaction")
    parser.add_argument('ratio_01', type=float, help="u_01 = ration_01 * bze")
    parser.add_argument('ratio_00', type=float, help="u_00 = ration_00 * bze")
    parser.add_argument('--qval', type=int, default=64, help="q = number of states for particles on lattice [64]")
    parser.add_argument('--output', type=str, default='test', help="file name for the output .dat file")
    clargs = parser.parse_args()

    q = int(clargs.qval)
    eps_11 = clargs.bze
    prefix = clargs.output
    eps_01, eps_00 = eps_11 * clargs.ratio_01,  eps_11 * clargs.ratio_00
    print(eps_11, eps_01, eps_00, clargs.ratio_01, clargs.ratio_00)
    K = 1 # this is a minimal model

    model = MFmodel(eps_11, eps_01, eps_00, q, K)
    is_multi_roots = model.predict_if_bifr()
    print('model prediction: if in the bifurcation region = ', is_multi_roots)
    model_summary = {'p_ord':None, 'p_dis':None, 'p_bifr':None, \
            'c_coex_ord':None, 'c_coex_dis':None, 'c_coex_ord_est':None, 'c_coex_dis_est':None, 'c_bifr':None}

    coex_pts = model.get_pcoex_sols()
    if len(coex_pts) < 0:
        print('NO coexistence points found')
        exit(1)
    elif len(coex_pts) == 1:
        model_summary['p_dis'] = coex_pts[0]
    elif len(coex_pts) == 3:
        model_summary['p_dis'] = coex_pts[0]
        model_summary['p_ord'] = coex_pts[-1]


    if is_multi_roots:
        model_summary['c_coex_ord_est'] = np.exp(model.est_order_coex_lnc())
        model_summary['c_coex_dis_est'] = np.exp(model.est_disorder_coex_lnc())
        model_summary['p_bifr'], model_summary['c_bifr'], _ = model.solve_for_bifr()
        Ndot_bifr = model.get_Ndot(model_summary['p_bifr'], model_summary['c_bifr'])
        print(model_summary)



    def solve_for_ptb_bifr(eps_11, eps_01, eps_00, q, K, k10, k01, init_guess=None):
        def gen_system():
            A = lambda p: np.exp((eps_11 - eps_01) * p + eps_01)
            B = lambda p: np.exp((eps_01 - eps_00) * p + eps_00)
            def gfunc(p, c): # master equation defining fixed points: dotp = 0
                # c is the concentration
                term = - p * (A(p) * k10 + B(p) * k01) + k01 * B(p)
                return c / q - p * (A(p) - B(p) + c) + p**2 * (A(p) - B(p)) + term
            def dgdp(p, c): # partial derivative of g with respect to p
                factor = (eps_11 - eps_01) * A(p) - (eps_01 - eps_00) * B(p)
                newterm = - A(p) * k10 + B(p) * k01 + k01 * (eps_01 - eps_00) * B(p) \
                        - p * ((eps_11 - eps_00) * A(p) * k10 + (eps_01 - eps_00) * B(p) * k01)
                return A(p) * (2*p - 1) + B(p) * (1 - 2*p) - c - p * factor + p**2 * factor + newterm
            return gfunc, dgdp

        gfunc, dgdp = gen_system()
        def gen_nonlinear_eqns(x):
            eqs = np.zeros(2)
            p, c = x[0], x[1]
            eqs[0] = gfunc(p, c)
            eqs[1] = dgdp(p, c)
            return eqs

        p_ans, c_ans, resid = None, None, None
        for t in range(101):
            try:
                if t == 0 and init_guess is not None:
                    x0 = np.array(init_guess)
                else:
                    p_guess = np.random.random()
                    mu_guess = np.random.random() * (eps_11 - (-1.)) + eps_11
                    x0 = np.array([p_guess, np.exp(mu_guess)])
                res = least_squares(gen_nonlinear_eqns, x0, ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=0)
                if 0 < res.x[0] < 1.:
                    if not c_ans:
                        p_ans, c_ans = res.x
                        resid = res.fun
                    else:
                        if np.linalg.norm(res.fun) < np.linalg.norm(resid):
                            p_ans, c_ans = res.x
                            resid = res.fun
            except:
                continue
        return p_ans, c_ans, resid

    list_k01 = np.logspace(-6, -1, num=20)
    pbifr_ptb = []
    cbifr_ptb = []

    guess = None
    if model_summary['p_bifr'] is not None:
        guess = [model_summary['p_bifr'], model_summary['c_bifr']]

    for k01 in list_k01:
        k10 = k01 * (model.q - model.K)

        test = solve_for_ptb_bifr(eps_11, eps_01, eps_00, q, K, k10, k01, init_guess=guess)
        print(k01, k10, *test)
        pbifr_ptb.append(test[0])
        cbifr_ptb.append(test[1])

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].plot(pbifr_ptb, cbifr_ptb, '-o')

    axs[0].scatter(model_summary['p_bifr'], model_summary['c_bifr'], color='green', s=150, marker='*')
    axs[0].set_ylabel(r'$c_\mathrm{bifr}$', fontsize=14)
    axs[0].set_xlabel(r'$p_\mathrm{bifr}$', fontsize=14)



    axs[1].plot(list_k01, pbifr_ptb, '-o')
    axs[1].scatter(0., model_summary['p_bifr'], color='green', s=150, marker='*')
    axs[1].set_xlabel(r'$k_{01}$', fontsize=14)
    axs[1].set_ylabel(r'$p_\mathrm{bifr}$', fontsize=14)


    axs[2].plot(list_k01, cbifr_ptb, '-o')
    axs[2].scatter(0., model_summary['c_bifr'], color='green', s=150, marker='*')
    axs[2].set_xlabel(r'$k_{01}$', fontsize=14)
    axs[2].set_ylabel(r'$c_\mathrm{bifr}$', fontsize=14)


    fig.tight_layout()
    plt.show()






