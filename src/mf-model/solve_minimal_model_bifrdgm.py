import math
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

    # model prediction
    model = MFmodel(eps_11, eps_01, eps_00, q, K)

    # model_status = predict_model_behavior(eps_11, eps_01, eps_00, q, K)

    is_multi_roots = model.predict_if_bifr()
    print('model prediction: if in the bifurcation region = ', is_multi_roots)

    model_summary = {'p_ord':None, 'p_dis':None, 'p_bifr':None, \
            'c_coex_ord':None, 'c_coex_dis':None, 'c_coex_ord_est':None, 'c_coex_dis_est':None, 'c_bifr':None}

    npannel = 3
    fig, axs = plt.subplots(1, npannel, figsize=(4*npannel, 3))
    if is_multi_roots:
        #coex_points = model.solve_coexistence()
        #coex_points = sorted(coex_points)
        #model_summary['p_ord'] = coex_points[-1]
        #model_summary['p_dis'] = coex_points[0]
        #model_summary['c_coex_ord'] = model.get_coex()
        #model_summary['c_coex_dis'] = model.get_coex()
        model_summary['c_coex_ord_est'] = np.exp(model.est_order_coex_lnc())
        model_summary['c_coex_dis_est'] = np.exp(model.est_disorder_coex_lnc())

        model_summary['p_bifr'], model_summary['c_bifr'], _ = model.solve_for_bifr()

        axs[0].vlines(np.log(model_summary['c_coex_ord_est']), 0, 1, linestyle='dashed', color='grey')
        axs[0].vlines(np.log(model_summary['c_coex_dis_est']), 0, 1, linestyle='dashed', color='grey')
        axs[0].scatter(np.log(model_summary['c_bifr']), model_summary['p_bifr'], marker='*', color='black')

        Ndot_bifr = model.get_Ndot(model_summary['p_bifr'], model_summary['c_bifr'])
        axs[1].scatter(np.log(model_summary['c_bifr']), Ndot_bifr)

    print(model_summary)
    print('start of the actual brute-force solver')
    FILENAME = prefix + '.dat'
    with open(FILENAME, 'w') as f:
        f.write("# bze = %g\n# K = %d\n#q = %g\n\n" % (eps_11, K, q))
        f.write("# c p_tot p_max |dotp| isstable dotN p_i... eig(J)...\n\n")
        prev_solution = []
        for mu in np.linspace(-14., 0., 51):
            c = np.exp(mu)
            dotp, dotN, ddotpdp = gen_minimal_model(eps_11, eps_01, eps_00, c) # N = 2
            sp = search_sp(dotp, K, prev_solution=prev_solution)
            for sp_j in sp:
                p_tot = sp_j[1:].sum()
                p_max = sp_j[1:].max()
                dp = dotp(sp_j)
                dN = dotN(sp_j)
                ddp = ddotpdp(sp_j)
                eigs = np.linalg.eigvals(ddp)
                isstable = np.all(eigs < 0.)

                if isstable:
                    prev_solution.append(sp_j)
                f.write("%g %g %g %g %d %g %s %s \n" % \
                        (c, p_tot, p_max, np.linalg.norm(dp), isstable, dN, \
                         ' '.join(str(x) for x in sp_j), ' '.join(str(x) for x in eigs)))
            f.write("\n")
            f.flush()

    # plot the bifurcation diagram
    data = np.genfromtxt(FILENAME, dtype=np.complex_)
    markers = ['o', 'x']
    mks = ['o' if _ > 1e-6 else 'x' for _ in data[:, 4]]
    mapper = color_mapping(np.array(data[:, 5], dtype=float))
    for i in range(len(data[:, 0])):
        mu = '%.4g' %np.real(data[i, 0])
        axs[0].scatter(np.log(data[i, 0]), data[i, -2], color=['r', 'b'][int(float(data[i, 5]) > 0)], marker=['.', 'x'][int(float(data[i, 4]) < 1e-6)], alpha=0.5)

    axs[0].set_xlabel(r'$\beta\mu$', fontsize=14)
    axs[0].set_ylabel(r'$p$', fontsize=14)

    # plot growth speed for
    bf_ordered, bf_disordered, bf_coex, bf_bfr_lf, bf_bfr_rt = get_solutions(FILENAME)

    # ordered branch
    datax, datay = [], []
    for mu in sorted(bf_ordered.keys()):
        datax.append(np.log(mu))
        datay.append(bf_ordered[mu][0][-1])
    datay_masked = np.ma.masked_greater_equal(datay, 0)
    axs[1].plot(datax, datay, 'b')
    #axs[1].plot(datax, datay_masked, 'r', linewidth=2)

    # disordered branch
    dataxx, datayy = [], []
    for mu in bf_disordered:
        dataxx.append(np.log(mu))
        datayy.append(bf_disordered[mu][0][-1])
    datayy_masked = np.ma.masked_greater_equal(datayy, 0)
    axs[1].plot(dataxx, datayy, 'b')
    #axs[1].plot(dataxx, datayy_masked, 'r', linewidth=2)
    #axs[1].set_ylim(-0.02, 0.02)
    axs[1].set_yscale('log')
    axs[1].set_xlim(-14, 0)
    axs[1].set_xlabel(r'$\beta\mu$', fontsize=14)
    axs[1].set_ylabel(r'$\dot{N}$', fontsize=14)

    # TODO: add the points predicted by the model

    #plt.show()
    fig.tight_layout()
    plt.savefig('diagram_%.3f_%.3f.svg' % (clargs.ratio_01, clargs.ratio_00))



