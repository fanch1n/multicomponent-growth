import numpy as np
from scipy.special import lambertw
from scipy.optimize import fsolve, minimize, least_squares
import subprocess


class MFmodel:
    def __init__(self, u_11, u_01, u_00, q, K=1):
        self.u_11 = u_11
        self.u_01 = u_01
        self.u_00 = u_00
        self.q = q
        self.K = K
        self.a = self.u_11 + self.u_00 - 2. * self.u_01
        self.b = np.log(self.q - self.K) - self.u_00 + self.u_01


    def predict_if_bifr(self):
        # convert to the a-b plane for prediction
        a = self.u_11 + self.u_00 - 2. * self.u_01
        b = np.log(self.q - self.K) - self.u_00 + self.u_01
        def F(x, r):
            return x * np.exp(x) + r * x

        r = np.exp(-b)
        alpha = lambertw(-r * np.exp(1), -1) - 1.
        beta = lambertw(-r * np.exp(1), 0) - 1.
        falpha = F(alpha, r)
        fbeta = F(beta, r)
        # condition for three distinct solutions:
        if np.real(fbeta) < a * r < np.real(falpha):
            return 1
        else:
            return 0

    def get_order_coex_lnc(self, p_ord):
        return np.log(p_ord) + self.u_11 * p_ord + self.u_01 * (1.-p_ord) + np.log(self.q)

    def get_disorder_coex_lnc(self, p_dis):
        return np.log(1.-p_dis) + self.u_00 * (1.-p_dis) + self.u_01 * p_dis + np.log(self.q/(self.q - self.K))

    def est_order_coex_lnc(self):
        # esitmation valid in the strong bond limit
        return self.u_11 + np.log(self.q)

    def est_disorder_coex_lnc(self):
        # estimation valid in the strond bond limit
        return self.u_00 + np.log(self.q / (self.q - self.K))

    def get_Ndot(self, p, c):
        return c - p * (1. - p) * np.exp((self.u_11 - self.u_01) * p + self.u_01) \
                - (1. - p) * np.exp((self.u_01 - self.u_00) * p + self.u_00)


    def solve_for_coexistence(self):
        '''
          brute-force way of solving for the coexistence
        '''
        ans = set()
        def coex_func(p, a, b):
            r = np.exp(-b)
            return r * p + p * np.exp(a * p) - np.exp(-b)
        for t in range(102):
            if t == 0:
                g0 = 1e-12
            elif t == 1:
                g0 = 0.99999
            else:
                g0 = np.random.rand()
                res = fsolve(coex_func, g0, args=(self.a, self.b), xtol=1e-12)
                p = res[0]
                if  abs(coex_func(p, self.a, self.b)) < 1e-9:
                    ans.add(np.round(p, 6))
        ans = sorted(ans)
        p_ord = ans[-1]
        p_dis = ans[0]

        return (p_ord, self.get_order_coex_lnc(p_ord)), (p_dis, self.get_disorder_coex_lnc(p_dis))

    def get_pcoex_sols(self):
        '''
          obtaining the coexistence solution(s) using the numerical approximation of the r-Lambert function
        '''
        output = subprocess.run(["./rLamb", str(self.a), str(self.b)], stdout = subprocess.PIPE,\
              universal_newlines = True).stdout

        lines = output.split('\n')
        p_sols = []
        for ln in lines:
            if 'W' in ln and '=' in ln:
                p_ = float(ln.split('=')[-1]) / self.a
                p_sols.append(p_)

        return sorted(p_sols)

    def gen_dynamical_system(self):
        A = lambda p: np.exp((self.u_11 - self.u_01) * p + self.u_01)
        B = lambda p: np.exp((self.u_01 - self.u_00) * p + self.u_00)

        def gfunc(p, c): # master equation defining fixed points: dotp = 0
            return c / self.q - p * (A(p) - B(p) + c) + p**2 * (A(p) - B(p))

        def dgdp(p, c): # partial derivative of g with respect to p
            factor = (self.u_11 - self.u_01) * A(p) - (self.u_01 - self.u_00) * B(p)
            return A(p) * (2*p - 1) + B(p) * (1 - 2*p) - c - p * factor + p**2 * factor

        return gfunc, dgdp


    def solve_for_bifr(self):
        gfunc, dgdp = self.gen_dynamical_system()

        def gen_nonlinear_eqns(x):
            '''
                Given the dynamical system dot p = g(p, c), simultaneoulsy solve for dot p = 0 and dg/dp = 0
            '''
            eqs = np.zeros(2)
            p, c = x[0], x[1]
            eqs[0] = gfunc(p, c)
            eqs[1] = dgdp(p, c)
            return eqs

        p_ans, c_ans, resid = None, None, None
        for t in range(100):
            try:
                p_guess = np.random.random()
                mu_guess = np.random.random() * (self.u_11 - (-1.)) + self.u_11
                x0 = np.array([p_guess, np.exp(mu_guess)])
                res = least_squares(gen_nonlinear_eqns, x0, ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=0)
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


    def get_equal_Ndot_sol(self, p_bifr, c_bifr):
        Ndot_reference = self.get_Ndot(p_bifr, c_bifr)
        gfunc, dgdp = self.gen_dynamical_system()

        def eqns(x):
            p, c = x
            eqns = np.zeros(2)
            eqns[0] = gfunc(p, c) # steady-state condition
            eqns[1] = self.get_Ndot(p, c) - Ndot_reference # equal speed condition
            return eqns

        p_ans, c_ans, resid = None, None, None
        for t in range(100):
            p_guess = np.random.random() * 1e-6
            mu_guess = np.random.random() * self.u_11 + self.u_11
            x0 = np.array([p_guess, np.exp(mu_guess)])
            res = least_squares(eqns, x0, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if 0 < res.x[0] < 1.:
                if not c_ans:
                    p_ans, c_ans = res.x
                    resid = res.fun
                else:
                    if np.linalg.norm(res.fun) < np.linalg.norm(resid):
                        p_ans, c_ans = res.x
                        resid = res.fun

        return p_ans, c_ans, resid


