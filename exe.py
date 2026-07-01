import rho
import baselinednn
import baselinekrr
import baselinepinn
import kernel
import numpy as np
import pandas as pd
import os

def res_krr(n, B, methods, case):
    col_name = []
    p, lam, gamma = get_hyper(n=n, case=case)

    for method in methods:
        col_name.append('MSE_' + method)
        col_name.append('ptcov_' + method)
        col_name.append('unicov_' + method)
        col_name.append('rad_' + method)

    num = len(methods) * 4
    res = pd.DataFrame(np.zeros([B, num]), columns=col_name)

    s = 2
    d = 3
    if case == 'bias':
        p_true = np.array([0.9, 0.8])
        q_true = np.array([0.0, 0.0])
    else:
        p_true = np.array([0.5, 0.5])
        q_true = np.array([0.6, 0.4])

    coeffs = {
        (2, 0, 0): p[0],
        (0, 2, 0): p[1],
        (0, 0, 2): 1.0,
    }

    test_size = 100
    for b in range(B):
        # p, lam, gamma = cal_hyper(n=n, case=case, seed=b)
        if case == 'bias':
            X_train, Y_train = rho.generate_data(n, p_true, d, seed=b)
        else:
            X_train, Y_train = rho.generate_full_data(n, p_true, q_true, d, seed=b)
        for method in methods:
            if method == 'ours':
                radius, point_coverage, unif_coverage, MSE \
                    = kernel.get_res(X_train, Y_train, test_size, lam, gamma, s, coeffs, case)
            elif method == 'krr':
                radius, point_coverage, unif_coverage, MSE \
                    = baselinekrr.get_res(X_train, Y_train, test_size, case)
            elif method == 'pinn':
                radius, point_coverage, unif_coverage, MSE \
                    = baselinepinn.get_res(X_train, Y_train, test_size, coeffs=p, seed=b, case=case)
            elif method == 'dnn':
                radius, point_coverage, unif_coverage, MSE \
                    = baselinednn.get_res(X_train, Y_train, test_size, seed=b, case=case)
            else:
                radius, point_coverage, unif_coverage, MSE = 0, 0, 0, 0

            col1 = 'rad_' + method
            res.loc[b, col1] = radius
            col2 = 'ptcov_' + method
            res.loc[b, col2] = point_coverage
            col3 = 'unicov_' + method
            res.loc[b, col3] = unif_coverage
            col4 = 'MSE_' + method
            res.loc[b, col4] = MSE
        if b % 20 == 0:
            print(b)

    if case == 'bias':
        res.to_csv('bias/n' + str(n) + '.csv')
    else:
        res.to_csv('appr/n' + str(n) + '.csv')
    return


def run():
    methods = ['ours', 'krr', 'pinn', 'dnn']
    for i in range(2000, 4500, 500):
        n = i
        B = 200
        case = 'bias'
        res_krr(n, B, methods, case)
        case = 'appr'
        res_krr(n, B, methods, case)


def get_hyper(n, case):
    hyper_bias = {
        2000: [[0.9127364, 0.8169467], 10 ** (-5), 0.05],
        2500: [[0.93032634, 0.8235161], 10 ** (-5), 0.05],
        3000: [[0.9409481, 0.81667525], 10 ** (-5), 0.05],
        3500: [[0.9348838, 0.8149152], 10 ** (-5), 0.05],
        4000: [[0.92824733, 0.8184325], 10 ** (-5), 0.05]
    }
    hyper_appr = {
        2000: [[0.5124496, 0.51500595], 10 ** (-4.6), 0.04],
        2500: [[0.51793784, 0.5177529], 10 ** (-4.6), 0.04],
        3000: [[0.5283357, 0.5259519], 10 ** (-4.6), 0.04],
        3500: [[0.533985, 0.51522547], 10 ** (-4.6), 0.04],
        4000: [[0.5318166, 0.5187776], 10 ** (-4.6), 0.04]
    }
    if case == 'bias':
        p = hyper_bias[n][0]
        lam = hyper_bias[n][1]
        gam = hyper_bias[n][2]
    elif case == 'appr':
        p = hyper_appr[n][0]
        lam = hyper_appr[n][1]
        gam = hyper_appr[n][2]
    else:
        p, lam, gam = 0, 0, 0
    return p, lam, gam

def cal_hyper(n, case, seed=0):
    s = 2
    d = 3
    if case == 'bias':
        p_true = np.array([0.9, 0.8])
        q_true = np.array([0.0, 0.0])
        X, Y = rho.generate_data(n, p_true, d, seed=seed)
    else:
        p_true = np.array([0.5, 0.5])
        q_true = np.array([0.6, 0.4])
        X, Y = rho.generate_full_data(n, p_true, q_true, d, seed=seed)

    p = rho.get_p(X, Y, seed=0)
    coeffs = {
        (2, 0, 0): p[0],
        (0, 2, 0): p[1],
        (0, 0, 2): 1.0,
    }

    lam, gamma = kernel.get_lam_gam(X, Y, coeffs, s)
    return p, lam, gamma

if __name__ == "__main__":
    paths = ['bias', 'appr']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    run()
    exit(0)