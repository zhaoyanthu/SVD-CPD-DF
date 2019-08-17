import pickle
import numpy as np
import pandas as pd
import copy
import random
import statistics
import math
from cvxopt import matrix, solvers
import tensorly as tl
from tensorly.decomposition import parafac

solvers.options['show_progress'] = False


def performance_measure(TT, TT_til, CT):
    num_missing_entries = np.count_nonzero(CT == -1)

    estimate_list_all = list(np.ravel(TT_til))
    ground_truth_list_all = list(np.ravel(TT))
    CT_list = list(np.ravel(CT))

    estimate_list = [estimate_list_all[i] for i in range(len(ground_truth_list_all)) if ground_truth_list_all[i] != 0 \
                     and CT_list[i] == -1]
    ground_truth_list = [ground_truth_list_all[i] for i in range(len(ground_truth_list_all)) if
                         ground_truth_list_all[i] != 0 \
                         and CT_list[i] == -1]

    mape_list = [abs(x - y) / y for x, y in zip(estimate_list, ground_truth_list)]
    MAPE = np.mean(mape_list) if mape_list else 0
    rmse_list = [(x - y) ** 2 for x, y in zip(estimate_list, ground_truth_list)]
    RMSE = np.sqrt(np.mean(rmse_list) if rmse_list else 0)

    return num_missing_entries, MAPE, RMSE


def get_probe_data(TT, penetration_rate):
    # construct PVT
    PVT = copy.deepcopy(TT)
    for i in range(TT.shape[0]):
        for j in range(TT.shape[1]):
            PVT[i][j] = np.random.binomial(TT[i, j], penetration_rate)
    return PVT


def get_corrupt_data_tensor(TT, cp):
    CT_construct_flag = False
    while (CT_construct_flag == False):
        CT = corrupt_data(TT, cp)
        sum1 = np.sum(CT, axis=0)
        if 0 in sum1:
            CT_construct_flag = False
        else:
            CT_construct_flag = True
    return CT


# data corruption
def corrupt_data(m, corruption_rate):
    CT = copy.deepcopy(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            CT[i, j] = CT[i, j] if random.random() > corruption_rate else -1
    return CT


def get_corrupt_data_tensor_entire_LD_missing(TT_tensor, cp):
    CT_tensor = copy.deepcopy(TT_tensor)

    num_missing_matrices = cp
    matrices_idx_list = range(TT_tensor.shape[0])
    missing_matrices_idx = random.sample(matrices_idx_list, num_missing_matrices)
    for matrices_idx in missing_matrices_idx:
        CT_tensor[matrices_idx, :, :] = -1

    cp_percent = round((np.count_nonzero(CT_tensor == -1) / TT_tensor.size) * 100, 2)

    return CT_tensor, cp_percent


def fuse_data_tensor_new(TT, Up, CT, rank, ub, lmd=1e-5):
    """
    ub now is a list for different TOD.
    num_of_LD is the number of LDs used for calculating specific LD results

    This function is for general tensor formulation which can calculate all TODs and single TOD situation, and it can
    also calculate for all LDs and specific LD.

    IF all_TODs = True then calculate results for all TODs
       else for each TOD

    IF specific_LD is not zero then we will output results for all LDs missing entries and this specific LD missing
    entries. Otherwise, it will output results for all LDs missing entries and all LDs all entries results.
    """
    TT_til = np.array([[] for i in range(TT.shape[0])])

    for i in range(CT.shape[1]):
        b_corrupt = CT[:, i]
        b_all = TT[:, i]
        # rank = max(sum(num > 0 for num in b_corrupt) - 10, rank)
        tt_til_list, xtil_zero, xreal_zero = projection_coordinates_finding(Up[:, :rank], b_corrupt, b_all, ub[i], lmd)
        tt_til_array = np.reshape(np.array(tt_til_list), (-1, 1))
        TT_til = np.hstack((TT_til, tt_til_array))

    return TT_til


# Projection and result coordinates finding
def projection_coordinates_finding(A, b, b_all, ub, lmd):
    """
    input:
        A: subspace matrix U[:,1:rank]
        b: data need to project

    return:
        x_til_all: result after projection

    """
    A_tmp = []
    CT_tmp = []
    non_zero_index = []
    xtil_zero, xreal_zero = [], []
    for i in range(len(b)):
        if b[i] > 0:
            CT_tmp.append(b[i])
            A_tmp.append(A[i, :])
            non_zero_index.append(i)
    A_tmp, CT_tmp = np.array(A_tmp), np.array(CT_tmp)

    alpha = np.linalg.lstsq(A_tmp, CT_tmp.T, rcond=-1)[0]
    xtil_all = list(np.dot(A, alpha))

    indicator = len(non_zero_index) - A.shape[1]
    if not all(0 <= num <= ub for num in xtil_all) or (indicator < 0):
        alpha = RidgeQP(A, A_tmp, CT_tmp.T, ub, lmd)
        aaa = (np.dot(A, alpha)).T
        xtil_all = list(aaa[0])
    #     alpha = LassoQP(A, A_tmp, CT_tmp.T, ub, lmd ** 3)

    for i in range(len(b)):
        if i not in non_zero_index:
            xtil_zero.append(xtil_all[i])
            xreal_zero.append(b_all[i])
    return xtil_all, xtil_zero, xreal_zero


# Ridge QP to find alpha
def RidgeQP(A, Atil, btil, ub, lmd):
    """
    minimize ||A_qp * x - b_qp||2 + lambda * x^T * x
    s.t. 0 <= A_qp * x <= UB
    input:
        A_qp: actually is U
        b_qp: data need to be projected
        ub: maximum volume
        lmd: lambda

    output:
        alpha: weight for each basis
    """
    A = A.astype(np.double, copy=False)
    Atil = Atil.astype(np.double, copy=False)
    btil = btil.astype(np.double, copy=False)
    m, n = A.shape

    Q = matrix(2 * np.dot(Atil.T, Atil) + lmd * np.identity(n))
    p = matrix(-2 * np.dot(Atil.T, btil))
    G = matrix(np.concatenate((A, -A), axis=0))
    UB = ub * np.ones((m, 1))
    h = matrix(np.concatenate((UB, np.zeros(UB.shape)), axis=0))

    alpha = np.array(solvers.qp(Q, p, G, h)['x'])

    return alpha


if __name__ == "__main__":
    # Step 0: Hyperparameter
    rank = 2

    # Step 1: ground-truth data
    num_TOD = 24
    num_day = 20
    TOD_list = range(24)
    with open('TT.pickle', 'rb') as file_in:
        TT, weekday, dates = pickle.load(file_in)
    TT = TT[:, -num_day * num_TOD:]
    original_tensor = np.reshape(TT, (15, 20, -1))
    TOD_matrix = np.array([np.ravel(original_tensor[:, :, tod]) for tod in TOD_list]).T
    ub = [1.5 * np.max(TOD_matrix[:, i]) for i in range(TOD_matrix.shape[1])]
    original_matrix = np.array([np.ravel(original_tensor[:, :, tod]) for tod in TOD_list]).T

    # Step 2: generate input
    prr = 10
    cp = 10
    CT = get_corrupt_data_tensor(original_matrix, cp / 100)
    PVT_matrix = get_probe_data(TT, prr / 100)
    PVT_tensor = PVT_matrix.reshape((15, 20, -1))

    # Step 3: reconstruction
    tensor = tl.tensor(PVT_tensor, dtype=float)
    [F, G, H] = parafac(tensor, rank=rank, n_iter_max=100)
    Up = np.array([np.ravel(np.outer(F[:, i], G[:, i])) for i in range(rank)]).T
    TT_til = fuse_data_tensor_new(original_matrix, Up, CT, rank, ub)

    for tod_idx in range(len(TOD_list)):
        tod = TOD_list[tod_idx]
        TT_tod = original_matrix[:, tod_idx]
        CT_tod = CT[:, tod_idx]
        TT_til_tod = TT_til[:, tod_idx]
        num_missing_entries, MAPE, RMSE = performance_measure(TT_tod, TT_til_tod, CT_tod)
        print('TOD: ', tod_idx, ' MAPE: ', MAPE, ' RMSE: ', RMSE) 