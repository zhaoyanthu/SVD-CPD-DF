import pickle
import numpy as np
import pandas as pd
import copy
import random
import statistics
import math
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


def get_corrupt_data(original_matrix, cp, missing_pattern):
    """
    Return corrupted data matrix and real missing ratio if it is entire row missing
    :param original_matrix: matrix for desired formulation
    :param cp: for random missing is missing ratio test2
    :param missing_pattern: Missing at random and entire row missing
    :return:
    """
    CT_construct_flag = False
    num_row = original_matrix.shape[0]
    all_zero_sum_num = num_row * -1
    num_roll = 0

    CT = copy.deepcopy(original_matrix)
    col_idx_list_to_corrupt = range(original_matrix.shape[1])
    while (CT_construct_flag == False):
        num_roll += 1
        CT = missing_pattern(original_matrix, CT, cp, col_idx_list_to_corrupt)
        sum_column = np.sum(CT, axis=0)
        if all_zero_sum_num in sum_column:  # At least one observation is needed
            CT_construct_flag = False
            col_idx_list_to_corrupt = [col_idx for col_idx in range(original_matrix.shape[1]) if
                                       sum_column[col_idx] == all_zero_sum_num]
        else:
            CT_construct_flag = True

    # calculate real missing ratio
    if missing_pattern == entire_row_missing:
        cp = round((np.count_nonzero(CT == -1) / original_matrix.size) * 100, 2)

    return CT, cp


# separate TOD
def get_TOD_data(TT, tod_idx, num_TOD):
    if tod_idx >= num_TOD:
        assert False, 'tod_idx > num_TOD'
    original_matrix = TT[:, tod_idx::num_TOD]
    return original_matrix


def random_missing(orignal_matrix, CT, cp, col_idx_list):
    # CT = copy.deepcopy(original_matrix)
    for i in range(orignal_matrix.shape[0]):
        for j in col_idx_list:
            CT[i, j] = orignal_matrix[i, j] if random.random() > (cp / 100) else -1
    return CT


def entire_row_missing(original_matrix, CT, cp, col_idx_list):
    """
    for entire row missing don't need CT and col_idx_list, this is just for xingshitongyi
    :param original_matrix:
    :param CT:
    :param cp: num_missing_rows
    :param col_idx_list:
    :return:
    """
    CT = copy.deepcopy(original_matrix)
    num_missing_rows = cp
    row_idx_list = range(original_matrix.shape[0])
    missing_rows_idx = random.sample(row_idx_list, num_missing_rows)
    for row_idx in missing_rows_idx:
        CT[row_idx, :] = -1
    return CT


def get_probe_vehicle_data(original_matrix, prr, prr_pattern='same_prr'):
    """
    Get PVT for 3 different types of methods
    1. prr is the same for all columns
    2. prr is different for each column and generate randomly in a given range
    3. prr is different for each column and given in advance

    :param original_matrix: matrix sample from. Cannot whole row be zero
    :param prr: int for 1, [prr lb, prr ub] for 2, [prr col1, prr col2, ......, prr coln] for 3
    :param prr_pattern: same_prr, range_prr, given_prr
    :return: PVT
    """
    if (type(prr), str(prr_pattern)) not in [(int, 'same_prr'), (list, 'range_prr'), (list, 'given_prr')]:
        assert False, 'prr not consistent with prr pattern!'

    num_row, num_col = original_matrix.shape[0], original_matrix.shape[1]

    penetration_rate_list = []
    if prr_pattern == 'same_prr':
        penetration_rate_list = [(prr / 100) for i in range(num_col)]
    elif prr_pattern == 'range_prr':
        prr_lb, prr_ub = prr[0], prr[1]
        penetration_rate_list = [(random.randrange(prr_lb, prr_ub, 1) / 100) for i in range(num_col)]
    elif prr_pattern == 'given_prr':
        penetration_rate_list = [(num / 100) for num in prr]

    PVT = copy.deepcopy(original_matrix)
    for row_idx in range(num_row):
        PVT_row_construct_OK_flag = False  # for each row at least 1 observation is needed
        while PVT_row_construct_OK_flag is False:
            for col_idx in range(num_col):
                penetration_rate = penetration_rate_list[col_idx]
                PVT[row_idx, col_idx] = np.random.binomial(original_matrix[row_idx, col_idx], penetration_rate)
            row_sum = np.sum(PVT[row_idx, :])
            if row_sum != 0:
                PVT_row_construct_OK_flag = True
    return PVT


def performance_measure(TT, TT_til, CT):
    '''
    Calculate RMSE and MAPE of estimation
    :param TT: Ground truth
    :param TT_til: Estimation
    :param CT: Corrupted data
    :return: RMSE and MAPE of estimation
    '''
    num_missing_entries = np.count_nonzero(CT == -1)

    estimate_list_all = list(np.ravel(TT_til))
    ground_truth_list_all = list(np.ravel(TT))
    CT_list = list(np.ravel(CT))

    estimate_list = [estimate_list_all[i] for i in range(len(ground_truth_list_all)) if ground_truth_list_all[i] != 0 \
                     and CT_list[i] == -1]
    ground_truth_list = [ground_truth_list_all[i] for i in range(len(ground_truth_list_all)) if
                         ground_truth_list_all[i] != 0 and CT_list[i] == -1]

    mape_list = [abs(x - y) / y for x, y in zip(estimate_list, ground_truth_list)]
    MAPE = np.mean(mape_list) if mape_list else 0
    rmse_list = [(x - y) ** 2 for x, y in zip(estimate_list, ground_truth_list)]
    RMSE = np.sqrt(np.mean(rmse_list) if rmse_list else 0)

    return num_missing_entries, MAPE, RMSE


def data_fusion(TT, CT, PVT, rank, ub_list, lmd):
    """
    SVD_
    :param TT: original matrix, the ground truth
    :param CT: corrupted matrix, include missing entries to impute
    :param PVT: probe vehicle data matrix, used to construct subspace
    :param rank: the rank of subspace
    :param ub_list: upper bound list for each sample(column) of volume
    :param lmd: the penalty parameter
    :return: matrix after imputation
    """
    Up, sp, Vp = np.linalg.svd(PVT)
    Ur = Up[:, :rank]
    TT_til = np.array([[] for i in range(TT.shape[0])])
    for i in range(TT.shape[1]):
        tt_til_list = projection(Ur, CT[:, i], ub_list[i], lmd)
        tt_til_array = np.reshape(np.array(tt_til_list), (-1, 1))
        TT_til = np.hstack((TT_til, tt_til_array))

    indicator_matrix, indicator_matrix_rev = gen_indicator_matrix(CT)
    TT_til = TT_til * indicator_matrix_rev + CT * indicator_matrix

    return TT_til


def gen_indicator_matrix(CT):
    indicator_matrix = copy.deepcopy(CT)
    indicator_matrix_rev = copy.deepcopy(CT)

    for i in range(CT.shape[0]):
        for j in range(CT.shape[1]):
            indicator_matrix[i, j] = 1 if CT[i, j] != -1 else 0
            indicator_matrix_rev[i, j] = 0 if CT[i, j] != -1 else 1
    return indicator_matrix, indicator_matrix_rev


def projection(A, b, ub, lmd):
    """
    input:
        A: subspace matrix U[:,1:rank]
        b: data need to project

    return:
        btil: result(list) after projection
    """
    A_truncated = []
    b_truncated = []
    non_missing_index = []
    for i in range(len(b)):
        if b[i] != -1:
            b_truncated.append(b[i])
            A_truncated.append(A[i, :])
            non_missing_index.append(i)
    A_truncated, b_truncated = np.array(A_truncated), np.array(b_truncated)

    alpha = np.linalg.lstsq(A_truncated, b_truncated.T, rcond=-1)[0]
    btil = list(np.dot(A, alpha))

    indicator = len(non_missing_index) - A.shape[1]
    if not all(0 <= num <= ub for num in btil) or (indicator < 0):
        alpha = RidgeQP(A, A_truncated, b_truncated.T, ub, lmd)
        aaa = (np.dot(A, alpha)).T
        btil = list(aaa[0])
    #     alpha = LassoQP(A, A_tmp, CT_tmp.T, ub, lmd ** 3)

    return btil


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


def impute(method, original_matrix, CT, PVT, rank, ub_list, lmd=1e-5, tol_mu=1e-6, 
    tol_sigma2=1e-6, tol_prr = 1e-6, tol_W=1e-6, tol_eta2 = 1e-6, max_iter_num=100, tol_SVD_iterative=1e-6,
    consecutive_iter_error=1e-8):
    dummy_flag = True
    if method == 'Data_fusion':
        TT_til = data_fusion(original_matrix, CT, PVT, rank, ub_list, lmd)
        return TT_til, dummy_flag


if __name__ == "__main__":
    method = 'Data_fusion'

    # Step 0: Hyperparameter
    rank = 2

    # Step 1: ground-truth data
    num_TOD = 24
    num_day = 20
    tod = 0
    with open('TT.pickle', 'rb') as file_in:
        TT, weekday, dates = pickle.load(file_in)
    TT = TT[:, -num_day * num_TOD:]
    original_matrix = get_TOD_data(TT, tod, num_TOD)
    ub_list = [1.5 * np.max(original_matrix[:, i]) for i in range(original_matrix.shape[1])]

    # Step 2: generate input
    missing_pattern = random_missing  # entire_row_missing / random_missing
    prr = 10
    cp = 10
    CT, cp = get_corrupt_data(original_matrix, cp, missing_pattern=missing_pattern)
    PVT = get_probe_vehicle_data(original_matrix, prr)

    # Step 3: reconstruction
    TT_til, converge_flag = impute(method, original_matrix, CT, PVT, rank, ub_list)
    num_missing_entries, MAPE, RMSE = performance_measure(original_matrix, TT_til, CT)
    
    # print('Ground truth: \n', TT)
    # print('Input 1, loop detector data: \n', CT)
    # print('Input 2, probe vehicle data: \n', PVT)
    # print('Reconstructed results: \n', TT_til)
    print('MAPE: ', MAPE, ' RMSE: ', RMSE)
