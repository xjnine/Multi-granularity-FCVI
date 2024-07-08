import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from skfuzzy.cluster import cmeans
from GBS import *
from LPS import *


def fcp(membership_matrix, data, centers, m):
    K = centers.shape[0]
    N = data.shape[0]
    fcp_k = 0
    max_membership = np.max(membership_matrix, axis=1)
    sum_max_membership = sum(max_membership)
    for k in range(K):
        numerator = sum(
            [(membership_matrix[i, k] ** m) * (np.linalg.norm(data[i] - centers[k]) ** 2) for i in range(N)])
        denominator = sum_max_membership
        fcp_k += numerator / denominator
    return fcp_k


def fm1(N, v):
    K = len(v)
    avg_v = np.mean(v, axis=0)
    sum_squared_distances = np.sum((v - avg_v) ** 2)
    fm1 = N * (sum_squared_distances / (K - 1))
    return fm1


def fm2(v):
    K = len(v)
    sum_squared_dist = 0.0
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            squared_dist = np.sum((v[i] - v[j]) ** 2)
            sum_squared_dist += squared_dist
            count += 1
    fm2 = sum_squared_dist / count
    return fm2


def fm3(v):
    min_dist = np.inf
    for i in range(len(v)):
        for j in range(len(v)):
            if i != j:
                dist = np.linalg.norm(v[i] - v[j]) ** 2
                min_dist = min(min_dist, dist)
    return min_dist


def fsp(centers, N):
    return fm1(N, centers) * fm2(centers) * fm3(centers)


def TCR(data, k, m=2):
    # labels, membership_matrix, centers = fcm_main(data, c_clusters=k, m=m, max_it=50)
    cntr, u, u0, d, jm, p, fpc = cmeans(data.T, k, m=2, error=0.005, maxiter=1000, init=None)
    # return fcp(membership_matrix, data, centers, m) / fsp(centers, len(data))
    return fcp(u.T, data, cntr, m) / fsp(cntr, len(data))


if __name__ == '__main__':
    datasets = ['4', '7', '8', '9', 'a1', 'a3', 'D19', 'D20', 'D21', 'data_01', 'data_02', 'E6', 'fc1',
                'sn', 'Triangle1']

    for d in datasets:
        data = pd.read_csv('./datasets/%s.csv' % d)
        data = data.values
        data = np.unique(data, axis=0)
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        # data = findCore(data)
        # data = get_GB_centers(data)
        # print(data)
        st = time.time()
        # 对于单个数据集的实验
        on = 0
        min_tcr = 1e9
        its = int(len(data) ** 0.5) + 1
        for k in range(2, its):
            for fcms in range(0, 10):
                tcr = TCR(data, k)
                if min_tcr > tcr:
                    min_tcr = tcr
                    on = k
            # print("k={0}:  {1}".format(k, tcr))
        ed = time.time()
        print("dataset: %s\non: %d\ntime: %.2fs\n" % (d, on, (ed - st) / 10))
