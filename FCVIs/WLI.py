import time
import numpy as np
import pandas as pd
from skfuzzy import cmeans
from sklearn.preprocessing import MinMaxScaler
from GBS import *
from LPS import *

def EuclidDist(x, y):
    x = x.T
    y = y.T
    res = []
    n = len(x)
    for i in range(n):
        res.append(np.linalg.norm(x[i] - y[i]))
    return np.array(res)


def WLI(u, v, q, X):
    # 传入的X是2行N列
    # u是正常的隶属度矩阵
    # v是2行m列
    # print(v)
    # print("X")
    # print(X)
    k = v.shape[1]
    N = X.shape[1]
    f = np.sum(u, axis=0) / N
    # print(f)
    com = [None] * k
    compactness = [None] * k
    for j in range(0, k):
        # print(np.tile(v[:, j:j+1], N))
        a = EuclidDist(X, np.tile(v[:, j:j + 1], N))
        a = a ** 2 * u[:, j] ** q
        a = a.T
        com[j] = a
        compactness[j] = np.sum(a) / np.sum(u[:, j])
        # a = a.T
        # print(a.shape)
        # time.sleep(10000)
    compactness1 = compactness.copy()
    compactness = np.sum(compactness)
    # print(compactness1)
    # print(compactness)

    d = []
    d_index = []
    d_min = []
    for i in range(0, k - 1):
        for j in range(i + 1, k):
            if f[i] > f[j]:
                delta = f[i] / f[j]
            else:
                delta = f[j] / f[i]
            d1 = np.linalg.norm(v[:, i] - v[:, j])
            d.append(d1 ** 2 * 1)
            d_index.append([i, j])

    d = np.array(d)
    d_index = np.array(d_index)

    if np.sum(d) == 0:
        cvi = -1
    else:
        separation = (np.min(d) + np.median(d)) / 2
        d = np.hstack((d.reshape(-1, 1), d_index))
        cvi = compactness / (2 * separation)
    return cvi, compactness1, d, com


if __name__ == '__main__':
    datasets = ['4', '7', '8', '9', 'a1', 'a3', 'D19', 'D20', 'D21', 'data_01', 'data_02', 'E6', 'fc1',
                'sn', 'Triangle1']
    for dd in datasets:
        data = pd.read_csv('./datasets/%s.csv' % dd)
        data = data.values
        data = np.unique(data, axis=0)
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        st = time.time()
        # data = findCore(data)
        # data = get_GB_centers(data)

        # 对于单个数据集的实验
        on = 0
        min_wli = np.inf
        its = int(len(data) ** 0.5) + 1
        for k in range(2, its):
            for fcms in range(10):
                cntr, u, u0, d, jm, p, fpc = cmeans(data.T, k, m=2, error=0.005, maxiter=1000, init=None)
                wli, _, _, _ = WLI(u.T, cntr.T, 2, data.T)
                if min_wli > wli:
                    min_imi = wli
                    on = k
        ed = time.time()
        print("dataset: %s\non: %d\ntime: %.2fs\n" % (dd, on, (ed - st) / 10))
