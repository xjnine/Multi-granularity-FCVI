import time
from GBS import *
import numpy as np
import pandas as pd
from skfuzzy import cmeans
from sklearn.preprocessing import MinMaxScaler
from LPS import *


def XBI(u, m, X, v):
    distances = []
    N, K = u.shape
    for i in range(K):
        for j in range(i + 1, K):
            distances.append(np.linalg.norm(v[i] - v[j]))
    min_dis = min(distances)
    xbi = 0
    for i in range(K):
        for j in range(N):
            term1 = np.linalg.norm(X[j] - v[i]) ** 2
            xbi += u[j, i] ** m * term1
    xbi /= (min_dis ** 2)
    return xbi


if __name__ == '__main__':
    datasets = ['4', '7', '8', '9', 'a1', 'a3', 'D19', 'D20', 'D21', 'data_01', 'data_02', 'E6', 'fc1',
                'sn', 'Triangle1']
    for dd in datasets:
        data = pd.read_csv('./datasets/%s.csv' % dd)
        data = data.values
        data = np.unique(data, axis=0)
        # 1 origin CVI
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        # 2 LPS CVI
        data = findCore(data)
        # 3 GB CVI
        data = get_GB_centers(data)
        st = time.time()
        on = 0
        xbi_min = np.inf
        its = int(len(data) ** 0.5) + 1
        for k in range(2, its):
            for fcms in range(0, 10):
                cntr, u, u0, _, jm, p, fpc = cmeans(data.T, k, m=2, error=0.005, maxiter=1000, init=None)
                xbi = XBI(u.T, 2, data, cntr)
                if xbi < xbi_min:
                    xbi_min = xbi
                    on = k
        ed = time.time()
        print("dataset: %s\non: %d\ntime: %.2fs\n" % (dd, on, (ed - st) / 10))

