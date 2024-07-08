import time
import numpy as np
import pandas as pd
from skfuzzy import cmeans
from sklearn.preprocessing import MinMaxScaler
from GBS import *
from LPS import *


def PC(mu):
    N, K = mu.shape
    return np.sum(mu ** 2) / N


if __name__ == '__main__':
    datasets = ['4', '7', '8', '9', 'a1', 'a3', 'D19', 'D20', 'D21', 'data_01', 'data_02', 'E6', 'fc1',
                'sn', 'Triangle1']
    for dd in datasets:
        data = pd.read_csv('./datasets/%s.csv' % dd)
        data = data.values
        data = np.unique(data, axis=0)
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        # data = findCore(data)
        # data = get_GB_centers(data)

        st = time.time()
        on = 0
        pc_max = -np.inf
        its = int(len(data) ** 0.5) + 1
        for k in range(2, its):
            for fcms in range(10):
                cntr, u, u0, _, jm, p, fpc = cmeans(data.T, k, m=2, error=0.005, maxiter=1000, init=None)
                pc = PC(u.T)
                if pc > pc_max:
                    pc_max = pc
                    on = k
        ed = time.time()
        print("dataset: %s\non: %d\ntime: %.2fs\n" % (dd, on, (ed - st) / 10))
