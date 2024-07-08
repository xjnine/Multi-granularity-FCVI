import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from skfuzzy import cmeans
from sklearn.preprocessing import MinMaxScaler
from GBS import *
from LPS import *


def Co(membership, data, centers):
    K = centers.shape[0]
    N = data.shape[0]
    compactness_values = []

    for k in range(K):
        distances = np.linalg.norm(data - centers[k], axis=1) ** 2
        numerator = np.sum((membership[:, k] ** 2) * distances)
        denominator = np.sum(membership[:, k])
        compactness_k = numerator / denominator if denominator != 0 else 0
        compactness_values.append(compactness_k)

    Co_K = (K - 1) * np.max(compactness_values)
    return Co_K


def build_clusters(data, u):
    clusters = {}
    k = u.shape[0]
    for i in range(u.shape[1]):
        max_membership = np.argmax(u[:, i])
        if max_membership not in clusters:
            clusters[max_membership] = []
        clusters[max_membership].append(data[i])
    for key in clusters:
        clusters[key] = np.array(clusters[key])
    return clusters


def S(data, u):
    clusters = build_clusters(data, u)
    num_clusters = len(clusters)
    min_distances = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            cluster1 = clusters[i]
            cluster2 = clusters[j]
            distances = cdist(cluster1, cluster2, metric='euclidean')
            min_distance = np.min(distances)
            min_distances[i, j] = min_distance
            min_distances[j, i] = min_distance
    min_distances_array = min_distances[np.triu_indices(num_clusters, k=1)]
    return np.min(min_distances_array)


def SMI(data, k):
    cntr, u, u0, d, jm, p, fpc = cmeans(data.T, k, m=2, error=0.005, maxiter=1000, init=None)
    return Co(u.T, data, cntr) / S(data, u)


if __name__ == '__main__':
    datasets = ['4', '7', '8', '9', 'a1', 'a3', 'D19', 'D20', 'D21', 'data_01', 'data_02', 'E6', 'fc1',
                'sn', 'Triangle1']
    datasets = ['DS9', 'DS10']

    for d in datasets:
        data = pd.read_csv('/Users/yuanchuzhang/Desktop/Fuzzy/ADS/%s.csv' % d)
        data = data.values
        data = np.unique(data, axis=0)
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        # data = findCore(data)
        # data = get_GB_centers(data)
        # print(data)
        st = time.time()
        # 对于单个数据集的实验
        on = 0
        min_smi = np.inf
        its = int(len(data) ** 0.5) + 1
        for k in range(2, its):
            for fcms in range(0, 10):
                smi = SMI(data, k)
                if min_smi > smi:
                    min_smi = smi
                    on = k
        ed = time.time()
        print("dataset: %s\non: %d\ntime: %.2fs\n" % (d, on, (ed - st) / 10))