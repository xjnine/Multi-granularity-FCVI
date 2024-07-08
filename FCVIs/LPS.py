import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial import distance_matrix, distance
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# 6
def Kruskal(D):
    a = np.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i < j:
                a[i, j] = np.sqrt((D[i, 0] - D[j, 0]) ** 2 + (D[i, 1] - D[j, 1]) ** 2)
    G = nx.from_numpy_matrix(a)
    T = nx.minimum_spanning_tree(G)
    edges = T.edges(data=True)
    result = [[u, v, data['weight']] for u, v, data in edges]
    result = np.transpose(result)

    return T, result


# 1
def findCore(D):
    Sup, NN, RNN, NNN, nb, A = NaNSearching(D)
    CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL = findCenter2(D, NN, NNN, A, nb)
    LPS, FLP, T2 = findDensityPeak(CP, D, r1, rf, Pr1, Pr2, nb, Sup, CL)
    return D[LPS, :]
    # return CL, LPS, CP, CPV


# 2
def NaNSearching(D):
    r = 1
    nb = np.zeros((D.shape[0], 1))
    C = [None] * D.shape[0]
    NN = [[] for _ in range(D.shape[0])]
    RNN = [[] for _ in range(D.shape[0])]
    NNN = [[] for _ in range(D.shape[0])]
    A = distance_matrix(D, D)
    Numb1 = 0
    Numb2 = 0
    for ii in range(D.shape[0]):
        sa_index = np.argsort(A[:, ii])
        C[ii] = np.column_stack([A[:, ii][sa_index], sa_index])
    while (r < D.shape[0]):
        for kk in range(D.shape[0]):
            x = kk
            y = C[x][r + 1, 1]
            nb[int(y)] = nb[int(y)] + 1
            NN[x].append(int(y))
            RNN[int(y)].append(x)
        Numb1 = np.sum(nb == 0)
        if Numb2 != Numb1:
            Numb2 = Numb1
        else:
            break
        r = r + 1
    for jj in range(D.shape[0]):
        NNN[jj] = np.intersect1d(NN[jj], RNN[jj])
    Sup = r
    return Sup, NN, RNN, NNN, nb, A


# 3
def findCenter2(D, NN, NNN, A, nb):
    r1 = np.zeros(D.shape[0])
    r2 = np.zeros(D.shape[0])
    rf = np.zeros(D.shape[0])
    Pr1 = np.zeros(D.shape[0])
    Pr2 = np.zeros(D.shape[0])
    CPV = np.zeros(D.shape[0])  # 用于噪声点的检查
    CP = np.zeros(D.shape[0])
    Nei1 = [None] * D.shape[0]
    Nei2 = [None] * D.shape[0]
    CL = np.zeros(D.shape[0])

    for kk in range(D.shape[0]):
        CL[kk] = 0

    for ii in range(D.shape[0]):
        if len(NN[ii]) != 0:
            r1[ii] = 1 * np.max(np.linalg.norm(D[ii, :] - D[NN[ii], :], axis=1))
            r2[ii] = np.max(np.linalg.norm(D[ii, :] - D[NN[ii], :], axis=1))
            rf = r1 * 0.95
            Nei1[ii] = np.where(A[:, ii] < r1[ii])[0]
            Nei2[ii] = np.where(A[:, ii] < rf[ii])[0]
            Pr1[ii] = Nei1[ii].shape[0]
            Pr2[ii] = Nei2[ii].shape[0]
        else:
            r1[ii] = 0
            r2[ii] = 0
            rf[ii] = 0

    # r2 就是 RVarList，表示动态扫描半径从小到大排序，再依次计算RVar。
    B = np.mean(r2) + 2 * np.std(r2)
    for ii in range(D.shape[0]):
        if r2[ii] > B:
            CL[ii] = -1
        if r2[ii] == 0:
            CL[ii] = -1
        if nb[ii] < 2:
            CL[ii] = -1

    for jj in range(D.shape[0]):
        Nei1[ii] = np.setdiff1d(Nei1[ii], np.where(CL[Nei1[ii]] == -1))

    for ii in range(D.shape[0]):
        if len(Nei1[ii]) != 0:
            if CL[ii] != -1:
                y = np.argmin(np.linalg.norm(D[Nei1[ii], :] - np.mean(D[Nei1[ii], :], axis=0), axis=1))
                if CPV[Nei1[ii][y]] == ii:
                    CPV[ii] = ii
                else:
                    CPV[ii] = Nei1[ii][y]
            else:
                CPV[ii] = ii
        else:
            CPV[ii] = ii

    for ii in range(D.shape[0]):
        if CL[ii] != -1:
            CP[ii] = ii
            while CP[ii] != CPV[int(CP[ii])]:
                CP[ii] = CPV[int(CP[ii])]
        else:
            CP[ii] = ii

    return CPV, CP, Pr1, Pr2, r1, rf, r2, Nei1, CL


# 4
def findDensityPeak(CP, D, r1, rf, Pr1, Pr2, nb, Sup, CL):
    LPS = []
    FLP = []
    T2 = []
    for ii in range(D.shape[0]):
        if CL[ii] != -1:
            if CP[ii] == ii:
                LPS.append(ii)
    return LPS, FLP, T2
