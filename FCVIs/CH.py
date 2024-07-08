from sklearn.cluster import KMeans
import time
from GBS import *
from LPS import *
from sklearn.preprocessing import MinMaxScaler


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def CH(data, labels):
    # 计算样本总数
    n_samples = data.shape[0]
    # 计算簇的数量
    k = len(np.unique(labels))

    # 当簇的数量为1时，CH指数未定义，抛出异常
    if k == 1:
        raise ValueError("CH Index is not defined for a single cluster.")

    # 计算每个簇的均值向量
    cluster_means = np.array([data[labels == label].mean(axis=0) for label in np.unique(labels)])
    # 计算数据集的总均值向量
    overall_mean = data.mean(axis=0)

    # 计算类间散布矩阵（Between-cluster scatter matrix，简称BSM）
    BSM = np.sum([np.sum(labels == label) * euclidean_distance(cluster_mean, overall_mean) ** 2
                  for label, cluster_mean in zip(np.unique(labels), cluster_means)])

    # 计算类内散布矩阵（Within-cluster scatter matrix，简称WSM）
    WSM = np.sum([np.sum([euclidean_distance(data[i], cluster_means[label]) ** 2
                          for i in range(n_samples) if labels[i] == label])
                  for label in np.unique(labels)])

    # 计算CH指数
    CH = (BSM / WSM) * (n_samples - k) / (k - 1)
    return CH


if __name__ == '__main__':
    datasets = ['4', '7', '8', '9', 'a1', 'a3', 'D19', 'D20', 'D21', 'data_01', 'data_02', 'E6', 'fc1',
                'sn', 'Triangle1']
    for dd in datasets:
        data = pd.read_csv('./datasets/%s.csv' % dd)
        data = data.values
        data = np.unique(data, axis=0)
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        st = time.time()
        on = 0
        ch_max = -np.inf
        its = int(len(data) ** 0.5) + 1
        for k in range(2, its):
            kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
            labels = kmeans.fit_predict(data)
            ch = CH(data, labels)
            if ch > ch_max:
                ch_max = ch
                on = k
        ed = time.time()
        print("dataset: %s\non: %d\ntime: %.2fs\n" % (dd, on, ed - st))