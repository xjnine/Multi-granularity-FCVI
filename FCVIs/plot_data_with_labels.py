import matplotlib.pyplot as plt
import numpy as np


def plot_data_with_labels(data, labels, title="Scatter Plot with Labels", xlabel="Feature 1", ylabel="Feature 2"):
    """
    绘制二维数据集的散点图，并根据标签用不同颜色表示。

    参数：
    - data: 二维数据集，每行包含两个特征。
    - labels: 数据点对应的标签数组。
    - title: 图形标题，默认为"Scatter Plot with Labels"。
    - xlabel: X轴标签，默认为"Feature 1"。
    - ylabel: Y轴标签，默认为"Feature 2"。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    colorbar = plt.colorbar()
    colorbar.set_label('Labels')
    plt.show()
