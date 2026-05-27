"""数据生成与可视化工具

提供二分类数据的生成和可视化功能，供后续教程使用。
"""

import numpy as np
import random
import matplotlib.pyplot as plt

# 数据生成配置
CLASSIFICATION_TYPE = 'ring'  # 可选: 'circle', 'ring', 'line', 'cross'


def tag_entry(x, y):
    """根据坐标 (x, y) 判定类别标签"""
    if CLASSIFICATION_TYPE == 'circle':
        return 1 if x**2 + y**2 > 1 else 0

    if CLASSIFICATION_TYPE == 'ring':
        return 1 if 1 < x**2 + y**2 < 2 else 0

    if CLASSIFICATION_TYPE == 'line':
        return 1 if x > 0 else 0

    if CLASSIFICATION_TYPE == 'cross':
        return 1 if x * y > 0 else 0

    return 0


def create_data(num_of_data):
    """生成随机二分类数据

    返回:
        np.ndarray: shape=(num_of_data, 3)，每行 [x, y, label]
    """
    entry_list = []
    for _ in range(num_of_data):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        tag = tag_entry(x, y)
        entry_list.append([x, y, tag])
    return np.array(entry_list)


def plot_data(data, title="Data"):
    """可视化二分类数据，橙色=类别0，蓝色=类别1"""
    colors = ["orange" if label == 0 else "blue" for label in data[:, 2]]
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
