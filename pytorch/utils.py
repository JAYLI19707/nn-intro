"""数据生成与可视化工具（PyTorch 版本）

与 numpy/utils.py 功能相同，额外支持 PyTorch Tensor。"""

import numpy as np
import random
import matplotlib.pyplot as plt
import torch

CLASSIFICATION_TYPE = 'ring'


def tag_entry(x, y):
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
    """生成随机二分类数据，返回 numpy 数组"""
    entry_list = []
    for _ in range(num_of_data):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        entry_list.append([x, y, tag_entry(x, y)])
    return np.array(entry_list)


def create_tensors(num_of_data):
    """生成 PyTorch 张量版本的数据"""
    data = create_data(num_of_data)
    X = torch.tensor(data[:, :2], dtype=torch.float32)
    y = torch.tensor(data[:, 2], dtype=torch.long)
    return X, y


def plot_data(data, title="Data"):
    """可视化二分类数据"""
    colors = ["orange" if label == 0 else "blue" for label in data[:, 2]]
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
