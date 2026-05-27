# NumPy 教程：从零手搓神经网络

用纯 NumPy 实现神经网络的每一个组件。不依赖任何深度学习框架，帮你理解底层原理。

## 前置知识

- Python 基础（函数、类、循环）
- 高中数学（矩阵、函数）

## 学习路线

| # | 文件 | 内容 | 预计时间 |
|---|------|------|----------|
| 1 | `01_introduce_numpy.ipynb` | NumPy 基础：数组、索引、矩阵运算、聚合 | 30 分钟 |
| 2 | `02_neurons.ipynb` | 单神经元 → 多神经元，ReLU，批处理 | 30 分钟 |
| 3 | `03_layers_and_network.ipynb` | Layer 类 + Network 类，面向对象封装 | 30 分钟 |
| 4 | `04_softmax.ipynb` | Softmax 激活函数，概率输出 | 20 分钟 |
| 5 | `05_classification.ipynb` | 分类任务实战，可视化预测结果 | 20 分钟 |
| 6 | `06_loss_and_demands.ipynb` | 损失函数 + 需求函数，One-hot 编码 | 30 分钟 |
| 7 | `07_backpropagation.ipynb` | 完整反向传播训练，看到网络学会分类 | 30 分钟 |
| 8 | `08_autograd_from_scratch.ipynb` | 自动微分 Value 类，计算图原理 | 40 分钟 |

**总预计时间：约 3.5 小时**

## 工具文件

- `utils.py` — 数据生成与可视化工具函数

## 学完后你将理解

- 神经网络的前向传播是如何用矩阵乘法实现的
- 激活函数（ReLU、Softmax）的作用
- 损失函数如何量化预测质量
- 反向传播如何计算梯度并更新权重
- 自动微分的核心原理（计算图 + 链式法则）
