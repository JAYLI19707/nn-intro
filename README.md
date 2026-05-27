# 从零开始的神经网络教程

一个面向初学者的神经网络学习项目。通过 **14 个 Notebook**，从 NumPy 基础到 PyTorch 框架，逐步构建你的第一个神经网络。

## 学习路线

```
                    ┌─────────────────────────────────────┐
                    │         从零开始的神经网络教程         │
                    └─────────────────────────────────────┘
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                         ▼
    ┌─────────────────────┐                 ┌─────────────────────┐
    │    NumPy 从零手搓     │                 │    PyTorch 框架      │
    │    （理解原理）       │                 │    （高效实践）       │
    └─────────────────────┘                 └─────────────────────┘
                │                                         │
    01 NumPy 基础                          01 Tensor 基础
    02 神经元与激活函数                     02 自动微分 (autograd)
    03 Layer 与 Network 类                 03 nn.Module 构建网络
    04 Softmax 激活函数                     04 Softmax 分类
    05 分类任务实战                         05 损失函数与优化器
    06 损失函数与需求函数                   06 综合实战
    07 完整反向传播训练
    08 自动微分 (Value 类)
                │                                         │
                └────────────────────┬────────────────────┘
                                     ▼
                        ┌─────────────────────────┐
                        │  你已理解神经网络的全部原理 │
                        └─────────────────────────┘
```

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 开始学习

1. 进入 `numpy/` 文件夹，按顺序学习 01-08
2. 进入 `pytorch/` 文件夹，按顺序学习 01-06

## 项目结构

```
NeuralNetwork/
├── README.md              ← 你在这里
├── requirements.txt       ← 依赖清单
├── .gitignore
│
├── numpy/                 ← NumPy 版：从零手搓
│   ├── README.md          ← NumPy 路线说明
│   ├── utils.py           ← 数据生成与可视化
│   ├── 01_introduce_numpy.ipynb
│   ├── 02_neurons.ipynb
│   ├── 03_layers_and_network.ipynb
│   ├── 04_softmax.ipynb
│   ├── 05_classification.ipynb
│   ├── 06_loss_and_demands.ipynb
│   ├── 07_backpropagation.ipynb
│   └── 08_autograd_from_scratch.ipynb
│
└── pytorch/               ← PyTorch 版：框架实现
    ├── README.md          ← PyTorch 路线说明
    ├── utils.py           ← 数据生成与可视化
    ├── 01_tensor_basics.ipynb
    ├── 02_autograd.ipynb
    ├── 03_nn_module.ipynb
    ├── 04_softmax_and_classification.ipynb
    ├── 05_loss_and_optimizer.ipynb
    └── 06_full_training.ipynb
```

## 两条路线怎么选？

| | NumPy 路线 | PyTorch 路线 |
|---|---|---|
| **目标** | 理解原理 | 高效实践 |
| **适合** | 想知道"为什么"的人 | 想知道"怎么用"的人 |
| **难度** | 较高（手写每个细节） | 较低（框架封装） |
| **时间** | ~3.5 小时 | ~2 小时 |
| **建议** | 先学这条 | 学完 NumPy 后再学 |

**推荐路线：** 先学 NumPy 理解原理，再学 PyTorch 体验框架。

## 技术栈

- Python 3.8+
- NumPy
- PyTorch
- Matplotlib
