# PyTorch 教程：用框架构建神经网络

用 PyTorch 实现与 NumPy 教程相同的知识点，体验框架带来的便利。

## 前置知识

- Python 基础
- 建议先完成 NumPy 教程（至少 01-05）

## 学习路线

| # | 文件 | 对应 NumPy | 内容 | 预计时间 |
|---|------|-----------|------|----------|
| 1 | `01_tensor_basics.ipynb` | 01 | Tensor 创建、运算、GPU、与 NumPy 互转 | 20 分钟 |
| 2 | `02_autograd.ipynb` | 08 | 自动微分：requires_grad, backward | 20 分钟 |
| 3 | `03_nn_module.ipynb` | 02-03 | nn.Module, nn.Linear, 构建网络 | 20 分钟 |
| 4 | `04_softmax_and_classification.ipynb` | 04-05 | Softmax 分类，argmax | 15 分钟 |
| 5 | `05_loss_and_optimizer.ipynb` | 06-07 | CrossEntropyLoss, optim.SGD, 训练循环 | 20 分钟 |
| 6 | `06_full_training.ipynb` | 综合 | 端到端训练、决策边界、模型保存 | 30 分钟 |

**总预计时间：约 2 小时**

## 工具文件

- `utils.py` — 数据生成与可视化工具（支持 NumPy 和 PyTorch）

## 学完后你将掌握

- PyTorch Tensor 的基本操作
- 自动微分机制（autograd）
- 用 nn.Module 构建自定义网络
- 标准的训练循环（forward → loss → backward → step）
- 模型的保存与加载
- 决策边界的可视化
