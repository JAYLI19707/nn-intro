"""Python手搓神经网络 - 完整实现版本

这是一个从零开始实现的神经网络，展示了矩阵级反向传播的完整应用。
相比于notebook中的教学版本,这个实现包含了完整的训练循环、
自适应学习策略和交互式界面。

主要特点：
- 矩阵级高效计算
- 自适应训练策略（强制训练、随机更新）
- 实时可视化
- 多种分类任务支持
"""

import numpy as np
import createDataAndPlot as cp
import copy
import math

# =========================== 超参数配置 ===========================
NETWORK_SHAPE = [2, 100, 200, 100, 50, 2]  # 网络结构：输入层->隐藏层->输出层
BATCH_SIZE = 30                             # 批次大小
LEARNING_RATE = 0.015                       # 学习率
LOSS_THRESHOLD = 0.1                        # 损失阈值，低于此值停止训练
FORCE_TRAIN_THRESHOLD = 0.05                # 强制训练阈值，改善率低于此值时强制更新

# =========================== 全局状态变量 ===========================
force_train = False      # 是否启用强制训练模式
random_train = False     # 是否启用随机更新模式
n_improved = 0           # 改善次数计数器
n_not_improved = 0       # 未改善次数计数器
current_loss = 1         # 当前损失值


# =========================== 数据预处理函数 ===========================

def normalize(array):
    """
    按行标准化函数
    
    将每行的所有元素除以该行绝对值的最大值，使每行的最大绝对值为1。
    这有助于防止梯度爆炸和改善训练稳定性。
    
    参数:
        array: numpy数组，形状为 (m, n)
    
    返回:
        norm: 标准化后的数组，每行最大绝对值为1
    
    示例:
        input:  [[2, 4], [6, 3]]  
        output: [[0.5, 1], [1, 0.5]]
    """
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)  # 每行的最大绝对值
    scale_rate = np.where(max_number == 0, 1, 1/max_number)        # 避免除零错误
    norm = array * scale_rate                                        # 按比例缩放
    return norm

def vector_normalize(array):
    """
    向量标准化函数
    
    将整个向量的所有元素除以向量中绝对值的最大值。
    主要用于偏置向量的标准化。
    
    参数:
        array: numpy数组，任意形状
    
    返回:
        norm: 标准化后的数组，最大绝对值为1
    """
    max_number = np.max(np.absolute(array))                    # 全局最大绝对值
    scale_rate = np.where(max_number == 0, 1, 1/max_number)   # 避免除零错误
    norm = array * scale_rate                                   # 按比例缩放
    return norm

# =========================== 激活函数 ===========================

def activation_ReLU(inputs):
    """
    ReLU激活函数
    
    ReLU(x) = max(0, x)
    引入非线性，同时解决梯度消失问题。
    
    参数:
        inputs: 输入数组
    
    返回:
        激活后的数组，负值被置为0
    """
    return np.maximum(0, inputs)

def classify(probabilities):
    """
    分类函数
    
    将softmax输出的概率转换为离散的类别标签。
    取第二列（正类）概率，四舍五入得到0或1的分类结果。
    
    参数:
        probabilities: softmax输出，形状为 (m, 2)
    
    返回:
        classification: 分类结果，形状为 (m,)，值为0或1
    """
    classification = np.rint(probabilities[:, 1])  # 对正类概率四舍五入
    return classification
    
def activation_softmax(inputs):
    """
    Softmax激活函数
    
    将任意实数向量转换为概率分布：
    - 每个元素都在 [0,1] 范围内
    - 所有元素的和为 1
    - 使用数值稳定版本，减去最大值防止指数溢出
    
    参数:
        inputs: 输入数组，形状为 (m, n)
    
    返回:
        norm_values: 概率分布，形状为 (m, n)
    
    数学公式:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    max_values = np.max(inputs, axis=1, keepdims=True)    # 每行最大值，防止溢出
    slided_inputs = inputs - max_values                   # 数值稳定性处理
    exp_values = np.exp(slided_inputs)                    # 指数运算
    norm_base = np.sum(exp_values, axis=1, keepdims=True) # 归一化分母
    norm_values = exp_values/norm_base                    # 最终概率分布
    return norm_values

# =========================== 损失函数 ===========================

def precise_loss_function(predicted, real):
    """
    精确损失函数
    
    计算预测概率与真实标签之间的精确损失。
    使用连续的概率值进行计算，保留更多信息。
    
    参数:
        predicted: 预测概率，形状为 (m, 2)
        real: 真实标签，形状为 (m,)，值为0或1
    
    返回:
        损失值数组，形状为 (m,)
    
    计算过程:
        1. 将真实标签转换为one-hot编码
        2. 计算预测值与真实标签的点积
        3. 损失 = 1 - 点积（完美预测时损失为0）
    """
    # 创建one-hot编码矩阵
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real      # 正类标签
    real_matrix[:, 0] = 1 - real  # 负类标签
    
    # 计算预测与真实的点积
    product = np.sum(predicted * real_matrix, axis=1)
    
    return 1 - product  # 损失值：1减去点积

def loss_function(predicted, real):
    """
    二值化损失函数
    
    与精确损失函数类似，但在计算前先将预测概率二值化。
    主要用于评估硬分类的准确性。
    
    参数:
        predicted: 预测概率，形状为 (m, 2)
        real: 真实标签，形状为 (m,)，值为0或1
    
    返回:
        损失值数组，形状为 (m,)
    
    注意：虽然定义了二值化条件，但实际计算仍使用原始概率值
    """
    # 二值化条件（虽然定义了但实际未使用）
    condition = (predicted > 0.5)
    binary_predicted = np.where(condition, 1, 0)
    
    # 创建one-hot编码矩阵
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real      # 正类标签
    real_matrix[:, 0] = 1 - real  # 负类标签
    
    # 使用原始概率计算点积（未使用二值化结果）
    product = np.sum(predicted * real_matrix, axis=1)
    
    return 1 - product

def get_final_layer_preAct_damands(predicted_values, target_vector):
    """
    获取最终层激活前的需求值（demands）
    
    这是反向传播的起始点，计算输出层的误差信号。
    基于预测是否正确来决定梯度的强度和方向。
    
    参数:
        predicted_values: 预测概率值，形状为 (m, 2)
        target_vector: 真实标签，形状为 (m,)，值为0或1
    
    返回:
        target: 需求值矩阵，形状为 (m, 2)
               - 预测正确时：[0, 0]（无需调整）
               - 预测错误时：放大的误差向量，指导权重调整方向
    
    算法逻辑:
        1. 将真实标签转换为one-hot编码
        2. 计算预测值与真实标签的点积
        3. 如果点积 > 0.5：预测正确，返回零向量
        4. 如果点积 ≤ 0.5：预测错误，返回放大的误差信号
    
    误差放大公式:
        (target - 0.5) * 2  # 将 -0.5~0.5 的误差放大到 -1~1
    """
    # 创建one-hot编码的目标矩阵
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector      # 正类标签
    target[:, 0] = 1 - target_vector  # 负类标签
    
    # 逐样本处理，根据预测准确性生成需求信号
    for i in range(len(target_vector)):
        # 计算预测值与真实标签的点积，判断预测是否正确
        if np.dot(target[i], predicted_values[i]) > 0.5:
            # 预测正确：设置为零向量，无需调整权重
            target[i] = np.array([0, 0])
        else:
            # 预测错误：生成放大的误差信号
            # (target - 0.5) * 2 将误差从 [-0.5, 0.5] 放大到 [-1, 1]
            target[i] = (target[i] - 0.5) * 2
    
    return target

# =========================== 神经网络层类 ===========================

class Layer:
    """
    神经网络层类
    
    实现了单层神经网络的前向传播和反向传播功能。
    使用矩阵运算实现高效的批量计算。
    
    属性:
        weights: 权重矩阵，形状为 (n_inputs, n_neurons)
        biases: 偏置向量，形状为 (n_neurons,)
        output: 层的输出（在前向传播后设置）
    """
    
    def __init__(self, n_inputs, n_neurons):
        """
        初始化神经网络层
        
        参数:
            n_inputs: 输入特征的数量
            n_neurons: 该层神经元的数量
        
        权重初始化策略:
            使用标准正态分布随机初始化，这是一种简单但有效的初始化方法。
            在实际应用中，可以考虑使用Xavier或He初始化。
        """
        # 权重矩阵：每一列对应一个神经元的所有输入权重
        self.weights = np.random.randn(n_inputs, n_neurons)
        # 偏置向量：每个神经元一个偏置值
        self.biases = np.random.randn(n_neurons)
    
    def layer_forward(self, inputs):
        """
        层的前向传播
        
        计算线性变换：output = inputs * weights + biases
        
        参数:
            inputs: 输入数据，形状为 (batch_size, n_inputs)
        
        返回:
            output: 层的输出，形状为 (batch_size, n_neurons)
        
        数学公式:
            Z = X·W + b
            其中 X 是输入，W 是权重矩阵，b 是偏置向量
        """
        # 矩阵乘法 + 广播加法
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def layer_backward(self, preWeights_values, afterWeights_demands):
        """
        层的反向传播
        
        根据链式法则计算梯度，并将误差信号向前传播。
        同时计算当前层权重的调整矩阵。
        
        参数:
            preWeights_values: 前一层的输出值（即当前层的输入），形状为 (batch_size, n_inputs)
            afterWeights_demands: 后一层的需求值（即来自后层的梯度），形状为 (batch_size, n_neurons)
        
        返回:
            tuple: (标准化的前层需求, 标准化的权重调整矩阵)
        
        计算步骤:
            1. 计算传递给前一层的梯度信号
            2. 应用ReLU的导数（激活函数的梯度）
            3. 计算当前层权重的调整矩阵
            4. 对结果进行标准化处理
        
        数学原理:
            基于链式法则：dL/dx = dL/dy * dy/dx
            其中 y 是当前层输出，x 是当前层输入
        """
        # 步骤1：通过权重矩阵的转置将梯度传播到前一层
        # 矩阵乘法：(batch_size, n_neurons) × (n_neurons, n_inputs) → (batch_size, n_inputs)
        preWeights_damands = np.dot(afterWeights_demands, self.weights.T)
        
        # 步骤2：应用ReLU激活函数的导数
        # ReLU的导数：f'(x) = 1 if x > 0, else 0
        condition = (preWeights_values > 0)
        value_derivatives = np.where(condition, 1, 0)

        # 步骤3：将激活函数导数与传播的梯度相乘
        preActs_demands = value_derivatives * preWeights_damands
        norm_preActs_demands = normalize(preActs_demands)  # 标准化防止梯度爆炸
        
        # 步骤4：计算权重调整矩阵
        weight_adjust_matrix = self.get_weight_adjust_matrix(preWeights_values, afterWeights_demands)
        norm_weight_adjust_matrix = normalize(weight_adjust_matrix)  # 标准化权重调整
        
        return (norm_preActs_demands, norm_weight_adjust_matrix)
    
    def get_weight_adjust_matrix(self, preWeights_values, aftWeights_demands):
        """
        计算权重调整矩阵（权重梯度）
        
        基于反向传播的链式法则计算每个权重的梯度。
        使用外积的方式计算：前层输出 ⊗ 后层需求 = 权重梯度
        
        参数:
            preWeights_values: 前一层的输出值，形状为 (batch_size, n_inputs)
            aftWeights_demands: 后一层的需求值，形状为 (batch_size, n_neurons)
        
        返回:
            weights_adjust_matrix: 权重调整矩阵，形状为 (n_inputs, n_neurons)
        
        数学原理:
            权重梯度公式：∂Loss/∂W[i,j] = ∂Loss/∂output[j] × ∂output[j]/∂W[i,j]
            其中：
            - ∂Loss/∂output[j] = aftWeights_demands[j] (当前层第j个神经元的误差)
            - ∂output[j]/∂W[i,j] = preWeights_values[i] (前一层第i个神经元的输出)
            
        直觉理解:
            - 前一层输出越大 → 该权重对最终结果影响越大 → 梯度越大
            - 当前层需求越大 → 该神经元误差越大 → 对应权重需要更大调整
            - 所以：前一层输出 × 当前层需求 = 权重梯度
        """
        # 创建与权重矩阵同形状的全1矩阵和零矩阵
        plain_weights = np.full(self.weights.shape, 1)      # 辅助计算用的全1矩阵
        weights_adjust_matrix = np.full(self.weights.shape, 0.0)  # 累积梯度的零矩阵
        plain_weights_T = plain_weights.T                   # 转置，用于广播运算
        
        # 遍历批次中的每个样本，累积梯度
        for i in range(BATCH_SIZE):
            # 外积计算：前层输出[i,:] ⊗ 后层需求[i,:]
            # (plain_weights_T * preWeights_values[i,:]).T 创建了前层输出的列向量
            # 然后与后层需求的行向量相乘，得到外积矩阵
            sample_gradient = (plain_weights_T * preWeights_values[i, :]).T * aftWeights_demands[i, :]
            weights_adjust_matrix += sample_gradient
        
        # 计算批次平均梯度
        weights_adjust_matrix = weights_adjust_matrix / BATCH_SIZE
        return weights_adjust_matrix
        
# =========================== 神经网络类 ===========================

class Network:
    """
    多层神经网络类
    
    封装了完整的神经网络，包括前向传播、反向传播和训练功能。
    支持自适应学习策略和实时可视化。
    
    属性:
        shape: 网络形状列表，如 [2, 100, 200, 100, 50, 2]
        layers: 网络层列表，每个元素是一个Layer对象
    
    训练特性:
        - 自适应学习率
        - 强制训练模式（当改善率过低时）
        - 随机更新模式（当完全无改善时）
    """
    
    def __init__(self, network_shape):
        """
        初始化神经网络
        
        参数:
            network_shape: 网络形状列表，如 [2, 100, 200, 100, 50, 2]
                          表示输入层2个特征，隐藏层分别有100,200,100,50个神经元，输出层2个类别
        
        创建过程:
            根据网络形状自动创建相应数量的层，每相邻两层之间创建一个Layer对象。
        """
        self.shape = network_shape
        self.layers = []
        
        # 根据网络形状创建层
        # 如果有n个节点层，则需要n-1个连接层
        for i in range(len(network_shape) - 1):
            # 每一层的输入大小为前一个节点层的大小，输出大小为后一个节点层的大小
            layer = Layer(network_shape[i], network_shape[i + 1])
            self.layers.append(layer)
     
    def network_forward(self, inputs):
        """
        网络前向传播
        
        将输入数据通过整个网络，计算最终输出。
        同时保存每一层的输出，供反向传播使用。
        
        参数:
            inputs: 输入数据，形状为 (batch_size, input_features)
        
        返回:
            outputs: 列表，包含每一层的输出
                    outputs[0] = 原始输入
                    outputs[1] = 第1层输出(经过ReLU和标准化)
                    outputs[2] = 第2层输出(经过ReLU和标准化)
                    ...
                    outputs[-1] = 最后一层输出(经过Softmax)
        
        处理流程:
            1. 隐藏层：线性变换 → ReLU激活 → 标准化
            2. 输出层：线性变换 → Softmax激活
        """
        outputs = [inputs]  # 保存每一层的输出，从原始输入开始
        
        # 逐层进行前向传播
        for i in range(len(self.layers)):
            # 计算当前层的线性输出（激活前）
            layer_sum = self.layers[i].layer_forward(outputs[i])
            
            # 根据层的位置选择不同的激活函数
            if i < len(self.layers) - 1:
                # 隐藏层：使用ReLU激活函数
                layer_output = activation_ReLU(layer_sum)
                # 标准化处理，提高训练稳定性
                layer_output = normalize(layer_output)
            else:
                # 输出层：使用Softmax激活函数，输出概率分布
                layer_output = activation_softmax(layer_sum)
            
            # 保存当前层的输出
            outputs.append(layer_output)
        
        return outputs
    
    def network_backward(self, layer_outputs, target_vector):
        """
        网络反向传播
        
        基于输出误差，使用反向传播算法更新整个网络的权重和偏置。
        为了保证安全性，使用深拷贝创建备份网络进行更新。
        
        参数:
            layer_outputs: 前向传播的输出结果，包含每一层的输出
            target_vector: 真实标签向量，形状为 (batch_size,)
        
        返回:
            backup_network: 更新后的网络对象副本
        
        反向传播流程:
            1. 创建网络的深拷贝作为备份
            2. 计算输出层的初始错误信号
            3. 从输出层开始，逐层向前传播误差
            4. 更新每一层的权重和偏置
            5. 对更新结果进行标准化处理
        
        安全特性:
            使用深拷贝确保原网络不受影响，只有确认改善后才应用更新。
        """
        # 创建网络的深拷贝，避免直接修改原网络
        backup_network = copy.deepcopy(self)
        
        # 计算输出层的初始错误信号（反向传播的起点）
        preAct_demands = get_final_layer_preAct_damands(layer_outputs[-1], target_vector)
        
        # 从最后一层开始，逐层向前进行反向传播
        for i in range(len(self.layers)):
            # 获取当前处理的层（倒序遍历）
            layer = backup_network.layers[len(self.layers) - (1 + i)]
            
            # 更新偏置（跳过第一次迭代，因为输出层的偏置更新需要特殊处理）
            if i != 0:
                # 偏置的梯度是需求值的平均
                bias_gradient = np.mean(preAct_demands, axis=0)
                layer.biases += LEARNING_RATE * bias_gradient
                # 标准化偏置，防止数值不稳定
                layer.biases = vector_normalize(layer.biases)
            
            # 获取当前层的输入（即前一层的输出）
            outputs = layer_outputs[len(layer_outputs) - (2 + i)]
            
            # 计算当前层的反向传播
            results_list = layer.layer_backward(outputs, preAct_demands)
            preAct_demands = results_list[0]        # 传给前一层的需求值
            weights_adjust_matrix = results_list[1] # 当前层的权重调整矩阵
            
            # 更新权重
            layer.weights += LEARNING_RATE * weights_adjust_matrix
            # 标准化权重矩阵，防止梯度爆炸
            layer.weights = normalize(layer.weights)
        
        return backup_network
    
    #单批次训练
    def one_batch_train(self, batch):
        global force_train, random_train, n_improved, n_not_improved

        inputs = batch[:,(0, 1)]
        targets = copy.deepcopy(batch[:, 2]).astype(int) # 标准答案
        outputs = self.network_forward(inputs)
        precise_loss = precise_loss_function(outputs[-1], targets)
        loss = loss_function(outputs[-1], targets)
            
        if np.mean(loss) <= LOSS_THRESHOLD:#损失函数小于这个值就不需要训练了
            print('No need for training')
        else:
            backup_network = self.network_backward(outputs, targets)
            backup_outputs = backup_network.network_forward(inputs)
            backup_precise_loss = precise_loss_function(backup_outputs[-1], targets)
            backup_loss = loss_function(backup_outputs[-1], targets)
            
            if np.mean(precise_loss) >= np.mean(backup_precise_loss) or np.mean(loss) >= np.mean(backup_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                print('Improved')
                n_improved += 1
            
            else:
                if force_train:
                    for i in range(len(self.layers)):
                        self.layers[i].weights = backup_network.layers[i].weights.copy()
                        self.layers[i].biases = backup_network.layers[i].biases.copy()
                    print('Force train')
                if random_train:
                    self.random_update()
                    print("Random update")
                else:  
                    print('No improvement')
                n_not_improved += 1
        print('-----------------------------------------')
            
    #多批次训练
    def train(self, n_entries):
        global force_train, random_train, n_improved, n_not_improved
        n_improved = 0
        n_not_improved = 0

        n_batches = math.ceil(n_entries/BATCH_SIZE)
        for i in range(n_batches):
            batch = cp.creat_data(BATCH_SIZE)
            self.one_batch_train(batch)
        improvement_rate =  n_improved/(n_improved + n_not_improved)
        print("Improvement rate")
        print(format(improvement_rate, ".0%"))
        if improvement_rate <= FORCE_TRAIN_THRESHOLD:
            force_train = True
        else:
            force_train = False
        if n_improved == 0:
            random_train = True
        else:
            random_train = False
        
        data = cp.creat_data(800)
        inputs = data[:, (0, 1)]
        outputs = self.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "After training")
        
    #随机更新
    def random_update(self):
        random_network = Network(NETWORK_SHAPE)
        for i in range(len(self.layers)):
            weights_change = random_network.layers[i].weights
            biases_change = random_network.layers[i].biases
            self.layers[i].weights += weights_change
            self.layers[i].biases += biases_change
            
    #核验LOSS值
    def check_loss(self):
        data = cp.creat_data(1000)
        inputs = data[:, (0, 1)]
        targets = copy.deepcopy(data[:, 2]).astype(int) # 标准答案
        outputs = self.network_forward(inputs)
        loss = loss_function(outputs[-1], targets)
        return np.mean(loss)
        
#-------------MAIN-------------------------
def main():
    global current_loss
    data = cp.creat_data(800) #生成数据
    cp.plot_data(data, "Right classification")

    #选择起始网络
    use_this_network = 'n' #No
    while use_this_network != 'Y' and use_this_network != 'y':
        network = Network(NETWORK_SHAPE)
        inputs = data[:, (0, 1)]
        outputs = network.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "Choose network")
        use_this_network = input("Use this network? Y to yes, N to No \n")
    
    #进行训练
    do_train = input("Train? Y to yes, N to No \n")
    while do_train == 'Y' or do_train == 'y' or do_train.isnumeric() == True:
        if do_train.isnumeric() == True:
            n_entries = int(do_train)
        else:
            n_entries = int(input("Enter the number of data entries used to train. \n"))
            
        network.train(n_entries)
        do_train = input("Train? Y to yes, N to No \n")
        
    #演示训练效果
    inputs = data[:, (0, 1)]
    outputs = network.network_forward(inputs)
    classification = classify(outputs[-1])
    data[:, 2] = classification
    cp.plot_data(data, "After training")
    print("谢谢，再见！")
#----------------TEST-------------------------
def test():
    pass

#--------------运行---------------------
main()