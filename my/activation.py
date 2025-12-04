from .layers import Layer
from .device_manager import device_manager


class ReLU(Layer):
    def __init__(self):
        # 缓存输入，用于反向传播
        self.input = None

    def forward(self, x):
        """ReLU前向传播：f(x) = max(0, x)"""
        xp = device_manager.get_xp()
        self.input = x  # 缓存输入，反向传播时需要
        return xp.maximum(0, x)  # ReLU激活函数

    def backward(self, delta, learning_rate=None):
        """
        ReLU反向传播：梯度 = 上游梯度 * ReLU导数
        ReLU导数为：输入>0时为1，输入<=0时为0
        """
        # ReLU的导数：当输入>0时为1，否则为0
        xp = device_manager.get_xp()
        relu_grad = (self.input > 0).astype(xp.float32)

        # 链式法则：将上游梯度乘以ReLU的局部梯度
        d_input = delta * relu_grad

        return d_input  # 传递给前一层

    def get_config(self):
        """返回ReLU层的配置信息"""
        config = super().get_config()
        config.update({
            'activation': 'relu',
            'params_count': 0
        })
        return config


class Sigmoid(Layer):
    def __init__(self):
        # 缓存输出，用于反向传播
        self.output = None

    def forward(self, x):
        """Sigmoid前向传播：f(x) = 1 / (1 + exp(-x))"""
        xp = device_manager.get_xp()

        # 数值稳定版：对负数输入进行计算，防止正数过大时exp溢出
        # 当x >= 0时，计算 1 / (1 + exp(-x))
        # 当x < 0时，计算 exp(x) / (1 + exp(x))
        positive_mask = (x >= 0)
        negative_mask = ~positive_mask  # 取反得到负数部分的掩码

        # 初始化输出数组
        output = xp.zeros_like(x)

        # 处理正数部分
        exp_negative = xp.exp(-x[positive_mask])
        output[positive_mask] = 1 / (1 + exp_negative)

        # 处理负数部分（更稳定）
        exp_positive = xp.exp(x[negative_mask])
        output[negative_mask] = exp_positive / (1 + exp_positive)

        self.output = output  # 缓存输出，用于反向传播
        return output

    def backward(self, delta, learning_rate=None):
        """
        Sigmoid反向传播：梯度 = 上游梯度 * Sigmoid导数
        Sigmoid导数为：f'(x) = f(x) * (1 - f(x))
        """
        # 使用缓存的输出计算导数
        sigmoid_derivative = self.output * (1 - self.output)

        # 链式法则
        d_input = delta * sigmoid_derivative
        return d_input

    def get_config(self):
        """返回Sigmoid层的配置信息"""
        config = super().get_config()
        config.update({
            'activation': 'sigmoid',
            'params_count': 0  # Sigmoid层无参数
        })
        return config


class Softmax(Layer):
    def __init__(self, axis=-1):
        """
        参数:
        - axis: 沿着哪个维度计算Softmax，默认为最后一个维度（通常是类别维度）
        """
        self.axis = axis
        # 缓存输出，用于反向传播
        self.output = None

    def forward(self, x):
        """Softmax前向传播（数值稳定版本）"""
        xp = device_manager.get_xp()

        # 数值稳定性处理：减去最大值防止指数运算溢出[7,8](@ref)}
        x_max = xp.max(x, axis=self.axis, keepdims=True)
        exp_x = xp.exp(x - x_max)  # 减去最大值后再计算指数

        # 计算归一化因子（每个样本的指数和）
        sum_exp = xp.sum(exp_x, axis=self.axis, keepdims=True)

        # 计算Softmax输出
        output = exp_x / sum_exp
        self.output = output  # 缓存输出，用于反向传播

        return output

    def backward(self, delta, learning_rate=None):
        return delta

    def get_config(self):
        """返回Softmax层的配置信息"""
        config = super().get_config()
        config.update({
            'activation': 'softmax',
            'axis': self.axis,
            'params_count': 0  # Softmax层无参数
        })
        return config
