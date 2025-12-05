from .device_manager import device_manager
from abc import ABC, abstractmethod
import cupy as cp


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 卷积核的高
    filter_w : 卷积核的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    # 输入数据的形状
    # N：批数目，C：通道数，H：输入数据高，W：输入数据长
    xp = device_manager.get_xp()
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 输出数据的高
    out_w = (W + 2 * pad - filter_w) // stride + 1  # 输出数据的长
    # 填充 H,W
    img = xp.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # (N, C, filter_h, filter_w, out_h, out_w)的0矩阵
    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # 按(0, 4, 5, 1, 2, 3)顺序，交换col的列，然后改变形状
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    xp = device_manager.get_xp()
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = xp.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Layer(ABC):
    """所有层的基类，定义统一接口"""

    def save_params(self, layer_index):
        """
        保存层参数到文件
        参数:
            filepath: 文件路径
            layer_index: 层在网络中的索引
        """
        # 默认实现：无参数层不需要保存
        params_dict = {}
        # 检查该层是否有可训练参数
        if hasattr(self, 'W') and hasattr(self, 'b'):
            if self.W is not None and self.b is not None:
                # 保存层类型标识
                params_dict[f'layer{layer_index}_type'] = self.__class__.__name__

                # 转换GPU数组到CPU numpy数组
                if hasattr(self.W, 'device'):  # CuPy数组
                    params_dict[f'layer{layer_index}_W'] = cp.asnumpy(self.W)
                    params_dict[f'layer{layer_index}_b'] = cp.asnumpy(self.b)
                else:  # NumPy数组
                    params_dict[f'layer{layer_index}_W'] = self.W
                    params_dict[f'layer{layer_index}_b'] = self.b
        return params_dict

    def load_params(self, data, layer_index):
        """
        从数据中加载参数
        参数:
            data: 加载的数据字典
            layer_index: 层索引
        """
        has_weights = (hasattr(self, 'W') and hasattr(self, 'b') and
                       self.W is not None and self.b is not None)

        if not has_weights:
            # 无参数层，静默返回成功
            return True
        weight_key = f'layer{layer_index}_W'
        bias_key = f'layer{layer_index}_b'

        if weight_key in data and bias_key in data:
            saved_weight_shape = data[weight_key].shape
            saved_bias_shape = data[bias_key].shape
            current_weight_shape = self.W.shape
            current_bias_shape = self.b.shape
            if saved_weight_shape != current_weight_shape:
                print(f"错误：第{layer_index}层权重形状不匹配")
                print(f"  保存的权重形状: {saved_weight_shape}")
                print(f"  当前的权重形状: {current_weight_shape}")
                return False

                # 偏置形状验证
            if saved_bias_shape != current_bias_shape:
                print(f"错误：第{layer_index}层偏置形状不匹配")
                print(f"  保存的偏置形状: {saved_bias_shape}")
                print(f"  当前的偏置形状: {current_bias_shape}")
                return False
            self.W = device_manager.to_device(data[weight_key])
            self.b = device_manager.to_device(data[bias_key])
            return True

        print(f"警告：第{layer_index}层在保存文件中找不到参数")
        return False

    def get_config(self):
        """返回层的配置信息，用于summary和验证"""
        return {
            'type': self.__class__.__name__,
            'has_weights': hasattr(self, 'W') and hasattr(self, 'b')
        }

    def get_params_count(self):
        """返回层的参数数量"""
        count = 0
        if hasattr(self, 'W') and self.W is not None:
            count += self.W.size
        if hasattr(self, 'b') and self.b is not None:
            count += self.b.size
        return count

    @abstractmethod
    def forward(self, x):
        """前向传播"""
        pass

    @abstractmethod
    def backward(self, delta, learning_rate, loss_name):
        """反向传播"""
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, input_dim, output_dim):
        xp = device_manager.get_xp()
        # 初始化参数 (Xavier初始化)，你要随机也可以，但效率不高，可以试试
        self.W = xp.random.randn(input_dim, output_dim) * xp.sqrt(2.0 / (input_dim + output_dim))
        self.b = xp.zeros((1, output_dim))
        self.Y_prev = None  # 前一层激活值（输入）
        self.Y = None  # 激活输出

        # 保存构造参数用于序列化
        self.input_dim = input_dim
        self.output_dim = output_dim

    def get_config(self):
        """返回层的详细配置信息"""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'params_count': self.get_params_count()
        })
        return config

    def forward(self, Y_prev):
        xp = device_manager.get_xp()
        self.Y_prev = Y_prev
        self.Y = xp.dot(Y_prev, self.W) + self.b

        return self.Y  # 传给下一层当输入

    def backward(self, delta_curr, learning_rate):
        xp = device_manager.get_xp()
        m = self.Y_prev.shape[0]  # 样本数

        delta = delta_curr

        # 计算权重和偏置的梯度
        dW = xp.dot(self.Y_prev.T, delta) / m  # 平均梯度
        db = xp.sum(delta, axis=0, keepdims=True) / m

        # 计算传递给前一层的梯度
        delta_prev = xp.dot(delta, self.W.T)

        # 更新参数
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return delta_prev


class Conv2DLayer(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu'):
        """初始化卷积层"""
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        xp = device_manager.get_xp()
        k_h, k_w = kernel_size

        # Xavier初始化权重
        fan_in = in_channels * k_h * k_w
        fan_out = out_channels * k_h * k_w
        self.W = xp.random.randn(out_channels, in_channels, k_h, k_w) * xp.sqrt(2.0 / (fan_in + fan_out))
        self.b = xp.zeros(out_channels)

        # 前向传播中间变量缓存
        self.x = None  # 输入数据
        self.col = None  # im2col后的矩阵
        self.col_W = None  # 权重展开矩阵
        self.conv_output = None  # 卷积输出（ReLU前）

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'params_count': self.get_params_count()
        })
        return config

    def forward(self, x):
        """前向传播（使用im2col优化）"""
        xp = device_manager.get_xp()
        batch_size, in_channels, h, w = x.shape
        k_h, k_w = self.kernel_size

        # 计算输出尺寸
        out_h = (h + 2 * self.padding - k_h) // self.stride + 1
        out_w = (w + 2 * self.padding - k_w) // self.stride + 1

        # 使用im2col展开输入数据
        self.col = im2col(x, k_h, k_w, self.stride, self.padding)

        # 展开权重矩阵
        self.col_W = self.W.reshape(self.out_channels, -1).T  # shape: (in_channels*k_h*k_w, out_channels)

        # 计算卷积：矩阵乘法
        self.conv_output = xp.dot(self.col, self.col_W) + self.b  # shape: (batch_size*out_h*out_w, out_channels)

        # 重塑为输出格式
        conv_output_reshaped = self.conv_output.reshape(batch_size, out_h, out_w, self.out_channels)
        conv_output_reshaped = conv_output_reshaped.transpose(0, 3, 1,
                                                              2)  # shape: (batch_size, out_channels, out_h, out_w)

        # 缓存用于反向传播
        self.x = x
        self.conv_output = conv_output_reshaped  # 保存ReLU前的输出

        return conv_output_reshaped

    def backward(self, d_out, learning_rate):
        """
        反向传播（使用col2im优化）
        参数:
        - d_out: 上一层传来的梯度，形状为 (batch_size, out_channels, out_h, out_w)
        - learning_rate: 学习率
        返回:
        - d_x: 传递给前一层的梯度，形状与输入x相同
        """
        xp = device_manager.get_xp()
        batch_size = d_out.shape[0]


        # 重塑梯度为二维矩阵: (batch_size*out_h*out_w, out_channels)
        d_2d = d_out.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # 计算偏置梯度：对每个输出通道的梯度求和
        d_bias = xp.sum(d_2d, axis=0) / batch_size

        # 计算权重梯度：矩阵乘法
        d_weights_2d = xp.dot(self.col.T, d_2d) / batch_size  # shape: (in_channels*k_h*k_w, out_channels)

        # 重塑权重梯度回四维
        k_h, k_w = self.kernel_size
        d_weights = d_weights_2d.T.reshape(self.out_channels, self.in_channels, k_h, k_w)

        # 计算输入梯度：传播到前一层的梯度
        d_col = xp.dot(d_2d, self.col_W.T)  # shape: (batch_size*out_h*out_w, in_channels*k_h*k_w)

        # 使用col2im将梯度还原为输入形状
        d_x = col2im(d_col, self.x.shape, k_h, k_w, self.stride, self.padding)

        # 更新参数
        self.W -= learning_rate * d_weights
        self.b -= learning_rate * d_bias

        return d_x


class MaxPool2D(Layer):
    """
    最大池化层 - 使用im2col/col2im优化
    前向传播：取每个池化窗口内的最大值
    反向传播：梯度只传递到前向传播中的最大值位置
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        参数:
        - kernel_size: 池化窗口大小，整数或元组 (k_h, k_w)
        - stride: 步长，默认为kernel_size
        - padding: 填充大小
        """
        super().__init__()
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.padding = padding
        self.cache = {}  # 用于缓存中间结果，如最大值位置

    def forward(self, x):
        """
        前向传播 - 最大池化
        """
        xp = device_manager.get_xp()
        batch_size, channels, h, w = x.shape
        k_h, k_w = self.kernel_size

        # 1. 计算输出尺寸
        out_h = (h + 2 * self.padding - k_h) // self.stride + 1
        out_w = (w + 2 * self.padding - k_w) // self.stride + 1

        k_h, k_w = self.kernel_size
        # 2. 使用im2col将输入数据展开为二维矩阵
        # col的形状为 (batch_size * out_h * out_w, channels * k_h * k_w)
        col = im2col(x, k_h, k_w, self.stride, self.padding)

        # 3. 重塑col，使每个池化窗口在一个独立的行中
        # 重塑后形状: (batch_size * out_h * out_w * channels, k_h * k_w)
        col_reshaped = col.reshape(-1, k_h * k_w)
        max_indices = xp.argmax(col_reshaped, axis=1)
        output = col_reshaped[xp.arange(len(max_indices)), max_indices]
        output = output.reshape(batch_size, out_h, out_w, channels).transpose(0, 3, 1, 2)

        # 6. 缓存反向传播所需信息
        self.cache = {
            'x_shape': x.shape,
            'col_shape': col.shape,
            'max_indices': max_indices,
        }

        return output

    def backward(self, d_out, learning_rate=None, loss_name=None):
        """
        反向传播 - 最大池化
        只将梯度传递到前向传播中最大值所在的位置
        """
        xp = device_manager.get_xp()
        x_shape = self.cache['x_shape']
        max_indices = self.cache['max_indices']
        batch_size, channels, h, w = x_shape
        k_h, k_w = self.kernel_size
        out_h, out_w = d_out.shape[2], d_out.shape[3]

        # 1. 将上游梯度重塑为二维形式
        d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1)

        # 2. 初始化一个全零矩阵，形状与im2col展开后的输入相同
        d_col = xp.zeros((batch_size * out_h * out_w * channels, k_h * k_w), dtype=d_out.dtype)

        # 最大池化：梯度只传递给前向传播中最大值所在的位置
        d_col[xp.arange(len(max_indices)), max_indices] = d_out_flat
        # 4. 重塑梯度以匹配im2col的输出格式
        d_col = d_col.reshape(self.cache['col_shape'])
        # 5. 使用col2im将梯度还原为原始输入形状
        d_x = col2im(d_col, x_shape, k_h, k_w, self.stride, self.padding)

        return d_x

    def get_config(self):
        """返回层配置信息"""
        return {
            'type': 'MaxPool2D',
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'params_count': 0  # 池化层无参数
        }


class AvgPool2D(Layer):
    """
    平均池化层 - 使用im2col/col2im优化
    前向传播：计算每个池化窗口内的平均值
    反向传播：梯度平均分配到窗口中的每个位置
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        参数:
        - kernel_size: 池化窗口大小，整数或元组 (k_h, k_w)
        - stride: 步长，默认为kernel_size
        - padding: 填充大小
        """
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.padding = padding
        self.cache = {}  # 用于缓存中间结果，如最大值位置

    def forward(self, x):
        """
        前向传播 - 平均池化
        """
        xp = device_manager.get_xp()
        batch_size, channels, h, w = x.shape
        k_h, k_w = self.kernel_size

        # 1. 计算输出尺寸
        out_h = (h + 2 * self.padding - k_h) // self.stride + 1
        out_w = (w + 2 * self.padding - k_w) // self.stride + 1

        k_h, k_w = self.kernel_size
        # 2. 使用im2col将输入数据展开为二维矩阵
        # col的形状为 (batch_size * out_h * out_w, channels * k_h * k_w)
        col = im2col(x, k_h, k_w, self.stride, self.padding)

        # 3. 重塑col，使每个池化窗口在一个独立的行中
        # 重塑后形状: (batch_size * out_h * out_w * channels, k_h * k_w)
        col_reshaped = col.reshape(-1, k_h * k_w)

        # 4. 根据池化模式进行计算

        output = xp.mean(col_reshaped, axis=1)


        # 5. 将输出重塑为4D张量 (batch_size, channels, out_h, out_w)
        output = output.reshape(batch_size, out_h, out_w, channels).transpose(0, 3, 1, 2)

        # 6. 缓存反向传播所需信息
        self.cache = {
            'x_shape': x.shape,
            'col_shape': col.shape,
            'input_col': col_reshaped   # 平均池化需要原始数据
        }

        return output

    def backward(self, d_out, learning_rate=None, loss_name=None):
        """
        反向传播 - 平均池化
        将梯度平均分配到池化窗口中的每个位置
        """
        xp = device_manager.get_xp()
        x_shape = self.cache['x_shape']
        batch_size, channels, h, w = x_shape
        k_h, k_w = self.kernel_size
        out_h, out_w = d_out.shape[2], d_out.shape[3]

        # 1. 将上游梯度重塑为二维形式
        d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1)

        # 2. 初始化一个全零矩阵，形状与im2col展开后的输入相同
        d_col = xp.zeros((batch_size * out_h * out_w * channels, k_h * k_w), dtype=d_out.dtype)

        # 3. 根据池化模式分配梯度
        avg_grad = d_out_flat.reshape(-1, 1) / (k_h * k_w)
        d_col[:] = avg_grad  # 将平均梯度分配到整个窗口

        # 4. 重塑梯度以匹配im2col的输出格式
        d_col = d_col.reshape(self.cache['col_shape'])

        # 5. 使用col2im将梯度还原为原始输入形状
        d_x = col2im(d_col, x_shape, k_h, k_w, self.stride, self.padding)

        return d_x

    def get_config(self):
        """返回层配置信息"""
        return {
            'type': 'AvgPool2D',
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'params_count': 0  # 池化层无参数
        }


class Flatten(Layer):
    """
    自定义展平层
    前向传播：将输入张量展平
    反向传播：将梯度重新恢复为输入张量的形状
    """

    def __init__(self):
        # 用于在反向传播中记录输入形状
        self.original_shape = None

    def get_config(self):
        config = super().get_config()
        config['params_count'] = 0
        return config

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状为 (batch_size, 通道数, 高度, 宽度) 或更高维
        返回:
            展平后的张量，形状为 (batch_size, -1)
        """
        # 记录输入形状，反向传播时需要使用
        self.original_shape = x.shape

        # 计算展平后的特征数
        # 例如，输入形状为(2, 3, 4, 4)：2 * 3 * 4 * 4 = 96
        # 输出形状即为 (2, 96)
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, d_out, learning_rate=None):
        """
        反向传播
        参数:
            d_out: 上游传回的梯度，形状为 (batch_size, flattened_size)
        返回:
            梯度 d_x，形状与原始输入 self.original_shape 相同
        """
        # 核心操作：将梯度重塑回前向传播输入时的形状
        return d_out.reshape(self.original_shape)


class Dropout(Layer):
    """
    Dropout层实现
    前向传播：随机失活部分神经元
    反向传播：只传递被保留神经元的梯度
    """

    def __init__(self, dropout_rate=0.5,seed=40):
        """
        参数:
        - dropout_rate: 神经元被丢弃的概率，通常设为0.2-0.5
        """
        self.dropout_rate = dropout_rate
        self.keep_prob = 1 - dropout_rate  # 神经元保留概率
        self.epoch=0
        self.seed=seed
        self.mask = None  # 用于记录哪些神经元被保留
        self.training = True  # 训练/预测模式标志

    def get_config(self):
        config = super().get_config()
        config['params_count'] = 0
        return config

    def forward(self, x):
        """
        前向传播
        参数:
        - x: 输入数据
        返回:
        - 经过Dropout处理后的输出
        """
        if not self.training or self.keep_prob == 1:
            # 预测模式或keep_prob=1时，直接返回输入
            return x
        xp = device_manager.get_xp()
        dynamic_seed = self.epoch+self.seed
        self.epoch+=1
        rng = xp.random.default_rng(dynamic_seed)
        self.mask = (rng.random(x.shape) < self.keep_prob).astype(xp.float32)
        # 应用掩码并缩放，保持期望值不变
        output = x * self.mask
        output /= self.keep_prob  # 重要：缩放以保持期望值

        return output

    def backward(self, d_out, learning_rate=None):
        """
        反向传播
        参数:
        - d_out: 上游传来的梯度
        返回:
        - 传递给前一层的梯度
        """
        if not self.training or self.keep_prob == 1:
            # 预测模式，梯度直接传递
            return d_out

        # 只传递被保留神经元的梯度
        d_x = d_out * self.mask
        d_x /= self.keep_prob  # 同样需要缩放

        return d_x

    def train(self):
        """设置为训练模式"""
        self.training = True

    def eval(self):
        """设置为预测/评估模式"""
        self.training = False


class BatchNormalization(Layer):
    """
    批量归一化层
    使用基类的 W 作为缩放参数 gamma，b 作为偏移参数 beta
    """

    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        """
        参数:
        - num_features: 特征数量（全连接层是神经元数，卷积层是通道数）
        - momentum: 移动平均的动量参数，用于更新全局统计量
        - epsilon: 防止除零错误的小常数
        """
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.training = True  # 训练/预测模式标志

        # 可训练参数：缩放gamma (W) 和 偏移beta (b)
        # 初始化：gamma为1，beta为0，这样初始状态相当于恒等变换
        xp = device_manager.get_xp()
        self.W = xp.ones(num_features)  # gamma
        self.b = xp.zeros(num_features)  # beta

        # 非训练参数：移动平均的均值和方差（用于预测模式）
        self.running_mean = xp.zeros(num_features)
        self.running_var = xp.ones(num_features)  # 初始为1，避免初始除零

        # 缓存：用于反向传播
        self.cache = None

    def forward(self, x):
        """
        前向传播
        参数:
        - x: 输入数据，形状取决于网络层类型：
            - 全连接层: (batch_size, num_features)
            - 卷积层: (batch_size, channels, height, width)
        """
        xp = device_manager.get_xp()

        if self.training:
            return self._forward_train(x, xp)
        else:
            return self._forward_test(x, xp)

    def _forward_train(self, x, xp):
        """训练模式前向传播"""
        # 根据输入维度计算统计量
        if len(x.shape) == 2:  # 全连接层 [N, D]
            axis = 0  # 沿批次维度计算
        elif len(x.shape) == 4:  # 卷积层 [N, C, H, W]
            # 在通道维度上计算每个通道的均值和方差
            axis = (0, 2, 3)  # 对批次、高度、宽度求平均
        else:
            raise ValueError(f"不支持的输入维度: {len(x.shape)}")

        # 计算当前批次的均值和方差
        batch_mean = xp.mean(x, axis=axis, keepdims=True)
        batch_var = xp.var(x, axis=axis, keepdims=True)

        # 更新移动平均（用于预测模式）
        self.running_mean = (self.momentum * self.running_mean +
                             (1 - self.momentum) * batch_mean.reshape(-1))
        self.running_var = (self.momentum * self.running_var +
                            (1 - self.momentum) * batch_var.reshape(-1))

        # 归一化
        x_hat = (x - batch_mean) / xp.sqrt(batch_var + self.epsilon)

        # 缩放和平移 (使用 W 作为 gamma, b 作为 beta)
        # 需要将W和b重塑为适合广播的形状
        if len(x.shape) == 4:
            # 卷积层: 重塑为 (1, C, 1, 1) 以支持广播
            gamma = self.W.reshape(1, -1, 1, 1)
            beta = self.b.reshape(1, -1, 1, 1)
        else:
            # 全连接层: 保持 (1, D) 或 (D,)
            gamma = self.W.reshape(1, -1)
            beta = self.b.reshape(1, -1)

        out = gamma * x_hat + beta

        # 缓存中间结果用于反向传播
        self.cache = {
            'x': x,
            'x_hat': x_hat,
            'batch_mean': batch_mean,
            'batch_var': batch_var,
            'gamma': gamma  # 缓存广播后的gamma
        }

        return out

    def _forward_test(self, x, xp):
        """预测模式前向传播：使用移动平均的统计量"""
        # 准备移动统计量的形状以支持广播
        if len(x.shape) == 4:  # 卷积层
            running_mean = self.running_mean.reshape(1, -1, 1, 1)
            running_var = self.running_var.reshape(1, -1, 1, 1)
            gamma = self.W.reshape(1, -1, 1, 1)
            beta = self.b.reshape(1, -1, 1, 1)
        else:  # 全连接层
            running_mean = self.running_mean.reshape(1, -1)
            running_var = self.running_var.reshape(1, -1)
            gamma = self.W.reshape(1, -1)
            beta = self.b.reshape(1, -1)

        # 使用移动统计量进行归一化
        x_hat = (x - running_mean) / xp.sqrt(running_var + self.epsilon)
        out = gamma * x_hat + beta

        return out

    def backward(self, dout, learning_rate):
        """优化版本的反向传播"""
        if not self.training:
            return dout

        xp = device_manager.get_xp()
        x, x_hat, batch_mean, batch_var, gamma = (self.cache['x'], self.cache['x_hat'],
                                                  self.cache['batch_mean'], self.cache['batch_var'],
                                                  self.cache['gamma'])

        N = x.shape[0]
        if len(x.shape) == 4:
            N *= x.shape[2] * x.shape[3]

        # 1. 参数梯度
        dgamma = xp.sum(dout * x_hat, axis=(0, 2, 3) if len(x.shape) == 4 else 0)
        dbeta = xp.sum(dout, axis=(0, 2, 3) if len(x.shape) == 4 else 0)

        # 2. 简化版输入梯度计算[5](@ref)
        dx_hat = dout * gamma
        sqrt_var = xp.sqrt(batch_var + self.epsilon)

        # 重塑统计量以支持广播
        if len(x.shape) == 4:
            batch_mean = batch_mean.reshape(1, -1, 1, 1)
            sqrt_var = sqrt_var.reshape(1, -1, 1, 1)
        else:
            batch_mean = batch_mean.reshape(1, -1)
            sqrt_var = sqrt_var.reshape(1, -1)

        # 更简洁的梯度计算
        dx = (1.0 / (N * sqrt_var)) * (
                N * dx_hat -
                xp.sum(dx_hat, axis=(0, 2, 3) if len(x.shape) == 4 else 0, keepdims=True) -
                x_hat * xp.sum(dx_hat * x_hat, axis=(0, 2, 3) if len(x.shape) == 4 else 0, keepdims=True)
        )

        # 更新参数
        self.W -= learning_rate * dgamma
        self.b -= learning_rate * dbeta

        return dx

    def save_params(self, layer_index):
        """
        扩展保存方法，包含移动平均统计量
        """
        params_dict = super().save_params(layer_index)  # 保存W和b (gamma和beta)

        # 添加移动平均统计量
        xp = device_manager.get_xp()
        if hasattr(self.running_mean, 'device'):  # CuPy数组
            params_dict[f'layer{layer_index}_running_mean'] = xp.asnumpy(self.running_mean)
            params_dict[f'layer{layer_index}_running_var'] = xp.asnumpy(self.running_var)
        else:
            params_dict[f'layer{layer_index}_running_mean'] = self.running_mean
            params_dict[f'layer{layer_index}_running_var'] = self.running_var

        params_dict[f'layer{layer_index}_num_features'] = self.num_features
        params_dict[f'layer{layer_index}_momentum'] = self.momentum
        params_dict[f'layer{layer_index}_epsilon'] = self.epsilon

        return params_dict

    def load_params(self, data, layer_index):
        """
        扩展加载方法，包含移动平均统计量
        """
        # 先加载可训练参数 W 和 b (gamma 和 beta)
        success = super().load_params(data, layer_index)

        if not success:
            return False

        # 加载移动平均统计量和其他配置
        running_mean_key = f'layer{layer_index}_running_mean'
        running_var_key = f'layer{layer_index}_running_var'

        if running_mean_key in data and running_var_key in data:
            self.running_mean = device_manager.to_device(data[running_mean_key])
            self.running_var = device_manager.to_device(data[running_var_key])

        # 加载其他超参数（可选，提供向后兼容性）
        num_features_key = f'layer{layer_index}_num_features'
        if num_features_key in data:
            self.num_features = data[num_features_key]

        return True

    def get_config(self):
        """返回层的配置信息"""
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'has_running_stats': True,
            'params_count': self.get_params_count()
        })
        return config

    def train(self):
        """设置为训练模式"""
        self.training = True

    def eval(self):
        """设置为评估/预测模式"""
        self.training = False