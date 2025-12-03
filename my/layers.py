from .device_manager import device_manager
from abc import ABC, abstractmethod


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

    def save_params(self, filepath, layer_index):
        """
        保存层参数到文件
        参数:
            filepath: 文件路径
            layer_index: 层在网络中的索引
        """
        # 默认实现：无参数层不需要保存
        pass

    def load_params(self, data, layer_index):
        """
        从数据中加载参数
        参数:
            data: 加载的数据字典
            layer_index: 层索引
        """
        # 默认实现：无参数层不需要加载
        pass

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
    def __init__(self, input_dim, output_dim, activation='sigmoid'):
        xp = device_manager.get_xp()
        # 初始化参数 (Xavier初始化)，你要随机也可以，但效率不高，可以试试
        self.W = xp.random.randn(input_dim, output_dim) * xp.sqrt(2.0 / (input_dim + output_dim))
        self.b = xp.zeros((1, output_dim))
        self.activation = activation
        self.Y_prev = None  # 前一层激活值（输入）
        self.Z = None  # 加权和
        self.Y = None  # 激活输出
        self.is_output_layer = False

        # 保存构造参数用于序列化
        self.input_dim = input_dim
        self.output_dim = output_dim

    def save_params(self, filepath, layer_index):
        """保存全连接层的权重和偏置"""
        xp = device_manager.get_xp()

        # 准备参数数据
        params_dict = {}

        # 转换GPU数组到CPU numpy数组
        if hasattr(self.W, 'device'):  # CuPy数组
            params_dict[f'layer{layer_index}_W'] = xp.asnumpy(self.W)
            params_dict[f'layer{layer_index}_b'] = xp.asnumpy(self.b)
        else:  # NumPy数组
            params_dict[f'layer{layer_index}_W'] = self.W
            params_dict[f'layer{layer_index}_b'] = self.b

        # 保存配置信息
        params_dict[f'layer{layer_index}_activation'] = self.activation

        return params_dict

    def load_params(self, data, layer_index):
        """加载全连接层的权重和偏置"""
        xp = device_manager.get_xp()

        # 检查参数是否存在
        weight_key = f'layer{layer_index}_W'
        bias_key = f'layer{layer_index}_b'

        if weight_key in data and bias_key in data:

            # 验证参数形状
            expected_shape = (self.input_dim, self.output_dim)
            if self.W.shape != expected_shape:
                print(f"警告：第{layer_index}层权重形状不匹配（期望：{expected_shape}，实际：{self.W.shape}）")
                return False

            # 检查激活函数一致性
            activation_key = f'layer{layer_index}_activation'
            if activation_key in data:
                saved_activation = data[activation_key].item() if hasattr(data[activation_key], 'item') else data[
                    activation_key]
                if saved_activation != self.activation:
                    print(f"警告：第{layer_index}层激活函数不匹配（当前：{self.activation}，保存的：{saved_activation}）")
            self.W = device_manager.to_device(data[weight_key])
            self.b = device_manager.to_device(data[bias_key])
            return True
        else:
            print(f"警告：第{layer_index}层在保存文件中找不到参数")
            return False

    def get_config(self):
        """返回层的详细配置信息"""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'activation': self.activation,
            'params_count': self.get_params_count()
        })
        return config

    def forward(self, Y_prev):
        xp = device_manager.get_xp()
        self.Y_prev = Y_prev
        self.Z = xp.dot(Y_prev, self.W) + self.b

        # 激活函数
        if self.activation == 'sigmoid':
            self.Y = 1 / (1 + xp.exp(-self.Z))
        elif self.activation == 'relu':
            self.Y = xp.maximum(0, self.Z)
        elif self.activation == 'softmax':
            # Softmax激活函数
            Z_max = xp.max(self.Z, axis=1, keepdims=True)
            exp_Z = xp.exp(self.Z - Z_max)  # 数值稳定的指数计算
            sum_exp_Z = xp.sum(exp_Z, axis=1, keepdims=True)
            self.Y = exp_Z / sum_exp_Z
        else:
            self.Y = self.Z  # 线性激活

        return self.Y  # 传给下一层当输入

    def backward(self, delta_curr, learning_rate, loss_name):
        xp = device_manager.get_xp()
        m = self.Y_prev.shape[0]  # 样本数

        # 计算激活函数的导数
        if self.activation == 'relu':
            d_activation = (self.Z > 0).astype(int)
        elif self.activation == 'sigmoid':
            if self.is_output_layer and loss_name == 'cross_entropy':
                # 输出层：假设使用交叉熵损失，梯度简化
                d_activation = 1
            else:
                # 隐藏层：需要完整计算sigmoid导数
                d_activation = self.Y * (1 - self.Y)

        else:
            d_activation = 1  # 线性激活导数为1
        # 累计梯度
        delta = delta_curr * d_activation

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

    def save_params(self, filepath, layer_index):
        """保存卷积层参数"""
        xp = device_manager.get_xp()

        params_dict = {}

        # 转换参数到CPU
        if hasattr(self.W, 'device'):
            params_dict[f'layer{layer_index}_W'] = xp.asnumpy(self.W)
            params_dict[f'layer{layer_index}_b'] = xp.asnumpy(self.b)
        else:
            params_dict[f'layer{layer_index}_W'] = self.W
            params_dict[f'layer{layer_index}_b'] = self.b

        # 保存配置
        params_dict[f'layer{layer_index}_type'] = self.__class__.__name__
        params_dict[f'layer{layer_index}_activation'] = self.activation

        return params_dict

    def load_params(self, data, layer_index):
        """加载卷积层参数"""
        xp = device_manager.get_xp()

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
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation,
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

        # 应用激活函数
        if self.activation == 'relu':
            output = xp.maximum(0, conv_output_reshaped)
        elif self.activation == 'sigmoid':
            output = 1 / (1 + xp.exp(-conv_output_reshaped))
        else:
            output = conv_output_reshaped

        # 缓存用于反向传播
        self.x = x
        self.conv_output = conv_output_reshaped  # 保存ReLU前的输出

        return output

    def backward(self, d_out, learning_rate, loss_name=None):
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

        # 计算ReLU的梯度
        d_relu = d_out * (self.conv_output > 0).astype(xp.float32)

        # 重塑梯度为二维矩阵: (batch_size*out_h*out_w, out_channels)
        d_relu_2d = d_relu.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # 计算偏置梯度：对每个输出通道的梯度求和
        d_bias = xp.sum(d_relu_2d, axis=0) / batch_size

        # 计算权重梯度：矩阵乘法
        d_weights_2d = xp.dot(self.col.T, d_relu_2d) / batch_size  # shape: (in_channels*k_h*k_w, out_channels)

        # 重塑权重梯度回四维
        k_h, k_w = self.kernel_size
        d_weights = d_weights_2d.T.reshape(self.out_channels, self.in_channels, k_h, k_w)

        # 计算输入梯度：传播到前一层的梯度
        d_col = xp.dot(d_relu_2d, self.col_W.T)  # shape: (batch_size*out_h*out_w, in_channels*k_h*k_w)

        # 使用col2im将梯度还原为输入形状
        d_x = col2im(d_col, self.x.shape, k_h, k_w, self.stride, self.padding)

        # 更新参数
        self.W -= learning_rate * d_weights
        self.b -= learning_rate * d_bias

        return d_x


class Pool2DLayer(Layer):
    def __init__(self, kernel_size, stride=None, padding=0, mode='max'):
        # ... 初始化部分与您的代码完全相同，保持不变 ...
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.padding = padding
        self.mode = mode
        self.cache = {}  # 用于缓存中间结果，如最大值位置

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'mode': self.mode,
            'params_count': 0  # 池化层无参数
        })
        return config

    def forward(self, x):
        """
        前向传播（使用im2col优化）
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
        if self.mode == 'max':
            # 记录最大值位置，用于反向传播
            max_indices = xp.argmax(col_reshaped, axis=1)
            output = col_reshaped[xp.arange(len(max_indices)), max_indices]
        else:  # 'avg'
            output = xp.mean(col_reshaped, axis=1)
            max_indices = None  # 平均池化无需记录位置

        # 5. 将输出重塑为4D张量 (batch_size, channels, out_h, out_w)
        output = output.reshape(batch_size, out_h, out_w, channels).transpose(0, 3, 1, 2)

        # 6. 缓存反向传播所需信息
        self.cache = {
            'x_shape': x.shape,
            'col_shape': col.shape,
            'max_indices': max_indices,
            'input_col': col_reshaped if self.mode == 'avg' else None  # 平均池化需要原始数据
        }

        return output

    def backward(self, d_out, learning_rate=0.01, loss_name='mse'):
        """
        反向传播（使用im2col优化）
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

        # 3. 根据池化模式分配梯度
        if self.mode == 'max':
            # 最大池化：梯度只传递给前向传播中最大值所在的位置
            d_col[xp.arange(len(max_indices)), max_indices] = d_out_flat
        else:
            # 平均池化：梯度平均分配到窗口中的每个位置
            avg_grad = d_out_flat.reshape(-1, 1) / (k_h * k_w)
            d_col[:] = avg_grad  # 将平均梯度分配到整个窗口

        # 4. 重塑梯度以匹配im2col的输出格式
        d_col = d_col.reshape(self.cache['col_shape'])

        # 5. 使用col2im将梯度还原为原始输入形状
        d_x = col2im(d_col, x_shape, k_h, k_w, self.stride, self.padding)

        return d_x


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

    def backward(self, d_out, learning_rate=None, loss_name=None):
        """
        反向传播
        参数:
            d_out: 上游传回的梯度，形状为 (batch_size, flattened_size)
        返回:
            梯度 d_x，形状与原始输入 self.original_shape 相同
        """
        # 核心操作：将梯度重塑回前向传播输入时的形状
        return d_out.reshape(self.original_shape)
