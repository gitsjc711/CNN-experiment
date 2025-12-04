import os
from .device_manager import device_manager, DeviceManager  # 导入设备管理器
from tqdm import tqdm
from my.loss_factory import LossFactory
import numpy as np
from .device_manager import device_manager


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers  # 层对象列表
        if self.layers:  # 防止空列表报错
            self.layers[-1].is_output_layer = True
        self.save_dir = "save"

    def save_dir(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_weights(self, filepath):
        """
        将网络所有权重和偏置参数保存到一个NPZ文件中。
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        # 创建一个空字典来存储所有参数
        save_dict = {}
        # 保存网络元信息
        save_dict['network_total_layers'] = len(self.layers)
        save_dict['network_save_time'] = np.array(str(np.datetime64('now')))  # 保存时间戳

        # 让每一层保存自己的参数
        for i, layer in enumerate(self.layers):
            layer_dict = layer.save_params( i)  # 每层返回自己的参数字典
            if layer_dict:  # 只有有参数的层会返回非空字典
                save_dict.update(layer_dict)

        # 使用压缩格式保存[5](@ref)
        np.savez_compressed(filepath, **save_dict)

    def load_weights(self, filename):
        try:
            import numpy as np
            filepath = os.path.join(self.save_dir, filename)
            if not os.path.exists(filepath):
                print(f"错误：文件 {filepath} 未找到")
                return False

            data = np.load(filepath)

            # 验证网络结构兼容性
            if 'network_total_layers' in data:
                saved_layers = data['network_total_layers']
                if saved_layers != len(self.layers):
                    print(f"网络结构不匹配：保存的模型有{saved_layers}层，当前网络有{len(self.layers)}层")
                    return False

            # 让每一层加载自己的参数
            success_count = 0
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'load_params'):
                    if layer.load_params(data, i):
                        success_count += 1

            print(f"权重加载成功：{success_count}/{len(self.layers)}层参数已加载")
            return success_count > 0  # 至少有一层成功加载即认为成功

        except Exception as e:
            print(f"加载权重时发生错误: {e}")
            return False

    def summary(self):
        """
        打印网络结构摘要，仅包含层名称、构造参数、总参数和激活函数。
        """
        print("=" * 80)
        print("神经网络结构摘要")
        print("=" * 80)
        print(f"{'层索引':<8} {'层类型':<20} {'参数数量':<12} {'激活函数':<12} {'配置信息'}")
        print("-" * 80)

        total_params = 0
        trainable_params = 0

        for i, layer in enumerate(self.layers):
            config = layer.get_config()
            layer_type = config.get('type', 'Unknown')
            params_count = config.get('params_count', 0)
            activation = config.get('activation', 'None')

            # 构造配置信息字符串
            config_info = ""
            if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                config_info = f"in={layer.in_channels}, out={layer.out_channels}"
                if hasattr(layer, 'kernel_size'):
                    k_size = layer.kernel_size
                    if isinstance(k_size, tuple):
                        k_size = f"{k_size[0]}x{k_size[1]}"
                    config_info += f", k={k_size}"
            elif hasattr(layer, 'input_dim') and hasattr(layer, 'output_dim'):
                config_info = f"in={layer.input_dim}, out={layer.output_dim}"

            total_params += params_count
            if params_count > 0:
                trainable_params += params_count

            print(f"{i:<8} {layer_type:<20} {params_count:<12} {activation:<12} {config_info}")

        print("-" * 80)
        print(f"总参数数: {total_params}")
        print(f"可训练参数: {trainable_params}")
        print(f"非可训练参数: {total_params - trainable_params}")
        print("=" * 80)

    def forward(self, X):
        Y = X
        for layer in self.layers:
            Y = layer.forward(Y)
        return Y

    def backward(self, delta, learning_rate):
        # 反向传播要从最后一层开始，所以要reversed（）生成一个反向迭代器
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)

    def predict(self, X):
        Y = device_manager.to_device(X)
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.eval()
        for layer in self.layers:
            Y = layer.forward(Y)
        return Y

    def train(self, X, y, epochs, lr, batch_size=None, loss_name='mse', save_mode=None, save_interval=100,
              checkpoint_path=None, use_inner_pbar=False):
        """
        增强的训练方法，支持从检查点恢复训练
        参数:
            X: 训练数据
            y: 训练标签
            epochs: 总训练轮次
            lr: 学习率
            batch_size:批量大小。如果为None，则使用批量梯度下降；如果为整数，则使用小批量梯度下降
            loss_name: 损失函数名称
            save_interval: 权重保存间隔
            checkpoint_path: 检查点权重文件路径。若提供，则从此检查点恢复训练
        """
        # 环境初始化
        loss_fn = LossFactory.get_loss(loss_name)
        start_epoch = 0  # 初始训练轮次
        initial_loss = None  # 初始损失值
        # 设置训练模式
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.train()
        if save_mode is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # 如果提供了检查点路径，则加载权重并获取初始轮次和损失
        if checkpoint_path is not None:
            if self.load_weights(checkpoint_path):
                # 从文件名解析初始轮次和损失，例如：epoch_0100_loss_0_123456.npz
                import re
                match = re.search(r'epoch_(\d+)_loss_([\d.]+)\.npz$', checkpoint_path)
                if match:
                    start_epoch = int(match.group(1))
                    # 文件名中损失值已经使用小数点，无需替换操作
                    initial_loss = float(match.group(2))  # 直接转换为 float
                    print(f"从检查点恢复训练: Epoch {start_epoch}, Loss: {initial_loss}")
                else:
                    print("警告: 无法从文件名解析训练进度，将从第0轮开始记录")
                    start_epoch = 0
            else:
                print("警告: 检查点加载失败，将从零开始训练")
        X = device_manager.to_device(X)
        y = device_manager.to_device(y)
        num_samples = X.shape[0]
        # 根据是否设置batch_size决定训练方式
        if batch_size is None or batch_size >= num_samples:
            # 批量梯度下降：使用全部数据作为一个批次[1](@ref)
            batch_size = num_samples
            num_batches = 1
            print(f"使用批量梯度下降，批次大小: {batch_size}")
        else:
            # 小批量梯度下降：将数据分为多个批次
            num_batches = (num_samples + batch_size - 1) // batch_size  # 向上取整
            print(f"使用小批量梯度下降，批次大小: {batch_size}, 总批次数: {num_batches}")
        # 外层进度条配置
        epoch_pbar = tqdm(range(start_epoch + 1, epochs + 1),
                          desc="总体训练进度",
                          unit="epoch",
                          position=0,
                          leave=True,
                          bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                          colour='green')

        for epoch in epoch_pbar:
            epoch_loss = 0.0  # 记录当前epoch的总损失
            # 如果使用小批量训练，每个epoch内需要遍历
            # 内层进度条配置
            if use_inner_pbar:
                batch_pbar = tqdm(range(num_batches),
                                  desc=f"Epoch {epoch:03d}/{epochs:03d}",
                                  unit="batch",
                                  position=1,
                                  leave=False,  # 重要：不保留，每个epoch结束后清除
                                  bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                                  colour='blue',
                                  mininterval=0.5)  # 更快的更新频率
            else:
                # 不使用内层进度条，只在epoch级别更新
                batch_range = range(num_batches)
            for batch_idx in (batch_pbar if use_inner_pbar else batch_range):
                # 计算当前批次的起始和结束索引
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)

                # 获取当前批次的数据
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                # 前向传播
                y_pred = self.forward(X_batch)
                batch_loss = loss_fn.compute_loss(y_pred, y_batch)
                delta = loss_fn.compute_gradient(y_pred, y_batch)
                # 累加损失（后续计算平均损失）
                epoch_loss += batch_loss
                # 反向传播
                self.backward(delta, lr)
            # 定期保存和打印（相对于当前起始轮次的间隔）
            avg_epoch_loss = epoch_loss / num_batches
            epoch_pbar.set_postfix(loss=f'{avg_epoch_loss:.4f}')
            current_epoch_relative = epoch - start_epoch
            if save_mode is not None:
                if current_epoch_relative % save_interval == 0:
                    filename = f"epoch_{epoch:04d}_loss_{avg_epoch_loss:.6f}.npz"
                    filepath = os.path.join(self.save_dir, filename)
                    self.save_weights(filepath)

        print(f"训练完成！")
