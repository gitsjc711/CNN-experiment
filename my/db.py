import gzip
import struct
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt


def read_idx_images(filename):
    # 检查文件扩展名，决定使用普通打开还是gzip打开
    if filename.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open

    with opener(filename, 'rb') as f:
        # 读取文件头（4个32位整数，大端序）
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        print(f"魔数: {magic}, 图像数量: {num}, 行数: {rows}, 列数: {cols}")
        # 读取所有图像数据
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # 重塑为 [数量, 行, 列] 的张量
        data = data.reshape(num, rows, cols)
        return data


def read_idx_labels(filename):
    # 检查文件扩展名，决定使用普通打开还是gzip打开
    if filename.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open

    with opener(filename, 'rb') as f:
        # 读取文件头（2个32位整数）
        magic, num = struct.unpack('>II', f.read(8))
        print(f"魔数: {magic}, 标签数量: {num}")
        # 读取所有标签数据
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


class DataPreprocessing:
    def __init__(self, data_path='./data/MNIST/mnist_dataset', save_path='./data/MNIST/my_mnist_dataset'):
        self.data_path = data_path
        self.save_path = save_path

    def images_to_numpy(self, extra_path, save_name):
        """
        将文件夹内的PNG图片转换为numpy数组并保存
        参数:
            data_path: 原始图片文件夹路径
            save_path: 保存numpy数组的路径
        """
        # 确保保存路径存在
        os.makedirs(self.save_path, exist_ok=True)
        data_path = self.data_path + extra_path
        # 获取所有PNG文件并排序
        image_files = [f for f in os.listdir(data_path) if f.endswith('.png')]
        image_files.sort(key=lambda x: int(x.split('.')[0]))  # 按数字排序

        print(f"找到 {len(image_files)} 张PNG图片")

        # 初始化列表存储图像数据
        images = []

        # 处理每张图片
        for img_file in tqdm(image_files, desc="处理图片", ncols=80):
            img_path = os.path.join(data_path, img_file)

            # 打开图片并转换为灰度
            img = Image.open(img_path).convert('L')

            # 确保图片尺寸为28x28
            if img.size != (28, 28):
                img = img.resize((28, 28), Image.Resampling.LANCZOS)

            # 转换为numpy数组并归一化到[0,1]
            img_array = np.array(img) / 255.0

            # 重塑为(1, 28, 28)形状
            img_array = img_array.reshape(1, 28, 28)

            # 添加到列表
            images.append(img_array)

        # 将所有图片堆叠成一个数组
        if images:
            images = np.vstack(images)
            print(f"最终数组形状: {images.shape}")

            # 保存为numpy文件
            save_file = os.path.join(self.save_path, save_name + '.npy')
            np.save(save_file, images)
            print(f"数据已保存到: {save_file}")

            return images
        else:
            print("未找到任何图片文件")
            return None

    # 和image_to_numpy差不多读取file_name.txt文件，然后封装成(n,1)数组并·保存为save_name+'.npy'
    def txt_to_numpy(self, file_name, save_name):
        """
        读取空格分隔的TXT文件，提取第二列标签数据并保存为numpy数组
        参数:
            file_name: TXT文件名
            save_name: 保存的numpy文件名
        """
        # 确保保存路径存在
        os.makedirs(self.save_path, exist_ok=True)

        # 构建完整文件路径
        file_path = os.path.join(self.data_path, file_name)

        try:
            print(f"正在读取文件: {file_path}")

            # 读取所有行
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 提取第二列数据（标签）
            labels = []
            for line in lines:
                parts = line.strip().split()  # 按空格分割
                if len(parts) >= 2:  # 确保至少有两列
                    labels.append(int(parts[1]))  # 第二列是标签

            # 转换为numpy数组并重塑为(n, 1)形状
            labels_array = np.array(labels).reshape(-1, 1)

            print(f"成功读取 {len(labels_array)} 个标签")
            print(f"数组形状: {labels_array.shape}")

            # 保存为numpy文件
            save_file = os.path.join(self.save_path, save_name + '.npy')
            np.save(save_file, labels_array)
            print(f"标签数据已保存到: {save_file}")

            return labels_array

        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 不存在")
            return None
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")
            return None

    # 读取文件，并转化npy到numpy里返回
    def load_file(self, file_name):
        # 构建完整文件路径
        file_path = os.path.join(self.save_path, file_name)
        try:
            # 加载.npy文件
            array_data = np.load(file_path)
            print(f"成功加载.npy文件: {file_path}")

            return array_data

        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 不存在")
            return None
        except Exception as e:
            print(f"加载文件时出错: {str(e)}")
            return None

    def load_file_as_onehot(self, file_name, num_classes=10):
        """
        加载文件并直接转换为 one-hot 编码
        """
        # 先加载原始数据
        labels = self.load_file(file_name)

        if labels is not None:
            # 转换为 one-hot 编码
            labels_flat = labels.flatten()
            onehot_labels = np.eye(num_classes)[labels_flat.astype(int)]
            print(f"成功将标签转换为 one-hot 编码，形状: {onehot_labels.shape}")
            return onehot_labels
        return None

    def display_sample_images(self, images, num_samples=5):
        """
        显示样本图片以验证转换结果
        """
        matplotlib.use('TkAgg')
        if images is not None:
            fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))

            if num_samples == 1:
                axes = [axes]

            for i in range(min(num_samples, len(images))):
                img = images[i]
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f'sample {i}')
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()
