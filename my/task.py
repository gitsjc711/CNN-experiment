from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from my.layers import FullyConnectedLayer, Conv2DLayer, Pool2DLayer,Flatten,Dropout,BatchNormalization
from my.models import NeuralNetwork
from my.device_manager import device_manager
from sklearn.metrics import accuracy_score, classification_report, f1_score
from my.db import DataPreprocessing
from my.activation import ReLU,Softmax
# 注意我把激活函数拆出来了，需要自己写
def full_connected_task(epochs, lr, batch_size, net_seed=40, db_seed=20, test_size=0.3):
    device_manager.set_device('gpu')
    xp = device_manager.get_xp()
    # 全连接任务
    xp.random.seed(net_seed)
    # 定义神经网络net
    net = NeuralNetwork([
        FullyConnectedLayer(2, 16, 'relu'),
        FullyConnectedLayer(16, 1, 'sigmoid')
    ])
    # 这是合成数据集
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1,
        random_state=db_seed
    )
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=test_size, random_state=db_seed)
    y_train_col = y_train.reshape(-1, 1)
    y_test_col = y_test.reshape(-1, 1)
    net.train(X_train, y_train_col, batch_size=batch_size, epochs=epochs, lr=lr, loss_name='cross_entropy')
    # net.load_weights('save/epoch_ 200000_loss_ 0.081261.npz')
    net_pred = net.predict(X_test)
    # 评价分类的，干别的任务这里记得换
    # 将概率值转换为类别标签（以0.5为阈值）
    threshold = 0.5
    net_pred_labels = (net_pred > threshold).astype(int)  # 这是关键修正
    if xp.__name__ == 'cupy':
        net_pred_labels = net_pred_labels.get()
    # 使用转换后的类别标签来计算 F1 分数
    f1 = f1_score(y_true=y_test, y_pred=net_pred_labels)
    print(f"F1 Score: {f1: .4f}")


# 注意我把激活函数拆出来了，需要自己写
def full_connected_task_for_image(epochs, lr, batch_size, net_seed=40, db_seed=20, test_size=0.3):
    device_manager.set_device('gpu')
    xp = device_manager.get_xp()
    # 全连接任务
    xp.random.seed(net_seed)
    # 定义神经网络net
    net = NeuralNetwork([
        FullyConnectedLayer(784, 1024, 'relu'),
        FullyConnectedLayer(1024, 1024, 'relu'),
        FullyConnectedLayer(1024, 10, 'softmax')
    ])
    print("构建网络完成")
    net.summary()
    # 处理数据集
    db = DataPreprocessing()
    x_train = db.load_file('train_image.npy')
    y_train = db.load_file_as_onehot('train_labs.npy')
    x_test = db.load_file('test_image.npy')
    y_test = db.load_file_as_onehot('test_labs.npy')
    x_train = x_train.reshape(-1, 784)  # 从(55000, 28, 28)变为(55000, 1, 28, 28)
    x_test = x_test.reshape(-1, 784)  # 同样处理测试集
    print("数据集初始化完成")
    net.train(x_train, y_train, epochs, lr, batch_size=batch_size, loss_name='cross_entropy',save_mode=1)
    # net.load_weights('./result/full_connected_net/save/epoch_2600_loss_0.002508.npz')
    y_pred_prob = net.predict(x_test)
    y_pred = xp.argmax(y_pred_prob, axis=1)
    if hasattr(y_pred, 'get'):  # 检查是否为CuPy数组
        y_pred_np = y_pred.get()
    else:
        y_pred_np = y_pred
    # 计算准确率
    import numpy as np
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred_np)
    print(f"测试集准确率 (Accuracy): {accuracy:.4f}")

    # 生成详细分类报告（包括精确率、召回率、F1分数等）
    print("\n分类报告 (Classification Report):")
    print(classification_report(y_test_labels, y_pred_np, target_names=[str(i) for i in range(10)]))

    f1_weighted = f1_score(y_test_labels, y_pred_np, average='weighted')
    print(f"\n加权平均F1分数 (Weighted F1-Score): {f1_weighted:.4f}")

def conv_task(epochs, lr, batch_size, net_seed=40):
    device_manager.set_device('gpu')
    xp = device_manager.get_xp()
    # 全连接任务
    xp.random.seed(net_seed)
    # 定义神经网络net
    net = NeuralNetwork([
        Conv2DLayer(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
        BatchNormalization(num_features=6),
        ReLU(),
        Pool2DLayer(kernel_size=2, stride=2, mode='max'),
        Conv2DLayer(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
        BatchNormalization(num_features=16),
        ReLU(),
        Pool2DLayer(kernel_size=2, stride=2, mode='max'),
        Conv2DLayer(in_channels=16, out_channels=64, kernel_size=4, stride=1, padding=0),
        ReLU(),
        Flatten(),
        Dropout(),
        FullyConnectedLayer(64, 10),
        Softmax()
    ])
    print("构建网络完成")
    net.summary()
    # 处理数据集
    db = DataPreprocessing()
    x_train = db.load_file('train_image.npy')
    y_train = db.load_file_as_onehot('train_labs.npy')
    x_test = db.load_file('test_image.npy')
    y_test = db.load_file_as_onehot('test_labs.npy')
    x_train = x_train.reshape(-1, 1, 28, 28)  # 从(55000, 28, 28)变为(55000, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)  # 同样处理测试集
    print("数据集初始化完成")
    net.set_save_dir("./result/卷积任务/有批量归一化")
    net.train(x_train, y_train, epochs, lr, batch_size=batch_size, loss_name='cross_entropy', save_mode=1, save_interval=10)
    # net.load_weights('epoch_0500_loss_0.071979.npz')
    y_pred_prob = net.predict(x_test)
    y_pred = xp.argmax(y_pred_prob, axis=1)
    if hasattr(y_pred, 'get'):  # 检查是否为CuPy数组
        y_pred_np = y_pred.get()
    else:
        y_pred_np = y_pred
    # 计算准确率
    import numpy as np
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred_np)
    print(f"测试集准确率 (Accuracy): {accuracy:.4f}")

    # 生成详细分类报告（包括精确率、召回率、F1分数等）
    print("\n分类报告 (Classification Report):")
    print(classification_report(y_test_labels, y_pred_np, target_names=[str(i) for i in range(10)]))

    f1_weighted = f1_score(y_test_labels, y_pred_np, average='weighted')
    print(f"\n加权平均F1分数 (Weighted F1-Score): {f1_weighted:.4f}")
