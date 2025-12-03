import numpy as np


class LossFunction:
    def __init__(self, name, loss_func, gradient_func):
        self.name = name
        self.loss_func = loss_func
        self.gradient_func = gradient_func

    def compute_loss(self, y_pred, y_true):
        return self.loss_func(y_pred, y_true)

    def compute_gradient(self, y_pred, y_true):
        return self.gradient_func(y_pred, y_true)


# 创建损失函数工厂
class LossFactory:
    @staticmethod
    def get_loss(loss_name):
        losses = {
            'mse': LossFunction('mse', mse_loss, mse_gradient),
            'cross_entropy': LossFunction('cross_entropy', cross_entropy_loss, cross_entropy_gradient),
        }
        return losses.get(loss_name, losses['mse'])


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mse_gradient(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_pred.shape[0]


def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    # 防止log(0)溢出
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss


def cross_entropy_gradient(y_pred, y_true):
    return (y_pred - y_true) / y_pred.shape[0]  # 梯度公式
