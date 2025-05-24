import numpy as np

#激活函数基类
class Activation:

    def __init__(self):
        pass

    #前向传播
    def forward(self, z):
        pass

    #反向传播
    def backward(self, dl_da):
        pass


class ReLU(Activation):

    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, z):
        self.input = z
        return np.maximum(0, z)

    def backward(self, dl_da):
        if self.input is None:
            raise ValueError("You must call the forward first!")

        #正值梯度为1，负值梯度为0
        da_dz = (self.input > 0).astype(float)
        dl_dz = da_dz * dl_da
        return dl_dz


class Softmax(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        #为防止上溢，给向量的每一项减去最大的一项
        processed_z = z - np.max(z, axis=1, keepdims=True)
        #processed与exp_z的形状都与z一致
        exp_z = np.exp(processed_z)
        output = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return output

    #损失函数是交叉熵，所以直接返回y_pred - y_true，简化运算
    def backward(self, dl_dz_from_loss):
        #这里把大部分逻辑交给损失函数处理
        return dl_dz_from_loss