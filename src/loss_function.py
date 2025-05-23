import numpy as np

#损失函数基类
class Loss:

    def __init__(self):
        pass

    #计算损失值
    def forward(self, y_pred, y_true):
        pass
    #损失函数关于预测值的梯度
    def backward(self, y_pred, y_true):
        pass


#L = -sum_i(y_true_i * log(y_pred_i))
class CrossEntropyLoss(Loss):

    def __init__(self):
        super().__init__()

    #y_pred: Softmax的输出，是预测概率分布
    #y_true: 从数据集读入的标签，独热编码
    #y_pred和y_true应有一样的形状，都是(batch_size, class_num)
    def forward(self, y_pred, y_true):
        #num_samples = y_pred.shape[0]

        #限制y_pred的范围在(0, 1)
        #定义一个极小数
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred, epsilon, 1.0-epsilon)

        #计算所有sample各自的损失，losses是一个长度为num_samples的向量
        all_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)

        #计算批中样本的平均损失 #这是一个标量
        mean_loss = np.mean(all_losses)
        return mean_loss

    def backward(self, y_pred, y_true):
        num_samples = y_pred.shape[0]

        #记录每个样本的梯度，形状与y_pred一致 #这里合并了Softmax和交叉熵的梯度
        dl_dz = y_pred - y_true

        dl_dz_average = dl_dz / num_samples

        return dl_dz_average