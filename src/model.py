import numpy as np
from src.activations import ReLU, Softmax
from src.layer import FullyConnectedLayer, DropoutLayer, FlattenLayer, ConvLayer, MaxPoolingLayer
from src.loss_function import CrossEntropyLoss
import pickle
import threading
from contextlib import contextmanager

#这是一个全局上下文标志
_no_grad = threading.local()
_no_grad.enable = False

#一个简单的卷积神经网络
#input -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> ReLU -> Dropout -> FC -> Softmax -> output
class NeuralNetwork:

    def __init__(self, input_shape=(32, 32, 3), hidden_size=128, output_class=10, dropout_rate=0.3):
        self.using_dropout = dropout_rate > 0
        self.is_training = True
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_class = output_class
        self.dropout_rate = dropout_rate

        #输入数据参数
        h, w, c = input_shape

        #初始化神经网络层
        self.layers = {}

        #卷积+池化层1
        self.layers['conv1'] = ConvLayer(in_channels=c, kernel_size=(3, 3), num_kernels=16)
        self.layers['relu1'] = ReLU()
        self.layers['pool1'] = MaxPoolingLayer(pool_size=(2, 2))
        #卷积层1的输出形状
        h_conv1 = (h - self.layers['conv1'].kernel_height) // self.layers['conv1'].stride + 1
        w_conv1 = (w - self.layers['conv1'].kernel_width) // self.layers['conv1'].stride + 1
        h_pool1 = (h_conv1 - self.layers['pool1'].pool_h) // self.layers['pool1'].stride[0] + 1
        w_pool1 = (w_conv1 - self.layers['pool1'].pool_w) // self.layers['pool1'].stride[1] + 1
        c_pool1 = 16

        #卷积+池化层2
        self.layers['conv2'] = ConvLayer(in_channels=c_pool1, kernel_size=(3, 3), num_kernels=32)
        self.layers['relu2'] = ReLU()
        self.layers['pool2'] = MaxPoolingLayer(pool_size=(2, 2))
        # 卷积层2的输出形状
        h_conv2 = (h_pool1 - self.layers['conv2'].kernel_height) // self.layers['conv2'].stride + 1
        w_conv2 = (w_pool1 - self.layers['conv2'].kernel_width) // self.layers['conv2'].stride + 1
        h_pool2 = (h_conv2 - self.layers['pool2'].pool_h) // self.layers['pool2'].stride[0] + 1
        w_pool2 = (w_conv2 - self.layers['pool2'].pool_w) // self.layers['pool2'].stride[1] + 1
        c_pool2 = 32

        #展平层
        self.layers['flatten'] = FlattenLayer()

        #全连接层1
        self.layers['fc1'] = FullyConnectedLayer(input_size=h_pool2 * w_pool2 * c_pool2, output_size=hidden_size)
        self.layers['relu3'] = ReLU()

        #dropout层
        if self.using_dropout:
            self.layers['dropout'] = DropoutLayer(dropout_rate=dropout_rate)

        #全连接层2
        self.layers['fc2'] = FullyConnectedLayer(input_size=hidden_size, output_size=output_class)
        self.layers['softmax'] = Softmax()

        #初始化损失函数
        self.loss_func = CrossEntropyLoss()


    #input形状: (batch_size, input_size)
    def forward(self, x):

        #卷积+池化层1
        c1 = self.layers['conv1'].forward(x)
        r1 = self.layers['relu1'].forward(c1)
        p1 = self.layers['pool1'].forward(r1)

        #卷积+池化层2
        c2 = self.layers['conv2'].forward(p1)
        r2 = self.layers['relu2'].forward(c2)
        p2 = self.layers['pool2'].forward(r2)

        #展平层
        f = self.layers['flatten'].forward(p2)

        #全连接层1
        h1 = self.layers['fc1'].forward(f)
        r3 = self.layers['relu3'].forward(h1)

        #dropout层
        if self.using_dropout and self.is_training:
            d = self.layers['dropout'].forward(r3)
        else:
            d = r3

        #全连接层2
        h2 = self.layers['fc2'].forward(d)
        #softmax层
        y_pred = self.layers['softmax'].forward(h2)

        return y_pred


    def loss(self, y_pred, y_true):

        #计算交叉熵损失
        data_loss = self.loss_func.forward(y_pred, y_true)

        return data_loss


    def backward(self, y_pred, y_true):

        #如果禁用了梯度计算，直接返回
        if _no_grad.enable:
            return None

        #合并了交叉熵和softmax的梯度
        dh2 = self.loss_func.backward(y_pred, y_true)

        #全连接层2
        dr3 = self.layers['fc2'].backward(dh2)
        if self.using_dropout and self.is_training:
            dr3 = self.layers['dropout'].backward(dr3)

        #全连接层1
        dh1 = self.layers['relu3'].backward(dr3)
        df = self.layers['fc1'].backward(dh1)

        #展平层
        dp2 = self.layers['flatten'].backward(df)

        #卷积+池化层2
        dr2 = self.layers['pool2'].backward(dp2)
        dc2 = self.layers['relu2'].backward(dr2)
        dp1 = self.layers['conv2'].backward(dc2)

        #卷积+池化层1
        dr1 = self.layers['pool1'].backward(dp1)
        dc1 = self.layers['relu1'].backward(dr1)
        dx = self.layers['conv1'].backward(dc1)


    def train(self, is_training=True):
        #决定是否设置为训练模式
        self.is_training = is_training

        if self.using_dropout:
            if self.is_training:
                self.layers['dropout'].train_mode()

            else:
                self.layers['dropout'].val_mode()

        return self


    #设为验证模式
    def val(self):
        return self.train(False)


    #清零所有参数的梯度
    def zero_grad(self):
        for layer_name in self.layers:
            if hasattr(self.layers[layer_name], 'grad_zero'):
                self.layers[layer_name].grad_zero()


    def get_params(self):
        params = {}
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'params') and layer.params:
                params[layer_name] = {'W': layer.params['W'].copy(),
                                      'b': layer.params['b'].copy()}

        return params


    def set_params(self, params):
        for layer_name, layer_params in params.items():
            if hasattr(self.layers[layer_name], 'params') and self.layers[layer_name].params:
                self.layers[layer_name].params['W'] = layer_params['W'].copy()
                self.layers[layer_name].params['b'] = layer_params['b'].copy()


    #用pickle来序列化模型参数
    def save_model(self, filepath):
        #用一个字典来存储参数和输入形状
        model_data = {
            "params": self.get_params(),
            "input_shape": self.input_shape,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)


    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            params = pickle.load(f)['params']
        self.set_params(params)

    #用contextmanager自定义上下文管理器，临时禁用梯度计算，退出上下文时恢复
    @contextmanager
    def no_grad(self):
        present_status = _no_grad.enable

        #禁用梯度计算
        _no_grad.enable = True

        try:
            #执行到这里时no_grad暂停执行，开始执行with后面的代码块
            yield
        #with后面的代码块执行结束
        finally:
            #恢复
            _no_grad.enable = present_status

    def predict(self, x):
        #设置为验证模式
        self.val()

        with self.no_grad():
            # 前向传播
            y_pred = self.forward(x)
            # 返回预测类别
            return np.argmax(y_pred, axis=1)
