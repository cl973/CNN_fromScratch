import numpy as np
import numba

#神经网络层基类
class Layer:

    def __init__(self):
        self.params = {}
        self.grads = {}
        self.input_shape = None

    def forward(self, inputs):
        pass

    def backward(self, dout):
        pass

    # 梯度归零
    def grad_zero(self):
        for grad in self.grads:
            self.grads[grad] = np.zeros_like(self.params[grad])

#全连接层
class FullyConnectedLayer(Layer):

    # input_size: 输入特征向量的维度
    # output_size: 输出特征向量的维度
    def __init__(self, input_size, output_size):
        super().__init__()

        #随机生成一个权重矩阵并进行He正态初始化，加速收敛
        scale = np.sqrt(2.0 / input_size)
        self.params['W'] = np.random.randn(input_size, output_size) * scale

        #生成一个全0的偏置向量
        self.params['b'] = np.zeros(output_size)

        #将W和b的梯度初始化为0
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])

        self.inputs = None

    #这里input的形状是(batch_size, input_size)
    def forward(self, inputs):
        #存储inputs，用于后续反向传播
        self.inputs = inputs
        #偏置向量会加到inputs @ self.params['W']的每一行
        output = inputs @ self.params['W'] + self.params['b']

        return output

    #dout: 从上游过来的输出的梯度
    #din: 传给下游的梯度
    def backward(self, dout):

        #output = inputs @ W + b, 所以doutput/dW = inputs^T, dL/dW = inputs^T @ dout
        self.grads['W'] = self.inputs.T @ dout

        #doutput/db = 1, 因此dL/db = 1 @ dout, 也就是把dout所有行加起来
        self.grads['b'] = np.sum(dout, axis=0)

        #doutput/dinputs = W^T, din = dout @ W^T
        din = dout @ self.params['W'].T

        return din


#进行一些简单的正则化
#dropout正则化层，用于将一部分神经元输出置0，防止过拟合
class DropoutLayer(Layer):

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        #以掩码的形式来丢弃前层输出
        self.mask = None
        #训练模式
        self.is_training = True

    def forward(self, inputs):
        if self.is_training:
            #生成一个元素值在0到1之间的与input形状相同的矩阵，元素大于dropout_rate的位置置1，小于的置0
            self.mask = np.random.rand(*inputs.shape) > self.dropout_rate

            #正则化后保持期望不变
            scale = 1 / (1 - self.dropout_rate)
            output = inputs * self.mask * scale

        #测试模式下正则化层不产生作用
        else:
            output = inputs

        return output

    #在正则化层的反向传播中，只传递被保留的神经元的梯度，被丢弃的神经元不传递梯度
    #这里默认只在训练模式下使用反向传播
    def backward(self, dout):
        #应用与前向传播相同的缩放
        #被保留的神经元输入x, 输出x / (1 - self.dropout_rate)，被丢弃的神经元输出0
        #综合来看，dout/dx = self.mask * scale
        scale = 1 / (1 - self.dropout_rate)
        din = dout * self.mask * scale

        return din

    def train_mode(self):
        self.is_training = True

    def val_mode(self):
        self.is_training = False


#展平层
class FlattenLayer(Layer):

    def __init__(self):
        super().__init__()
        self.original_shape = None

    def forward(self, inputs):
        self.original_shape = inputs.shape

        #把原张量展平成(sample_num, num_row * num_column * num_channel)
        return inputs.reshape(self.original_shape[0], -1)

    def backward(self, dout):
        #只负责把形状转回去
        return dout.reshape(self.original_shape)

#尝试用numba来加速运算
#把卷积层前向运算的内层循环提取为Numba JIT函数
@numba.njit(cache=True)
def _conv_forward_numba(inputs_sample_np,   # 单个样本的输入 (h_in, w_in, c_in)
                        w_kernel_np,           # 单个卷积核的权重
                        b_kernel_np,           # 单个卷积核的偏置
                        output_slice_np,    # 用于填充的单个输出通道切片 (H_out, W_out)
                        h_out, w_out,
                        kernel_height, kernel_width,
                        stride, in_channels_for_kernel):
    for row in range(h_out):
        for column in range(w_out):
            start_row = row * stride
            start_column = column * stride

            #点乘计算
            for k_height in range(kernel_height):
                for k_width in range(kernel_width):
                    for c_in in range(in_channels_for_kernel):
                        output_slice_np[row, column] += inputs_sample_np[
                                                            start_row + k_height, start_column + k_width, c_in] * \
                                                        w_kernel_np[k_height, k_width, c_in]
            #加上偏置
            output_slice_np[row, column] += b_kernel_np


#把卷积层反向运算的内层循环提取为Numba JIT函数
@numba.njit(cache=True)
def _conv_backward_numba(inputs_sample_np,      # 单个样本的输入
                        dout_sample_np,         # 单个样本的 dout (h_out, w_out, NumKernels)
                        w_all_kernels_np,       # 所有滤波器权重 (KH, KW, C_in, NumKernels)
                        grad_w_np,              # 梯度 dL/dW (KH, KW, C_in, NumKernels)
                        din_sample_np,          # 梯度 dL/dinput (H_in, W_in, C_in)
                        h_out, w_out,
                        kernel_height, kernel_width,
                        stride, in_channels, num_kernels_dout):
    for kernel in range(num_kernels_dout):
        current_kernel = w_all_kernels_np[:, :, :, kernel]
        for row in range(h_out):
            for column in range(w_out):
                start_row = row * stride
                start_column = column * stride

                dout_val = dout_sample_np[row, column, kernel]

                #计算dl/dW
                for k_height in range(kernel_height):
                    for k_width in range(kernel_width):
                        for c_in in range(in_channels):
                            grad_w_np[k_height, k_width, c_in, kernel] += dout_val * inputs_sample_np[
                                start_row + k_height, start_column + k_width, c_in]

                #计算dl/dinput
                for k_height in range(kernel_height):
                    for k_width in range(kernel_width):
                        for c_in in range(in_channels):
                            din_sample_np[start_row + k_height, start_column + k_width, c_in] += dout_val * \
                                                                                                 current_kernel[
                                                                                                     k_height, k_width,
                                                                                                     c_in]



#卷积层
class ConvLayer(Layer):

    #在本实验中只实现步长为1，padding为valid的版本
    #kernel_size只存放卷积核的高与宽
    #卷积核的通道数与in_channels相同
    def __init__(self, in_channels, kernel_size, num_kernels, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.stride = stride

        #kernel的形状(kernel_height, kernel_width, in_channels, num_kernels)
        self.kernel_height, self.kernel_width = kernel_size
        #用何恺明的方法保持ReLU激活值方差稳定
        scale = np.sqrt(2.0 / (self.kernel_height * self.kernel_width * self.in_channels))
        self.params['W'] = np.random.randn(self.kernel_height, self.kernel_width, in_channels, num_kernels).astype(
            np.float32) * scale
        self.params['b'] = np.zeros(num_kernels, dtype=np.float32)
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])

        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs.astype(np.float32, copy=False)

        #记录输入数据的形状，规定输出数据的形状
        n, h_in, w_in, c_in = inputs.shape
        h_out = (h_in - self.kernel_height) // self.stride + 1
        w_out = (w_in - self.kernel_width) // self.stride + 1
        #卷积核数决定了输出的通道数
        output = np.zeros((n, h_out, w_out, self.num_kernels), dtype=np.float32)

        '''#计算output中output[i, row, column, kernel]这个位置的数值
                for i in range(n):
                    #对每个位置应用每个卷积核
                    for kernel in range(self.num_kernels):
                        for row in range(h_out):
                            for column in range(w_out):
                                #输入张量中被这次卷积覆盖的部分
                                #起始行和列
                                start_row = row * self.stride
                                start_column = column * self.stride

                                #终点行和列
                                #这个是边界，不含入计算范围内，便于与python的特性配合
                                end_row = start_row + self.kernel_height
                                end_column = start_column + self.kernel_width

                                #感受野 #形状为(self.kernel_height, self.kernel_width, in_channels)
                                receptive_field = inputs[i, start_row:end_row, start_column:end_column, :]
                                #当前计算的卷积核 #形状为(self.kernel_height, self.kernel_width, in_channels)
                                current_kernel = self.params['W'][:, :, :, kernel]

                                #使用点乘计算output[i, row, column, kernel]，即两个形状相同的张量对应位置元素相乘
                                output[i, row, column, kernel] = np.sum(receptive_field * current_kernel) + self.params['b'][
                                    kernel]
        '''

        kernel_height = self.kernel_height
        kernel_width = self.kernel_width
        stride = self.stride
        in_channels = self.in_channels
        for i in range(n):
            #当前的单个样本
            inputs_sample_np = self.inputs[i, :, :, :]
            for kernel in range(self.num_kernels):
                w_kernel = self.params['W'][:, :, :, kernel]
                b_kernel = self.params['b'][kernel]
                output_slice_np = output[i, :, :, kernel]

                #调用numba JIT函数
                _conv_forward_numba(inputs_sample_np,
                                    w_kernel,
                                    b_kernel,
                                    output_slice_np,
                                    h_out, w_out,
                                    kernel_height, kernel_width,
                                    stride, in_channels)


        return output

    #dout的形状: (n, h_out, w_out, self.num_kernels)
    def backward(self, dout):
        dout = dout.astype(np.float32, copy=False)
        n, h_out, w_out, dout_num_kernels = dout.shape

        #初始化W的梯度，计算b的梯度
        self.grads['W'].fill(0)
        self.grads['b'] = np.sum(dout, axis=(0, 1, 2))
        #初始化din
        din = np.zeros_like(self.inputs, dtype=np.float32)

        ''' for i in range(n):
                for kernel in range(self.num_kernels):
                    for row in range(h_out):
                        for column in range(w_out):
                            #print("calculating...")
                            #这一段与前向相同
                            start_row = row * self.stride
                            start_column = column * self.stride
                            end_row = start_row + self.kernel_height
                            end_column = start_column + self.kernel_width
    
                            #感受野
                            receptive_field = self.inputs[i, start_row:end_row, start_column:end_column, :]
                            #当前卷积核
                            current_kernel = self.params['W'][:, :, :, kernel]
    
                            #这个位置的dl/dout
                            dout_val = dout[i, row, column, kernel]
    
                            #更新模型参数的梯度
                            #dl/dout * dout/dW
                            self.grads['W'][:, :, :, kernel] += dout_val * receptive_field
                            #dl/dout * 1
                            self.grads['b'][kernel] += dout_val
    
                            #dl/dout * dout/din
                            din[i, start_row:end_row, start_column:end_column, :] += dout_val * current_kernel'''
        kernel_height = self.kernel_height
        kernel_width = self.kernel_width
        stride = self.stride
        in_channels = self.in_channels

        for i in range(n):
            current_input = self.inputs[i]
            current_dout = dout[i]
            current_din = din[i]
            w_all_kernels_np = self.params['W']

            _conv_backward_numba(current_input,
                                 current_dout,
                                 w_all_kernels_np,
                                 self.grads['W'],
                                 current_din,
                                 h_out, w_out,
                                 kernel_height, kernel_width,
                                 stride, in_channels, dout_num_kernels)

        return din


#最大池化层
class MaxPoolingLayer(Layer):

    def __init__(self, pool_size, stride=None):
        super().__init__()
        self.pool_h, self.pool_w = pool_size
        if stride is None:
            self.stride = (self.pool_h, self.pool_w)
        else:
            self.stride = stride
        self.inputs = None
        #标记最大值的来源位置
        self.max_note = None

    def forward(self, inputs):
        self.inputs = inputs
        n, h_in, w_in, c_in = inputs.shape

        #输出的形状
        h_out = (h_in - self.pool_h) // self.stride[0] + 1
        w_out = (w_in - self.pool_w) // self.stride[1] + 1
        output = np.zeros((n, h_out, w_out, c_in))

        self.max_note = np.zeros_like(self.inputs)

        for i in range(n):
            for channel in range(c_in):
                for row in range(h_out):
                    for column in range(w_out):
                        start_row = row * self.stride[0]
                        start_column = column * self.stride[1]
                        end_row = start_row + self.pool_h
                        end_column = start_column + self.pool_w

                        pool = self.inputs[i, start_row:end_row, start_column:end_column, channel]
                        max_val = np.max(pool)
                        output[i, row, column, channel] = max_val

                        #填充max_note
                        max_index = (pool == max_val)
                        self.max_note[i, start_row:end_row, start_column:end_column, channel][max_index] = True

        return output

    def backward(self, dout):
        n, h_out, w_out, c_in = dout.shape
        din = np.zeros_like(self.inputs)

        for i in range(n):
            for channel in range(c_in):
                for row in range(h_out):
                    for column in range(w_out):
                        start_row = row * self.stride[0]
                        start_column = column * self.stride[1]
                        end_row = start_row + self.pool_h
                        end_column = start_column + self.pool_w

                        dout_val = dout[i, row, column, channel]

                        max_mask = self.max_note[i, start_row:end_row, start_column:end_column, channel]
                        din[i, start_row:end_row, start_column:end_column, channel] += dout_val * max_mask

        return din