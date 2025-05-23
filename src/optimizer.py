#优化器基类
class Optimizer:

    def __init__(self):
        pass

    #更新模型参数
    def update(self, model):
        pass


#随机梯度下降
class SGD(Optimizer):

    def __init__(self, learning_rate=0.001):
        super().__init__()

        self.learning_rate = learning_rate

    def update(self, model):
        #遍历模型的每一层
        for layer_name, layer in model.layers.items():
            if hasattr(layer, 'params') and layer.params and hasattr(layer, 'grads') and layer.grads:
                # 层的参数'W'和'b'
                for param_name in layer.params:
                    # 如果该参数有梯度
                    if param_name in layer.grads:
                        layer.params[param_name] -= self.learning_rate * layer.grads[param_name]



#指数衰减学习率调度器，每学习一定步数就下调学习率
class ExponentialScheduler:

    #initial_lr: 起始学习率
    #stage_length: 每次调整间隔的学习步数
    #decay_rate: 每次调整的衰减率
    def __init__(self, initial_lr=0.001, stage_length=500, decay_rate=0.9):
        self.initial_lr = initial_lr
        self.stage_length = stage_length
        self.decay_rate = decay_rate

    #调用调度器
    #step: 当前训练步数
    def __call__(self, step):

        #当前在第几个学习阶段
        current_stage = step // self.stage_length

        return self.initial_lr * (self.decay_rate ** current_stage)
