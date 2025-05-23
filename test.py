import numpy as np
import os
import pickle
from src.model import NeuralNetwork
from src.data_utils import load_cifar10_data, BatchIterator

def load_model_and_test(model_weight_path, model_config_path, test_data, batch_size=32):

    #检查模型是否存在
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"The model doesn't exist: {model_weight_path}")

    #加载测试数据
    test_labels, test_images = test_data

    if os.path.exists(model_config_path):
        with open(model_config_path, 'rb') as f:
            model_config = pickle.load(f)
    else:
        raise FileNotFoundError(f"The model config doesn't exist: {model_config_path}")

    #创建模型
    model = NeuralNetwork(
        input_shape=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        output_class=model_config['output_class'],
        dropout_rate=model_config['dropout_rate']
    )
    #加载参数
    model.load_model(model_weight_path)

    #创建测试批处理迭代器
    test_batchiterator = BatchIterator(test_labels, test_images, batch_size=batch_size, shuffle=False)

    #设置为验证模式
    model.val()

    total_acc = 0
    sample_count = 0

    # 使用no_grad上下文管理器禁用梯度计算
    with model.no_grad():
        for batch_labels, batch_images in test_batchiterator:
            # 前向传播
            y_pred = model.forward(batch_images)

            # 计算损失
            loss = model.loss(y_pred, batch_labels)

            # 计算准确率
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(batch_labels, axis=1)
            acc = np.mean(pred_classes == true_classes)

            # 获取当前批次的样本数
            current_batch_size = len(batch_images)

            # 累计加权准确率
            total_acc += acc * current_batch_size
            sample_count += current_batch_size

    #平均准确率
    acc = total_acc / sample_count

    return acc

def main():
    #加载测试数据
    _, _, (test_labels, test_images) = load_cifar10_data(train_path='./data/cifar10/train',
                                                                      test_path='./data/cifar10/test')
    print('Start testing...')
    test_acc = load_model_and_test('./model/model.pkl', './model/model_config.pkl', (test_labels, test_images))
    print(f'The accuracy of the test is {test_acc:.3f}')


if __name__ == "__main__":
    main()