import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from src.model import NeuralNetwork
from src.optimizer import SGD, ExponentialScheduler
from src.data_utils import load_cifar10_data, BatchIterator

def train(model, train_data, val_data, optimizer, batch_size=16, num_epoches=10, seed=64,
          model_path='./model', decay_rate=0.9):
    #设置随机数种子
    np.random.seed(seed)

    train_labels, train_images = train_data
    train_labels = train_labels[:100]
    train_images = train_images[:100]
    val_labels, val_images = val_data

    #为训练数据和验证数据分别设置批处理迭代器
    train_batchiterator = BatchIterator(train_labels, train_images, batch_size=batch_size, seed=seed, shuffle=True)
    val_batchiterator = BatchIterator(val_labels, val_images, batch_size=batch_size, seed=seed, shuffle=False)

    # 初始化学习率
    initial_lr = optimizer.learning_rate
    # 创建学习率调度器实例
    # 让学习率在每个epoch衰减一次
    scheduler = ExponentialScheduler(initial_lr=initial_lr, stage_length=train_batchiterator.num_batches,
                                     decay_rate=decay_rate)

    #训练的历史记录，用来进行训练过程可视化
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    #初始化训练步数
    total_iter = 0

    for epoch in range(num_epoches):
        #为每个epoch重置计时器
        epoch_start_time = time.time()
        total_loss = 0
        total_acc = 0
        batch_count = 0
        sample_count = 0

        #设置为训练模式
        model.train()

        #训练一个epoch
        for batch_labels, batch_images in train_batchiterator:
            #递增训练步数
            total_iter += 1

            # 更新学习率
            optimizer.learning_rate = scheduler(total_iter)

            # 每个batch前清零梯度
            model.zero_grad()

            # 前向传播
            y_pred = model.forward(batch_images)

            # 计算损失
            loss = model.loss(y_pred, batch_labels)

            # 反向传播
            model.backward(y_pred, batch_labels)

            # 更新参数
            optimizer.update(model)

            #比对预测准确率
            pred_class = np.argmax(y_pred, axis=1)
            true_class = np.argmax(batch_labels, axis=1)
            acc = np.mean(pred_class == true_class)

            #防止最后一个batch不全
            current_batch_size = len(batch_images)

            total_loss += current_batch_size * loss
            total_acc += current_batch_size * acc
            batch_count += 1
            sample_count += current_batch_size

        #计算平均损失和准确度
        avg_loss = total_loss / sample_count
        avg_acc = total_acc / sample_count

        #用验证集测试性能
        val_loss, val_acc = evaluate(model, val_batchiterator)

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        #记录这一epoch训练耗时
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch{epoch+1}   time: {epoch_time:.2f}   train_loss: {avg_loss:.3f}   "
              f"train_acc: {avg_acc:.3f}   val_loss: {val_loss:.3f}   val_acc: {val_acc:.3f}")


    print("training completed.")

    #保存模型
    if os.path.exists(model_path):
        model_file_path = os.path.join(model_path, 'model.pkl')
        model.save_model(model_file_path)

    #可视化训练过程
    plot_training_history(history, model_path=model_path)

    return history

def evaluate(model, iterator):
    #设置为验证模式
    model.val()

    total_loss = 0
    total_acc = 0
    sample_count = 0

    # 使用no_grad上下文管理器禁用梯度计算
    with model.no_grad():
        for batch_labels, batch_images in iterator:
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

            # 累计加权损失和准确率
            total_loss += loss * current_batch_size
            total_acc += acc * current_batch_size
            sample_count += current_batch_size

    # 计算平均损失和准确率
    avg_loss = total_loss / sample_count
    avg_acc = total_acc / sample_count

    return avg_loss, avg_acc

#训练过程可视化函数
def plot_training_history(history, model_path='./model'):
    #定义图像尺寸
    plt.figure(figsize=(12, 5))

    #子图1
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['val_loss'], 'g-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    #子图2
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(history['val_acc'], 'g-', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    # 保存图像
    plt.savefig(os.path.join(model_path, 'training_history.png'))
    plt.close()


def main():
    print("Start training...")
    #加载数据
    (train_labels, train_images), (val_labels, val_images), (
    test_labels_processed, test_images_processed) = load_cifar10_data(train_path='./data/cifar10/train',
                                                                      test_path='./data/cifar10/test')
    print("Data loaded successfully...")
    #创建模型
    input_size = (32, 32, 3)
    hidden_size = 128
    output_class = 10
    dropout_rate = 0.3
    model = NeuralNetwork(input_size, hidden_size, output_class, dropout_rate)
    #创建优化器
    optimizer = SGD()

    #保存模型的结构，以便在test.py中以正确形式导入参数
    model_config = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_class': output_class,
        'dropout_rate': dropout_rate
    }
    config_path = os.path.join('./model', 'model_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(model_config, f)

    #开始训练
    history = train(
        model = model,
        train_data = (train_labels, train_images),
        val_data = (val_labels, val_images),
        optimizer = optimizer,
    )

    #保存训练历史
    history_path = os.path.join('./model', 'history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    main()
