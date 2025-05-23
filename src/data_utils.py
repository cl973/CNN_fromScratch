import numpy as np
import pandas as pd
import os

#将一个batch的bin文件加载进来
def load_single_batch(filepath):
    #规定每个图片的数据大小
    num_row = 32
    num_column = 32
    num_channel = 3
    label_length = 1
    record_length = num_row * num_column * num_channel + label_length

    with open(filepath,'rb') as f:
        data=f.read()

    #计算读取到的图像总数，并检查数据集是否完好
    record_num = len(data) // record_length
    if len(data) % record_length != 0:
        raise ValueError(f"Data in {filepath} may be damaged!")
    #将读取到的数据转化为numpy数组
    raw_data = np.frombuffer(data, dtype=np.uint8)
    #reshape为以图像编号为行，以图像数据位为列的矩阵
    records = raw_data.reshape(record_num, record_length)

    #将labels和images分装到两个矩阵
    #读取所有行的第1位
    labels = records[:, 0]
    #读取所有行的后3072位
    images_flat = records[:, 1:]

    #由于数据集中每样本的存储方式是RRRR...GGGG...BBBB...所以reshape时第二维度为num_channel
    images_reshaped = images_flat.reshape(record_num, num_channel, num_row, num_column)
    #一般num_channel应放在最后一维
    images = images_reshaped.transpose(0, 2, 3, 1)

    return labels, images

#对数据进行预处理，包括uint8数据转格式，像素值归一化，标签转化为独热编码
def preprocess(labels, images):
    #将images中的uint8数据转为float32，为归一化做准备
    float32_images = images.astype(np.float32)
    #将像素值归一化到0-1之间
    normalized_images = float32_images / 255.0

    #将labels转化为独热编码
    #将labels展开成一维数组
    flattened_labels = labels.flatten()
    series_labels = pd.Series(flattened_labels, name='labels')
    #独热编码
    one_hot_labels = pd.get_dummies(series_labels, dtype=np.uint8)
    #转回numpy数组
    one_hot_numpy_labels = one_hot_labels.to_numpy()

    return normalized_images, one_hot_numpy_labels

#加载所有数据，方便以后自己灵活地抽取小批量进行训练，也方便划分出验证集来监控训练效果
def load_all_batches(train_path, test_path):
    all_train_labels = []
    all_train_images = []

    for i in range(1,6):
        train_batch_path = os.path.join(train_path, f'data_batch_{i}.bin')
        labels, images = load_single_batch(train_batch_path)
        all_train_labels.append(labels)
        all_train_images.append(images)

    full_train_images = np.concatenate(all_train_images, axis=0)
    full_train_labels = np.concatenate(all_train_labels, axis=0)

    test_batch_path = os.path.join(test_path, 'test_batch.bin')
    test_labels, test_images = load_single_batch(test_batch_path)

    return (full_train_labels, full_train_images), (test_labels, test_images)

#封装的顶层函数
def load_cifar10_data(train_path, test_path, validation_size=0.1, seed=64):
    #为随机数设置种子
    np.random.seed(seed)

    (train_labels_raw, train_images_raw), (test_labels_raw, test_images_raw) = load_all_batches(train_path, test_path)
    train_images_processed, train_labels_processed = preprocess(train_labels_raw, train_images_raw)
    test_images_processed, test_labels_processed = preprocess(test_labels_raw, test_images_raw)

    #划分验证集
    train_labels, train_images = train_labels_processed, train_images_processed #默认为原来预处理好的所有训练集样本
    val_labels, val_images = np.array([]), np.array([]) #默认为0
    if validation_size > 0 and validation_size < 1:
        #获取训练集中的样本总数
        num_train_samples = train_images_processed.shape[0]
        #获取一个打乱过的索引数组
        shuffled_index = np.random.permutation(num_train_samples)
        #确定验证集大小
        split_index = int(num_train_samples * (1-validation_size)) #取整

        train_index, val_index = shuffled_index[:split_index], shuffled_index[split_index:]
        train_labels, train_images = train_labels_processed[train_index], train_images_processed[train_index]
        val_labels, val_images = train_labels_processed[val_index], train_images_processed[val_index]

    return (train_labels, train_images), (val_labels, val_images), (test_labels_processed, test_images_processed)


class BatchIterator:

    #初始化批处理迭代器
    def __init__(self, labels, images, batch_size=32, seed=64, shuffle=True):
        self.labels = labels
        self.images = images
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.ran_generator = np.random.RandomState(seed) #创建于一个独立于全局状态的随机数生成器
        self.num_samples = images.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / batch_size)) #对批次数量向上取整

        self.reset()

    #重置迭代器的计数器并打乱数据
    def reset(self):
        self.current_batch = 0

        if self.shuffle:
            shuffled_index = self.ran_generator.permutation(self.num_samples)
            self.labels = self.labels[shuffled_index]
            self.images = self.images[shuffled_index]

    #返回迭代器对象自己
    def __iter__(self):
        return self

    #获取下一batch的数据
    def __next__(self):
        #当前epoch的数据已经全部迭代完毕
        if self.current_batch >= self.num_batches:
            self.reset()
            #抛出StopIteration异常，通知迭代器所处的for循环结束当前周期
            raise StopIteration

        #当前所需的batch的起始与结束索引
        start_index = self.current_batch * self.batch_size
        end_index = min(start_index + self.batch_size, self.num_samples) #最后一个批次可能不完整

        #提取出数据
        batch_labels = self.labels[start_index:end_index]
        batch_images = self.images[start_index:end_index]

        self.current_batch += 1

        return batch_labels, batch_images