from config import Config
from data import NERDataset
import torch
from torch.utils.data import DataLoader
from models import DNNModel, LstmModel
import numpy as np
import pickle
import os


def read_train_data(data_path):
    """
    在程序最开始执行，负责读取训练集，并保存word2id、label2id
    保存为npz和pickle两种格式
    :param data_path:
    :return:
    """
    word2id = {}
    label2id = {}
    word2id['PAD'] = 0
    word2id['UNK'] = 1
    # label2id['PAD'] = 0
    f_r = open(data_path, 'r', encoding='utf-8')
    for line in f_r.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        else:
            if line.split(' ')[0] not in word2id.keys():
                word2id[line.split(' ')[0]] = len(word2id)
            if line.split(' ')[-1] not in label2id.keys():
                label2id[line.split(' ')[-1]] = len(label2id)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    np.savez(opt.model_path +'dicts',
             word2id=word2id,
             label2id=label2id)
    with open(opt.model_path+'label2id.pkl', 'wb') as f:
        pickle.dump(label2id, f)
        f.close()

    with open(opt.model_path+'word2id.pkl', 'wb') as f:
        pickle.dump(word2id, f)
        f.close()

def train(opt):
    """
    训练函数，执行流程为：读取训练集-》模型初始化-》按批次训练
    :return:
    """

    read_train_data(opt.data_path + opt.train_file)#只在训练过程中保存训练集的word2id充当字典
    dataset = NERDataset(opt, train=True)#读取训练集，以句子为单位读取数据，并利用word2id将汉字转化为id，细节写在源码注释中
    id2word, id2label = dataset.getvocab()#工具函数，备用
    vocab_size = len(id2word)
    label_size = len(id2label)#获得字典大小和标签种类数，下一步模型初始化使用
    train_data_loader = DataLoader(dataset, batch_size=opt.batch_size)#将dataset包装


    model = LstmModel(vocab_size, opt.embedding_dim, label_size)#模型初始化
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    for epoch in range(opt.epoch):
        for batch, (data, label) in enumerate(train_data_loader):
            x = data.transpose(1,0)#(batch_size,seq_len) -> (seq_len,batch_size)
            optimizer.zero_grad()
            y = model(x)[0]#(batch_size*seq_len,label_size)
            label = label.transpose(1,0)#(batch_size,seq_len) -> (seq_len,batch_size)
            loss = criterion(y, label.reshape(-1))

            loss.backward()
            optimizer.step()
    model.save(opt.model_path)

def test(opt):
    dataset = NERDataset(opt, test=True)
    test_data_loader = DataLoader(dataset, batch_size=opt.batch_size)
    id2word, id2label = dataset.getvocab()
    vocab_size = len(id2word)
    label_size = len(id2label)

    for batch, (data, label) in enumerate(test_data_loader):
        batch_size, seq_len = data.size()
        model = LstmModel(vocab_size, opt.embedding_dim, label_size)
        model.load(opt.model_path)
        model.eval()
        x = data.transpose(1,0)
        y = model(x)[0]
        y = y.reshape(seq_len,batch_size,label_size)
        y = y.transpose(1,0)
        y = torch.nn.Softmax(2)(y)

        for i in range(batch_size):
            for ii in range(seq_len):
                print(id2label[y[i][ii].topk(1)[1].item()])



if __name__ == '__main__':
    opt = Config()
    read_train_data(opt.data_path + opt.train_file)
    if opt.train:
        train(opt)
    if opt.test:
        test(opt)




