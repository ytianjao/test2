from config import Config
from data import NERDataset
import torch
from torch.utils.data import DataLoader
from models import DNNModel, LstmModel
import numpy as np

opt = Config()
dataset = NERDataset(opt)
id2word, id2label = dataset.getvocab()
vocab_size = len(id2word)
data_loader = DataLoader(dataset, batch_size=4)
# model = DNNModel()

'''
逻辑：训练集读取：保存字典表、word2id、id2word（非必须）、label2id、id2label
在训练时直接查询字典表构造Iter
测试集读取：读取训练集字典表，未登录词统一UNK或随机生成
'''


def read_data(data_path):
    word2id = {}
    label2id = {}
    word2id['PAD'] = 0
    word2id['UNK'] = 1
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
    np.savez(opt.model_path +'/dicts',
                        word2id=word2id,
                        label2id=label2id)
def data_loader():
    dataset = np.load(opt.model_path +'/dicts.npz', allow_pickle=True)
    word2id = dataset['word2id'].item()



def train():
    read_data(opt.data_path + opt.train_file)
    model = LstmModel(vocab_size, opt.embedding_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(2):
        for ii, (data, label) in enumerate(data_loader):

            x = data.transpose(1,0)#(4,128) -> (128,4)
            optimizer.zero_grad()

            # y = model(x)#(512,17)
            y = model(x)[0]#(512,17)

            label = label.transpose(1,0)#(4,128) ->(128,4)
            loss = criterion(y, label.reshape(-1))

            loss.backward()
            optimizer.step()
    model.save()

def test():
    for ii, (data, label) in enumerate(data_loader):
        model = LstmModel(vocab_size, opt.embedding_dim)
        model.load(opt.model_path)
        model.eval()
        x = data.transpose(1,0)
        # y = model(x)
        y = model(x)[0]
        y = y.reshape(128,4,17)
        y = y.transpose(1,0)
        y = torch.nn.Softmax(2)(y)

        for i in range(4):
            for ii in range(128):
                print(id2label[y[i][ii].topk(1)[1].item()])
# train()
# test()
read_data(opt.data_path + opt.train_file)



