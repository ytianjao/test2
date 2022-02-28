import os.path

import torch
from torch.utils.data import Dataset
import numpy as np


class NERDataset(Dataset):
    def __init__(self, opt, train = False,val = False, test = False):

        if train:
            self.data_path = opt.data_path + opt.train_file
        elif test:
            self.data_path = opt.data_path + opt.test_file
        elif val:
            self.data_path = opt.data_path + opt.val_file
        else:
            raise Exception("dataset参数中至少有一个是True")
        self.pad_size = opt.maxlen

        #读取数据集，以句子为单位
        f_r = open(self.data_path, 'r', encoding='utf-8')
        lines = f_r.readlines()
        context = []
        sentence = []
        for line in lines:
            line = line.strip()
            # 以‘。’后的换行符为单位分割文件
            if len(line) == 0 and len(sentence) != 0:
                context.append(sentence)
                sentence = []
                continue
            if len(line) == 0:
                continue
            sentence.append(line)
        context.append(sentence)

        #       文字转数字工具
        word2id, label2id = self.load_data(opt)#训练和测试过程都使用训练集的word2id，测试集中的未登录词在后面处理
        id2word, id2label = {}, {}

        # 得到每一句的汉字序列和标签序列
        data = []
        label = []
        for i in range(len(context)):
            sentence = context[i]
            tmp_data = []
            tmp_label = []
            for text in sentence:
                text = text.split(' ')
                tmp_data.append(text[0])
                tmp_label.append(text[-1])

            data.append(tmp_data)
            label.append(tmp_label)#到这一步，保存的仍是汉字，不是id

        #工具，方便预测值转汉字
        for key, value in word2id.items():
            id2word[value] = key
        for key, value in label2id.items():
            id2label[value] = key


        self.data = data
        self.label = label
        self.word2id = word2id
        self.label2id = label2id
        self.id2word = id2word
        self.id2label = id2label



    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        # 将文本转换成数字
        word2id_list = []
        for word in data:
            if word in self.word2id.keys():
                word2id_list.append(self.word2id[word])
            else:
                word2id_list.append(self.word2id['UNK'])#处理未登录词

        label2id_list = []
        for word_label in label:
            if word_label in self.label2id.keys():
                label2id_list.append(self.label2id[word_label])
            else:
                raise Exception('训练集没有当前标签，请调整训练集')


        # 补全到固定长度
        while len(word2id_list) < 128:
            word2id_list.append(self.word2id['PAD'])
        while len(label2id_list) < 128:
            label2id_list.append(self.label2id['O'])
        # 转换成tensor格式
        word2id_list = torch.tensor(word2id_list).view(-1)
        label2id_list = torch.tensor(label2id_list).view(-1)
        return word2id_list, label2id_list

    def __len__(self):
        return len(self.data)

    def getvocab(self):
        return self.id2word, self.id2label

    def load_data(self, opt):
        dataset = np.load(opt.model_path +'dicts.npz', allow_pickle=True)
        word2id = dataset['word2id'].item()
        label2id = dataset['label2id'].item()
        return word2id, label2id
