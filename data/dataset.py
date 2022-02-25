import os.path

import torch
from torch.utils.data import Dataset
import numpy as np


class NERDataset(Dataset):
    def __init__(self, opt):

        # self.data_path = 'D:/pycode/torchtest/data/测试用.txt'
        self.data_path = opt.data_path + opt.train_file
        self.pad_size = opt.maxlen
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
        word2id = {}
        label2id = {}
        id2label = {}
        id2word = {}
        word2id['PAD'] = 0
        label2id['PAD'] = 0

        # 得到每一句的文本数据和标签序列
        data = []
        label = []
        for i in range(len(context)):
            sentence = context[i]
            tmp_data = []
            tmp_label = []
            for text in sentence:
                text = text.split(' ')
                if text[0] not in word2id.keys():
                    word2id[text[0]] = len(word2id)
                if text[-1] not in label2id.keys():
                    label2id[text[-1]] = len(label2id)
                tmp_data.append(text[0])
                tmp_label.append(text[-1])

            data.append(tmp_data)
            label.append(tmp_label)
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
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)
        np.savez_compressed(opt.model_path +'/test',
                            word2id=word2id,
                            id2word=id2word,
                            id2label=id2label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        # 将文本转换成数字
        word2id_list = []
        for word in data:
            word2id_list.append(self.word2id[word])

        label2id_list = []
        for word_label in label:
            label2id_list.append(self.label2id[word_label])

        # 补全到固定长度
        while len(word2id_list) < 128:
            word2id_list.append(self.word2id['PAD'])
        while len(label2id_list) < 128:
            label2id_list.append(self.label2id['PAD'])
        # 转换成tensor格式
        word2id_list = torch.tensor(word2id_list).view(-1)
        label2id_list = torch.tensor(label2id_list).view(-1)
        # label2id_list=torch.zeros(128, 17).scatter_(1, label2id_list, 1)
        return word2id_list, label2id_list

    def __len__(self):
        return len(self.data)

    def getvocab(self):
        return self.id2word, self.id2label
