import torch
from .Basic import BasicModel
class LstmModel(BasicModel):

    def __init__(self, vocab_size, embedding_dim, label_size):
        super(LstmModel, self).__init__()
        self.hidden_dim = 64
        self.embedding_dim = embedding_dim
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        self.linear1 = torch.nn.Linear(self.hidden_dim*2, label_size)#默认隐藏层初始化使输出变成2倍

    def forward(self, input):
        input = input.transpose(1,0)#(batch_size,seq_len) -> (seq_len,batch_size)
        seq_len, batch_size = input.size()
        embds = self.embeddings(input)#(seq_len,batch_size, embdding_dim)
        output, hidden = self.lstm(embds)#output:(seq_len,batch_size, hidden_dim*2)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return output, hidden

