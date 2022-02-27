import torch
from .Basic import BasicModel
class LstmModel(BasicModel):

    def __init__(self, vocab_size, embedding_dim):
        super(LstmModel, self).__init__()
        self.hidden_dim = 64
        self.embedding_dim = embedding_dim
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        self.linear1 = torch.nn.Linear(self.hidden_dim*2, 17)#默认隐藏层初始化使输出变成2倍

    def forward(self, input):
        seq_len, batch_size = input.size()
        embds = self.embeddings(input)
        # embds = torch.rand(seq_len, batch_size, self.embedding_dim)
        output, hidden = self.lstm(embds)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return output, hidden

