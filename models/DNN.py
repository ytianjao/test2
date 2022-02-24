
import torch.nn as nn

from .Basic import BasicModel


class DNNModel(BasicModel):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.embd = nn.Embedding(226,32)

        self.classify = nn.Sequential(
            nn.Linear(32,100),
            nn.ReLU(),
            nn.Linear(100,17)
        )

    def forward(self, x):
        # x = x.transpose(1,0)
        x = self.embd(x)#(128,4,32)
        x = self.classify(x)#(128,4,17)

        x = x.view(128*4,-1)#(128,4,17) -> (128*4, 17)
        return x