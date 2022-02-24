from data import NERDataset
import torch
from torch.utils.data import DataLoader
from models import DNNModel, LstmModel

dataset = NERDataset()
id2word, id2label = dataset.getvocab()
data_loader = DataLoader(dataset, batch_size=4)
# model = DNNModel()


def train():
    model = LstmModel(226, 32, 64)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(20):
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
        model = LstmModel(226, 32, 64)
        model.load("./checkpoints/")
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
train()
test()



