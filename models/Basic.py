import torch
import time
import os
class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.model_name = self.__class__.__name__

    def load(self, model_path):
        list_dir = os.listdir(model_path)
        if len(list_dir) != 0:

            for file in list_dir:
                if file.endswith('.pth'):
                    self.load_state_dict(torch.load(model_path + file))


    def save(self, model_path, name=None):
        if name is None:
            prefix = model_path+self.model_name + '-'
            name = time.strftime(prefix + '%m%d-%H-%M-%S.pth')
        # print(os.listdir('./checkpoints/'))
        list_dir = os.listdir(model_path)
        for file in list_dir:
            if file.endswith('.pth'):
                os.remove(model_path + file)
        torch.save(self.state_dict(), name)