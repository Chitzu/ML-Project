import json

import torch.utils.data
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
# ##########
# file = r'C:\Users\Chitzu\Desktop\test\data_batch_1'
#
# data_batch_1 = unpickle(file)
#
# # print(data_batch_1)
#
# data = data_batch_1[b'data']
# data = data.reshape(len(data),3,32,32).transpose(0,2,3,1)
#############
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        # It must be implemented
        self.data = []

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__

###########
# config = json.load(open(r'C:\Users\Chitzu\Desktop\test\config.json'))
# config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
# obj = BaseDataset(config)
# print(obj.__getitem__(2))
# print(obj.__len__())
# print(obj.__repr__())
