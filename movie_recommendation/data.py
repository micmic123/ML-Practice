import os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

path_base = os.path.expanduser('~/dataset/ml-1m')
if not os.path.isdir(path_base):
    print(f'[ERROR] please put ml-1m_pre.npz in {path_base}')
seed = 5
random.seed(seed)
torch.manual_seed(seed)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] {DEVICE} activated')


def get_data_dict():
    data_np = np.load(path_base, allow_pickle=True)
    data = dict()
    data['train_x'] = data_np['train_x']
    data['train_y_pos'] = data_np['train_y_pos']
    data['train_y_neg'] = data_np['train_y_neg']
    data['train_user'] = data_np['train_user']
    data['movie'] = data_np['movie'].all()
    data['test_x'] = data_np['test_x']
    data['text_y'] = data_np['test_y']
    data['test_user'] = data_np['test_user']

    data_np.close()

    return data


# ml-1m Dataset
# refer to https://grouplens.org/datasets/movielens/1m/
class MovieDataset(Dataset):
    def __init__(self, data, type):
        self.type = type

        if self.type == 'train':
            self.x = torch.from_numpy(data['train_x']).type(torch.int32)
            self.y_pos = torch.from_numpy(data['train_y_pos']).type(torch.int32)
            self.y_neg = torch.from_numpy(data['train_y_neg']).type(torch.int32)
            self.user = torch.from_numpy(data['train_user']).type(torch.int32)
        elif self.type == 'test':
            self.x = torch.from_numpy(data['test_x']).type(torch.int32)
            self.y = torch.from_numpy(data['test_y']).type(torch.int32)
            self.user = torch.from_numpy(data['test_user']).type(torch.int32)

    def __getitem__(self, index):
        if self.type == 'train':
            return self.x[index], self.y_pos[index], self.y_neg[index], self.user[index]
        elif self.type == 'test':
            return self.x[index], self.y[index], self.user[index]

    def __len__(self):
        return len(self.x)


def get_loader(batch_size):
    data = get_data_dict()
    trainset = MovieDataset(data, 'train')
    testset = MovieDataset(data, 'test')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=len(testset), shuffle=True, num_workers=1)
    item_size = len(data['movie'])
    user_size = len(data['test_user'])

    return train_loader, test_loader, item_size, user_size
