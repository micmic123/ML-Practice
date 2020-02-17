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
    data_np = np.load(os.path.join(path_base, 'ml-1m_pre.npz'), allow_pickle=True)
    data = dict()
    data['train_x'] = data_np['train_x']
    data['train_y_pos'] = data_np['train_y_pos']
    data['train_y_neg'] = data_np['train_y_neg']
    data['train_user'] = data_np['train_user']
    data['movie'] = data_np['movie'].all()
    data['test_x'] = data_np['test_x']
    data['test_y'] = data_np['test_y']
    data['test_user'] = data_np['test_user']

    data_np.close()

    return data


# ml-1m Dataset
# refer to https://grouplens.org/datasets/movielens/1m/
class MovieDataset(Dataset):
    def __init__(self, data):
        self.x = torch.from_numpy(data['train_x']).type(torch.long)
        self.y_pos = torch.from_numpy(data['train_y_pos']).type(torch.long)
        self.y_neg = torch.from_numpy(data['train_y_neg']).type(torch.long)
        self.user = torch.from_numpy(data['train_user']).type(torch.long)

    def __getitem__(self, index):
        return self.x[index], self.y_pos[index], self.y_neg[index], self.user[index]

    def __len__(self):
        return len(self.x)


def get_data(batch_size):
    data = get_data_dict()
    trainset = MovieDataset(data)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = get_testset(data)
    item_size = len(data['movie'])
    user_size = len(data['test_user'])

    return train_loader, testset, item_size, user_size


def get_testset(data):
    x = torch.from_numpy(data['test_x']).type(torch.long)
    y = data['test_y']  # ndarray of numpy.object
    user = torch.from_numpy(data['test_user']).type(torch.long)

    return x, y, user