import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


seed = 143
random.seed(seed)
torch.manual_seed(seed)


class MovieDataset(Dataset):
    def __init__(self, users, items, labels):
        self.users = users
        self.items = items
        self.labels = labels

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def __len__(self):
        return self.users.size(0)


class TestDataset(Dataset):
    def __init__(self, users, items):
        self.users = users
        self.items = items

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

    def __len__(self):
        return self.users.size(0)


class SampleGenerator:
    def __init__(self, data, num_user=6040, num_item=3706):
        """
        :param
            - data: ndarray of [user, item]
        """
        self.data = data
        self.num_user = num_user
        self.num_item = num_item
        self.ma = {tuple(line) for line in data}

    def get_loader(self, num_neg, batch_size):
        """ Loader for an epoch wih negative sampled data """
        users, items, labels = [], [], []
        ma = self.ma
        num_item = self.num_item

        for user, item in self.data:
            users.append(user)
            items.append(item)
            labels.append(1)

            # negative sampling
            for i in range(num_neg):
                sample = int(num_item * random.random())
                while (user, sample) in ma:
                    sample = int(num_item * random.random())
                users.append(user)
                items.append(sample)
                labels.append(0)
        dataset = MovieDataset(torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return loader


class TestGenerator:
    def __init__(self, data):
        users = data[:, 0]
        items = data[:, 1]
        self.dataset = TestDataset(torch.LongTensor(users), items)

    def get_loader(self):
        return DataLoader(self.dataset, batch_size=128, shuffle=False)
