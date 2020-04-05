import os, json, random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence as pad_sequence


seed = 143
random.seed(seed)
torch.manual_seed(seed)


class EcomDataset(Dataset):
    def __init__(self, users, items, behaviors, labels):
        self.users = users  # tensor
        self.items = items  # 2d list
        self.behaviors = behaviors  # 2d list
        self.labels = labels  # tensor

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.behaviors[idx], self.labels[idx]

    def __len__(self):
        return self.users.size(0)


class SampleGenerator:
    def __init__(self, data, meta, item_dict, is_test=False, model='mrn'):
        self.X_user = torch.LongTensor(data['X_user'])
        self.X_item = data['X_item']
        self.X_behavior = data['X_behavior']
        self.item_dict = item_dict
        self.sample_pos = torch.LongTensor(data['y'])
        self.user_num = meta['user_num']
        self.item_num = meta['item_num']  # 0 is for padding
        self.is_test = is_test
        self.model = model

    def get_loader(self, num_neg=4, batch_size=2048):
        """ Loader for an epoch wih negative sampled data
        The loader returns user, item, behavior, sample, mask_len
        """

        num_item = self.item_num
        sample_neg = []
        if self.model == 'mrn':
            collate_fn = custom_collate_mrn
        elif self.model =='lstm':
            collate_fn = custom_collate_lstm

        if self.is_test:
            dataset = EcomDataset(self.X_user, self.X_item, self.X_behavior, self.sample_pos)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

            return loader

        for user in self.X_user:
            exceptions = self.item_dict[user.item()]
            tmp = []
            for i in range(num_neg):
                sample = int(num_item * random.random()) + 1
                while sample in exceptions:
                    sample = int(num_item * random.random()) + 1
                tmp.append(sample)
            sample_neg.append(tmp)
        sample_neg = torch.LongTensor(sample_neg).view(-1, num_neg)
        samples = torch.cat([self.sample_pos.view(-1, 1), sample_neg], dim=1)
        dataset = EcomDataset(self.X_user, self.X_item, self.X_behavior, samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

        return loader


def read_data():
    version = 'seq_v6_extreme_nodup'
    base_path = os.path.expanduser('~/dataset/ecom_big')

    with open(os.path.join(base_path, f'{version}/test_X_item.json'), 'r') as f:
        test_X_item = json.load(f)
    with open(os.path.join(base_path, f'{version}/test_X_behavior.json'), 'r') as f:
        test_X_behavior = json.load(f)
    with open(os.path.join(base_path, f'{version}/train_item.json'), 'r') as f:
        train_item = json.load(f, object_hook=lambda d: {int(k): {int(i) for i in v} for k, v in d.items()})
    with open(os.path.join(base_path, f'{version}/train_X_item.json'), 'r') as f:
        train_X_item = json.load(f)
    with open(os.path.join(base_path, f'{version}/train_X_behavior.json'), 'r') as f:
        train_X_behavior = json.load(f)
    trainset = pd.read_csv(os.path.join(base_path, f'{version}/trainset.csv'))
    testset = pd.read_csv(os.path.join(base_path, f'{version}/testset.csv'))
    train_X_user = trainset['user'].tolist()
    train_y = trainset['y'].tolist()
    test_X_user = testset['user'].tolist()
    test_y = testset['y'].tolist()

    data = {
        'train': {
            'X_user': train_X_user,
            'X_item': train_X_item,
            'X_behavior': train_X_behavior,
            'y': train_y
        },

        'test': {
            'X_user': test_X_user,
            'X_item': test_X_item,
            'X_behavior': test_X_behavior,
            'y': test_y
        },

        'meta': {
            'item_num': 9055,  # id is 1~9055. 0 is left for padding.
            'user_num': 23699,
            'behavior_num': 3
        },

        'train_item': train_item,
    }

    return data


def custom_collate_mrn(batch):
    """ returns user, item, behavior, sample, mask_len """
    batch = sorted(batch, key=lambda x: -len(x[1]))
    transposed = zip(*batch)

    mask_len = None
    ls = []
    for i, samples in enumerate(transposed):
        # user
        if i == 0:
            ls.append(torch.LongTensor(samples))
        # item
        elif i == 1:
            samples = [torch.LongTensor(seq) for seq in samples]
            samples_pad = pad_sequence(samples, batch_first=True)
            mask = samples_pad.type(torch.bool)
            mask_len = torch.sum(mask, dim=0)
            ls.append(samples_pad)
        # behavior
        elif i == 2:
            samples = [torch.LongTensor(seq) for seq in samples]
            samples_pad = pad_sequence(samples, batch_first=True)
            ls.append(samples_pad)
        # sample
        elif i == 3:
            ls.append(torch.stack(samples))
    ls.append(mask_len)

    return ls


def custom_collate_lstm(batch):
    """ returns user, item, (behavior), sample, sample_len """
    batch = sorted(batch, key=lambda x: -len(x[1]))
    transposed = zip(*batch)

    sample_len = None
    ls = []
    for i, samples in enumerate(transposed):
        # user
        if i == 0:
            ls.append(torch.LongTensor(samples))
        # item
        elif i == 1:
            samples = [torch.LongTensor(seq) for seq in samples]
            samples_pad = pad_sequence(samples, batch_first=True)
            sample_len = [len(s) for s in samples]
            ls.append(samples_pad)
        # behavior
        elif i == 2:
            ls.append(None)
        # sample
        elif i == 3:
            ls.append(torch.stack(samples))
    ls.append(sample_len)

    return ls