import itertools, argparse, os, dill
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from data import SampleGenerator
from model import GMF, MLP, NeuMF

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', required=True, help='tuning or train')
parser.add_argument('--model', required=True, help='GMF, MLP or NeuMF')
parser.add_argument('--device', required=True, help='GPU id to use')
args = parser.parse_args()

# set GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] {DEVICE} activated')

# data path
base_path = os.path.expanduser('~/dataset/ml-1m/leave-one-out')
train_tuning_path = os.path.join(base_path, 'train_tuning.csv')
train_real_path = os.path.join(base_path, 'train_real.csv')
validset_path = os.path.join(base_path, 'validset.csv')
testset_path = os.path.join(base_path, 'testset.csv')

# data of ndarray
print(f'[INFO] data loading started')
train_tuning = pd.read_csv(train_tuning_path, header='None').values
train_real = pd.read_csv(train_real_path, header='None').values
validset = pd.read_csv(validset_path, header='None').values
testset = pd.read_csv(testset_path, header='None').values

# batch generator
train_tuning_generator = SampleGenerator(train_tuning)
train_real_generator = SampleGenerator(train_real)
print(f'[INFO] data loading finished')

user_num = 6040
item_num = 3706


def entry():
    mode = args.mode.lower()
    model = args.model.lower()

    if mode == 'tuning':
        if model == 'gmf':
            config = {
                'lr': [1e-4, 5e-4, 1e-3],
                'embed_dim': [128, 256],
                'neg_num': [4],
                'batch_size': [256]
            }
            tuning_GMF(config)
        elif model == 'mlp':
            pass
        elif model == 'neumf':
            pass
    elif mode == 'train':
        # TODO
        """
        1. get best hp
        2. train with train_real and save model
        3. test
        """
        pass


def train_epoch(model, optimizer, loader, epoch):
    for user, item, label in tqdm(loader, desc=f'epoch {epoch}'):
        user, item, label = user.to(DEVICE), item.to(DEVICE), label(DEVICE)

        scores = model(user, item)
        loss = torch.mean(F.binary_cross_entropy_with_logits(scores, label))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(model, optimizer, loader, epochs=10, verbose=True):
    """
    :return
        - best loss_val, acc_val
    """
    best_hr = -1
    best_loss = float('inf')
    model.train()

    for epoch in range(epochs):
        start = time()
        train_epoch(model, optimizer, loader, epoch)
        end = time()
        loss_val, hr_val = evaluate(model, validset)

        # TODO
        # verbose
        # update


def evaluate(model, data):
    model.eval()
    user = torch.LongTensor(data[:, 0]).to(DEVICE)
    item_all = torch.LongTensor(range(item_num)).to(DEVICE)
    label = torch.LongTensor(data[:, 1]).to(DEVICE)

    with torch.no_grad():
        scores = model(user, item_all)  # (user_num, item_num)
        # TODO
        # metric with label

    return loss, val


def save_model():
    pass


def save_hp():
    pass


def tuning_GMF(config):
    candidates = [config['lr'], config['embed_dim'], config['neg_num'], config['batch_size']]
    best_hp = None
    best_hr = -1
    best_model = None

    for hps in itertools.product(*candidates):
        lr, embed_dim, num_neg, batch_size = hps
        hp = {
            'model': {
                'user_num': user_num,
                'item_num': item_num,
                'embed_dim': embed_dim
            },
            'etc': {
                'lr': lr,
                'num_neg': num_neg,
                'batch_size': batch_size
            }
        }

        model = GMF(hp['model'])
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=hp['etc']['lr'])
        loader = train_tuning_generator.get_loader(hp['etc']['num_neg'], hp['etc']['batch_size'])
        loss_val, hr_val = train(model, optimizer, loader)

        if hr_val > best_hr:
            best_hr = hr_val
            best_hp = hp
            best_model = model


if __name__ == '__main__':
    entry()