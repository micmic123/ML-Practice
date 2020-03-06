import itertools, argparse, os, json
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from data import SampleGenerator, TestGenerator
from model import GMF, MLP, NeuMF
from utils import eval


parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', required=True, help='train or supplement')
parser.add_argument('--model', required=True, help='GMF, MLP or NeuMF')
parser.add_argument('--device', required=True, help='GPU id to use')
parser.add_argument('--target', required=False, help='target model to supplement')
parser.add_argument('--epochs', required=False, help='epochs to supplement')
args = parser.parse_args()
print(args)

# set GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] {DEVICE} activated')

# data path
base_path = os.path.expanduser('~/dataset/ml-1m/leave-one-out')
# train_tuning_path = os.path.join(base_path, 'train_tuning.csv')
train_real_path = os.path.join(base_path, 'train_real.csv')
train_real_dict_path = os.path.join(base_path, 'train_real_dict.json')
# validset_path = os.path.join(base_path, 'validset.csv')
testset_path = os.path.join(base_path, 'testset.csv')
model_path = './snapshot'

# data of ndarray
print(f'[INFO] data loading started')
# train_tuning = pd.read_csv(train_tuning_path, header=None).values
train_real = pd.read_csv(train_real_path, header=None).values
# validset = pd.read_csv(validset_path, header=None).values
testset = pd.read_csv(testset_path, header=None).values
with open(train_real_dict_path, 'r', encoding='utf-8') as f:
    train_real_dict = json.load(f, object_hook=lambda d: {int(k): {int(i) for i in v} for k, v in d.items()})

# batch generator
# train_tuning_generator = SampleGenerator(train_tuning)
train_real_generator = SampleGenerator(train_real)
test_loader = TestGenerator(testset).get_loader()
print(f'[INFO] data loading finished')

# etc
user_num = 6040
item_num = 3706


def entry():
    mode = args.mode.lower()
    model = args.model.lower()

    # tuning model with train_real and save snapshot of the best model
    if mode == 'train':
        if model == 'gmf':
            config = {
                'lr': [1e-3],  # [1e-4, 5e-4, 1e-3],
                'embed_dim': [4, 8, 16],
                'neg_num': [4],
                'batch_size': [512, 1024],
                'epochs': 100
            }
            tuning_GMF(config)
        elif model == 'mlp':
            pass
        elif model == 'neumf':
            pass
    # load the best model and run test with testset
    elif mode == 'supplement':
        model = args.target
        epochs = args.epochs
        # TODO
        """
        1. load the best model
        2. test and save result
        """
        pass


def train_epoch(model, optimizer, loader, epoch):
    """ train for one epoch and return average loss """
    loss_ls = []

    for user, item, label in tqdm(loader, desc=f'epoch {epoch}'):
        user, item, label = user.to(DEVICE), item.to(DEVICE), label.to(DEVICE)

        scores = model(user, item)
        loss = F.binary_cross_entropy_with_logits(scores, label)  # default reduction='mean'
        loss_ls.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(loss_ls) / len(loss_ls)


def train(model, optimizer, loader, epochs=10, verbose=True):
    """  train the given model

    :return
        - best HR@k, loss
    """
    best_HR = -1
    best_loss = float('inf')  # loss for one epoch with training set
    model.train()

    for epoch in range(epochs):
        start = time()
        loss = train_epoch(model, optimizer, loader, epoch)
        end = time()
        HR, NDCG = evaluate(model, test_loader, train_real_dict)

        if HR > best_HR:
            best_HR = HR
            best_loss = loss
            save_model(model, HR)
        if verbose:
            t = end - start
            print(f'[Epoch {epoch}, {t:.1f}s] loss: {loss} | HR@k: {HR} | NDCG@k: {NDCG}')

    if verbose:
        print(f'[INFO] the best HR@k was {best_HR} with loss {best_loss}')

    return best_HR, best_loss


def evaluate(model, loader, train_dict):
    """ return average HR@k, NDCG@k for valid or test set """
    model.eval()
    item_all = torch.LongTensor(range(item_num)).to(DEVICE)
    label = []
    score_all = []

    with torch.no_grad():
        for user, item in loader:
            user, item = user.to(DEVICE), item.to(DEVICE)
            scores = model.predict(user, item_all)  # (128, item_num)
            score_all.append(scores)
            label.extend(item)
        score_all = torch.cat(tuple(score_all), dim=0)
        HR, NDCG = eval(score_all, label, train_dict, k=50)

    return HR, NDCG


def save_model(model, HR):
    # TODO
    # save epoch, model and optimizer for subsequent additional learning
    # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    target_path = os.path.join(model_path, f'{model.__class__.__name__}_{HR:.4f}.pt')
    if os.path.isfile(target_path):
        print('[WARNING] snapshot overwriting occured')
        target_path = os.path.join(model_path, f'{model.__class__.__name__}_{HR:.4f}_2.pt')

    torch.save(model.state_dict(), target_path)


def tuning_GMF(config):
    candidates = [config['lr'], config['embed_dim'], config['neg_num'], config['batch_size']]
    best_hp = None
    best_loss = float('inf')
    best_hr = -1

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
        print(f'[INFO] hyperparameters:\n{hp}')
        model = GMF(hp['model'])
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=hp['etc']['lr'])
        loader = train_real_generator.get_loader(hp['etc']['num_neg'], hp['etc']['batch_size'])
        hr, loss = train(model, optimizer, loader, epochs=config['epochs'])

        if hr > best_hr:
            best_hr = hr
            best_hp = hp
            best_loss = loss

    print(f'[INFO] the best hyperparameters is \n{best_hp}\n HR@k: {best_hr} | loss: {best_loss}')


if __name__ == '__main__':
    entry()