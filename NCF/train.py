import itertools, argparse, os, json
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from data import SampleGenerator, TestGenerator
from models.GMF import GMF
from models.MLP import MLP
from models.NeuMF import NeuMF
from utils.metrics import eval
from utils.Tuner import GMFTuner, MLPTuner, NeuMFTuner


parser = argparse.ArgumentParser(description='e.g. nohup python -u train.py --options > [log_name].log &')
parser.add_argument('--mode', required=True, help='train or supplement')
parser.add_argument('--model', required=True, help='GMF, MLP or NeuMF')
parser.add_argument('--device', required=True, help='GPU id to use')
parser.add_argument('--target', required=False, help='target model to supplement')
parser.add_argument('--epochs', required=False, help='epochs to supplement')
parser.add_argument('--tqdm', required=False, help='on or off for printing progressbar', default='off')
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

# tqdm on/off
is_tqdm = True if args.tqdm.lower() == 'on' or args.tqdm.lower() == 'true' else False

# meta data information
meta = {
    'user_num': 6040,
    'item_num': 3706
}


def entry():
    mode = args.mode.lower()
    model = args.model.lower()

    # tuning model with train_real and save snapshot of the best model
    if mode == 'train':
        if model == 'gmf':
            config = {
                'lr': [1e-3],  # 1e-3,
                'embed_dim': [16, 32, 64],  # 32
                'neg_num': [4],
                'batch_size': [512],
                'epochs': 200
            }
            tuner = GMFTuner(config, meta)
            MODEL = GMF
        elif model == 'mlp':
            config = {
                'lr': [1e-3],  # 1e-3,
                'embed_dim': [16, 32, 64],  # 64
                'layer_num': [3],
                'neg_num': [4],
                'batch_size': [512],
                'epochs': 80
            }
            tuner = MLPTuner(config, meta)
            MODEL = MLP
        elif model == 'neumf':
            config = {

            }
            tuner = NeuMFTuner(config, meta)
            MODEL = NeuMF
        tuning(tuner, MODEL)

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


def tuning(tuner, MODEL):
    best_hp = None
    best_loss = float('inf')
    best_hr = -1
    print(f'model: {MODEL.__name__}')

    # for each hyperparamter combination
    for hp in tuner.get_hp():
        print(f'[INFO] hyperparameters:\n{hp}')
        model = MODEL(hp['model'])
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=hp['etc']['lr'])
        loader_option = (hp['etc']['num_neg'], hp['etc']['batch_size'])
        hr, loss = train(model, optimizer, loader_option, hp, epochs=hp['etc']['epochs'])

        if hr > best_hr:
            best_hr = hr
            best_hp = hp
            best_loss = loss

    print(f'[INFO] the best hyperparameters is \n{best_hp}\n HR@k: {best_hr} | loss: {best_loss}')


def train_epoch(model, optimizer, loader, epoch):
    """ train for one epoch and return average loss """
    loss_ls = []
    loader = tqdm(loader, desc=f'epoch {epoch}') if is_tqdm else loader

    for user, item, label in loader:
        user, item, label = user.to(DEVICE), item.to(DEVICE), label.to(DEVICE)

        scores = model(user, item)
        loss = F.binary_cross_entropy_with_logits(scores, label)  # default reduction='mean'
        loss_ls.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if is_tqdm:
        loader.close()
    return sum(loss_ls) / len(loss_ls)


def train(model, optimizer, loader_option, hp, epochs=10, verbose=True):
    """  train the given model

    :return
        - best HR@k, loss
    """
    best_HR = -1
    best_loss = float('inf')  # loss for one epoch with training set
    best_epoch = -1
    model.train()

    for epoch in range(epochs):
        loader = train_real_generator.get_loader(*loader_option)
        start = time()
        loss = train_epoch(model, optimizer, loader, epoch)
        end = time()
        HR, NDCG = evaluate(model, test_loader, train_real_dict)

        if HR > best_HR:
            best_HR = HR
            best_loss = loss
            best_epoch = epoch
            save_model(model, optimizer, HR, hp, epochs)
        if verbose:
            t = end - start
            print(f'[Epoch {epoch:>2}, {t:.1f}s] loss: {loss:.4f} | HR@k: {HR:.4f} | NDCG@k: {NDCG}')

    if verbose:
        print(f'[INFO] the best HR@k was {best_HR} with loss {best_loss} at {best_epoch}-th epoch')

    return best_HR, best_loss


def evaluate(model, loader, train_dict):
    """ return average HR@k, NDCG@k for valid or test set """
    model.eval()
    item_all = torch.LongTensor(range(meta['item_num'])).to(DEVICE)
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


def save_model(model, optimizer, HR, hp, epochs):
    """ save model, optimizer, hyperparameters and epochs for subsequent additional learning """
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'hp': hp,
        'epochs': epochs
    }
    target_path = os.path.join(model_path, f'{model.__class__.__name__}_{HR:.4f}.pt')
    cnt = 0
    while os.path.isfile(target_path):
        print('[WARNING] snapshot overwriting occured')
        cnt += 1
        target_path = os.path.join(model_path, f'{model.__class__.__name__}_{HR:.4f}_{cnt}.pt')

    torch.save(state, target_path)


if __name__ == '__main__':
    entry()