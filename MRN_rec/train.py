import argparse, os
from time import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from data import SampleGenerator, read_data
from models.VanillaLSTM import VanillaLSTM, VanillaLSTM_v0, VanillaGRU_v0
from models.MRNRec_v3 import MRN4Rec
from utils.metrics import eval
from utils.Tuner import MRNRecTuner, LSTMTuner


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
model_path = './snapshot'

# data of ndarray
print(f'[INFO] data loading started')
data = read_data()
meta = data['meta']
trainset = data['train']
testset = data['test']
train_item = data['train_item']

# batch generator
train_generator = None
test_loader = None

# tqdm on/off
is_tqdm = True if args.tqdm.lower() == 'on' or args.tqdm.lower() == 'true' else False


def entry():
    mode = args.mode.lower()
    model = args.model.lower()
    global train_generator
    global test_loader

    # tuning model with train_real and save snapshot of the best model
    if mode == 'train':
        if model == 'mrn':
            config = {
                # 'lr': [1e-4],  # 1e-3,
                # 'embed_size': [32, 64],  # 32
                # 'hidden_size': [16, 32, 64],
                # 'mrn_in_size': [64],
                # 'fcl_size': [64],
                # 'num_neg': [4],
                # 'batch_size': [1024],
                # 'epochs': 80,
                # 'device': DEVICE
                'lr': [1e-4],  # 5e-5, 3e-6 was good
                'embed_size': [100],  # 32
                'hidden_size': [512],
                'mrn_in_size': [100],
                'fcl_size': [256],
                'num_neg': [16, 32],
                'reg': [0],
                'batch_size': [512],
                'epochs': 80,
                'device': DEVICE
            }
            tuner = MRNRecTuner(config, meta)
            MODEL = MRN4Rec
        elif model == 'lstm':
            config = {
                'lr': [3e-4],
                'embed_size': [100],
                'hidden_size': [512],
                'num_neg': [32],
                'reg': [0],
                'batch_size': [512],
                'epochs': 100,
                'device': DEVICE
            }
            tuner = LSTMTuner(config, meta)
            MODEL = VanillaLSTM_v0
        elif model == 'gru':
            config = {
                'lr': [3e-4],
                'embed_size': [100],
                'hidden_size': [512],
                'num_neg': [32],
                'reg': [0],
                'batch_size': [512],
                'epochs': 80,
                'device': DEVICE
            }
            tuner = LSTMTuner(config, meta)
            MODEL = VanillaGRU_v0
        elif model == 'neumf':
            config = {

            }
            # tuner = NeuMFTuner(config, meta)
            # MODEL = NeuMF
        train_generator = SampleGenerator(trainset, meta, train_item, model=model)
        test_loader = SampleGenerator(testset, meta, train_item, is_test=True, model=model).get_loader()
        print(f'[INFO] data loading finished')
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
        optimizer = optim.Adam(model.parameters(), lr=hp['etc']['lr'], weight_decay=hp['etc']['reg'])
        loader_option = (hp['etc']['num_neg'], hp['etc']['batch_size'])
        hr, loss = train(model, optimizer, loader_option, hp, epochs=hp['etc']['epochs'])

        if hr > best_hr:
            best_hr = hr
            best_hp = hp
            best_loss = loss

    print(f'[INFO] the best hyperparameters is \n{best_hp}\n HR@k: {best_hr} | loss: {best_loss}')


def train_epoch(model, optimizer, loader, epoch, num_neg):
    """ train for one epoch and return average loss """
    model.train()
    loss_ls = []
    loader = tqdm(loader, desc=f'epoch {epoch}') if is_tqdm else loader
    # cnt = 0
    for user, item, behavior, sample, mask_len in loader:
        user, item, sample = user.to(DEVICE), item.to(DEVICE), sample.to(DEVICE)

        # sample, label, scores: (B, 1 + neg_num)
        scores = model(user, item, behavior, mask_len)
        scores = scores[[[i] for i in range(user.size(0))], sample - 1]
        label = torch.FloatTensor([([1] + ([0] * num_neg)) for i in range(user.size(0))]).to(DEVICE)

        loss = F.binary_cross_entropy_with_logits(scores, label)
        loss_ls.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # cnt += 1
        # if cnt == 1: break

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

    for epoch in range(epochs):
        start = time()
        loader = train_generator.get_loader(*loader_option)
        loss = train_epoch(model, optimizer, loader, epoch, loader_option[0])
        HRs, NDCGs = evaluate(model, test_loader, train_item)
        end = time()

        if HRs[3] > best_HR:
            best_HR = HRs[3]
            best_loss = loss
            best_epoch = epoch
            save_model(model, optimizer, HRs[3], hp, epochs)
        if verbose:
            t = end - start
            print(f'[Epoch {epoch:>2}, {t:.1f}s] loss: {loss:.4f} | k: [1, 5, 10, 20, 50]')
            print(f'HR@k: {HRs:} | NDCG@k: {NDCGs}')

    if verbose:
        print(f'[INFO] the best HR@20 was {best_HR} with loss {best_loss} at {best_epoch}-th epoch')

    return best_HR, best_loss


def evaluate(model, loader, train_dict):
    """ return average HR@k, NDCG@k for valid or test set """
    model.eval()
    test_num = 0
    HR_all = []
    NDCG_all = []

    with torch.no_grad():
        for user, item, behavior, y, mask_len in loader:
            y = y.view(-1).tolist()
            user, item, mask_len = user.to(DEVICE), item.to(DEVICE), mask_len
            scores = model(user, item, behavior, mask_len)
            HR_sum, NDCG_sum = eval(user.tolist(), scores, y, train_dict, k_ls=[1, 5, 10, 20, 50])
            HR_all.append(HR_sum)
            NDCG_all.append(NDCG_sum)
            test_num += len(y)
        HRs = [round(hr, 4) for hr in (np.array(HR_all).sum(axis=0) / test_num)]
        NDCGs = [round(ndcg, 4) for ndcg in (np.array(NDCG_all).sum(axis=0) / test_num)]

    return HRs, NDCGs


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
