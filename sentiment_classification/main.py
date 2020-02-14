import os, dill
from time import time
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from data import get_data, DEVICE
from model import SimpleGRU


train_loader, val_loader, test_loader, vocab_size = get_data()


# train for one epoch
def train_epoch(model, optimizer, loader, epoch):
    model.train()
    for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f'epoch {epoch}'):
        # print(f'INFO train {i} starts')
        # print(batch)
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        scores = model(x)
        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f'INFO train {i} end')


def evaluate(model, loader):
    model.eval()
    loss_total, correct = 0, 0
    for i, batch in enumerate(loader):
        # print(f'INFO eval {i} starts')
        # print(batch)
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        scores = model(x)
        loss_total += F.cross_entropy(scores, y, reduction='sum').item()
        pred = scores.max(dim=1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        # print(f'INFO eval {i} end')

    num = len(loader.dataset)
    loss_avg = loss_total / num
    acc = 100 * correct / num

    return loss_avg, acc


def train(model, optimizer, epochs=10, verbose=False):
    val_loss, val_acc = None, None

    for epoch in range(epochs):
        start = time()
        train_epoch(model, optimizer, train_loader, epoch)
        end = time()
        val_loss, val_acc = evaluate(model, val_loader)

        if verbose:
            print(f'[Epoch {epoch}] val_loss: {val_loss:.2f} | val_acc: {val_acc:.2f}% | time: {end-start:.2f}s')

    return val_loss, val_acc


def save_hp(hp, name):
    if not os.path.isdir('snapshot'):
        os.mkdir('snapshot')
    with open(f'snapshot/{name}.hp', 'wb') as f:
        dill.dump(hp, f)


def save_model(model, name):
    if not os.path.isdir('snapshot'):
        os.mkdir('snapshot')
    torch.save(model.state_dict(), f'./snapshot/{name}.pt')


def tunning_hp():
    candidate = {
        'model': {
            'vocab_size': vocab_size,
            'embed_dim': [32, 64, 128],
            'class_num': [2],
            'hidden_dim': [64, 128, 196],
            'dropout': [0.5],
            'num_layers': [2]
        },
        'optim': {
            'lr': [1e-5, 5e-5, 1e-4]
        }
    }

    best_loss = float('inf')
    best_model = None
    best_hp = None
    for embed_dim in candidate['model']['embed_dim']:
        for hidden_dim in candidate['model']['hidden_dim']:
            for lr in candidate['optim']['lr']:
                hp = {
                    'model': {
                        'vocab_size': vocab_size,
                        'embed_dim': embed_dim,
                        'class_num': 2,
                        'hidden_dim': hidden_dim,
                        'dropout': 0.5,
                        'num_layers': 2
                    },
                    optim: {
                        'lr': lr
                    }
                }
                print(f'[INFO] current hyper-parameter: \n{hp}')
                model = SimpleGRU(**hp['model'])
                model.to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=hp['optim']['lr'])
                val_loss, val_acc = train(model, optimizer, 5, verbose=True)

                if val_loss < best_loss:
                    best_model = model
                    best_hp = hp
                    best_loss = val_loss

    # additional train for best_model with best_hp
    optimizer = optim.Adam(best_model.parameters(), lr=best_hp['optim']['lr'])
    train(best_model, optimizer, 5, verbose=True)
    test_loss, test_acc = evaluate(best_model, test_loader)
    print(f'[INFO] the best hyper-parameter: \n{best_hp}')
    print(f'[INFO] test_loss: {test_loss:.2f} | test_acc: {test_acc:.2f}%')
    name = f'{test_acc:.2f}'
    save_hp(best_hp, name)
    save_model(best_model, name)


if __name__ == '__main__':
    tunning_hp()
