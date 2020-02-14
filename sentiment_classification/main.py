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


def train(model, optimizer, epochs=10):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        start = time()
        train_epoch(model, optimizer, train_loader, epoch)
        end = time()
        val_loss, val_acc = evaluate(model, val_loader)

        print(f'[Epoch {epoch}] val_loss: {val_loss:.2f} | val_acc: {val_acc:.2f}% | time: {end-start:.2f}s')

        # save best_val_loss model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    test_loss, test_acc = evaluate(model, train_loader)
    print(f'[Final] test_loss: {test_loss:.2f} | test_acc: {test_acc:.2f}%')
    if not os.path.isdir('snapshot'):
        os.mkdir('snapshot')
    torch.save(model.state_dict(), f'./snapshot/model_{test_acc:.2f}.pt')

    return f'model_{test_acc:.2f}'


def save_hp(hp, name):
    with open(f'snapshot/{name}.hp', 'wb') as f:
        dill.dump(hp, f)


if __name__ == '__main__':
    hp = {'vocab_size': vocab_size,
          'embed_dim': 64,
          'class_num': 2,
          'hidden_dim': 128,
          'dropout': 0.5,
          'num_layers': 2
    }
    model = SimpleGRU(**hp)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    name = train(model, optimizer, 8)
    save_hp(hp, name)
