import time, tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from data import DEVICE, get_loader
from model import BasicLSTM
from util import evaluate


train_loader, test_loader, item_size, user_size = get_loader(batch_size=1024)


def train_epoch(model, optimizer, epoch):
    loss_total = 0.
    for batch_idx, (x, y_pos, y_neg, u) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                                                     desc=f'epoch {epoch:2d}'):
        x, y_pos, y_neg, u = x.to(DEVICE), y_pos.to(DEVICE), y_neg.to(DEVICE), u.to(DEVICE)
        scores = model(x)  # B x 63
        scores[:, :3] = F.logsigmoid(scores[:, :3])  # loss_pos
        scores[:, 3:] = torch.log(1 - F.sigmoid(scores[:, 3:]))  # loss_neg

        loss = torch.sum(scores)  # sum of each loss_pos and loss_neg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    return loss_total


def train(model, optimizer, epochs=5):
    model.train()
    k = 10
    for epoch in epochs:
        start = time.time()
        loss = train_epoch(model, optimizer, epoch)
        end = time.time()
        time_epoch = end - start

        start = time.time()
        prec, recall, mAP = eval(model, test_loader)
        end = time.time()
        time_eval = end - start
        print(f'Epoch: {epoch + 1}\tLoss: {loss:.4f} [{time_epoch:.1f} s]\t'
              f'Prec@{k}: {prec:.4f}\tRecall@{k}: {recall:.4f}\tMAP: {mAP:.4f} [{time_eval:.1f} s]')


def eval(model, loader):
    model.eval()
    for x, y, u in loader:  # 한 번에 다 갖고 오기 때문에 사실 1회만 반복
        x, y, u = x.to(DEVICE), y.to(DEVICE), u.to(DEVICE)
        scores = model.predict(x, u)
        prec, recall, mAP = evaluate(scores, y)

    return prec, recall, mAP


if __name__ == '__main__':
    # hp: item_size, user_size, embed_dim, hidden_dim, num_lstm, embed_user_dim
    model = BasicLSTM(item_size, user_size, 50, 50, 1, 50)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, 15)
