import time, tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from data import DEVICE, get_data
from model import BasicLSTM
from util import evaluate


batch_size = 1024
train_loader, testset, item_size, user_size = get_data(batch_size=batch_size)
# torch.autograd.set_detect_anomaly(True)


def train_epoch(model, optimizer, epoch):
    model.train()
    loss_total = 0.
    epsilon = 1e-10
    for batch_idx, (x, y_pos, y_neg, u) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                                                     desc=f'epoch {epoch:2d}'):
        x, y_pos, y_neg, u = x.to(DEVICE), y_pos.to(DEVICE), y_neg.to(DEVICE), u.to(DEVICE)
        scores = model(x, u, y_pos, y_neg)  # B x 63
        # if epoch > 2:
        #     with torch.no_grad():
        #         # print(torch.sigmoid(scores[:, :3]))
        #         print(1-torch.sigmoid(scores[:, 3:]))
        loss_pos = -torch.sum(F.logsigmoid(scores[:, :3]), dim=1)
        loss_neg = -torch.sum(torch.log(1 - torch.sigmoid(scores[:, 3:]) + epsilon), dim=1)
        loss = torch.mean(loss_pos + loss_neg)  # mean of each loss for one data

        # Below is error since some data of scores is replaced in-place, so autograd cannot trace it.
        # scores[:, :3] = F.logsigmoid(scores[:, :3])  # loss_pos
        # scores[:, 3:] = torch.log(1 - torch.sigmoid(scores[:, 3:]))  # loss_neg
        # loss = torch.sum(scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
    loss_total /= (batch_idx+1)
    return loss_total


def train(model, optimizer, epochs=5):
    k = 10
    for epoch in range(epochs):
        start = time.time()
        loss = train_epoch(model, optimizer, epoch)
        end = time.time()
        time_epoch = end - start

        start = time.time()
        prec, recall, mAP = eval(model)
        end = time.time()
        time_eval = end - start

        print(f'Epoch: {epoch + 1}\tLoss: {loss:.4f} [{time_epoch:.1f} s]\t'
              f'Prec@{k}: {prec:.4f}\tRecall@{k}: {recall:.4f}\tMAP: {mAP:.4f} [{time_eval:.1f} s]')


def eval(model):
    model.eval()
    x, y, u = testset
    x, u = x.to(DEVICE), u.to(DEVICE)
    scores = model.predict(x, u)
    prec, recall, mAP = evaluate(scores, y)

    return prec, recall, mAP


if __name__ == '__main__':
    # hp: item_size, user_size, embed_dim, hidden_dim, num_lstm, embed_user_dim
    model = BasicLSTM(item_size, user_size, 50, 50, 1, 50)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, 15)
