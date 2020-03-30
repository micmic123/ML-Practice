import os, tqdm
import torch
import torch.nn as nn


input_size, hidden_size, batch_size = 128, 128, 1024
times = 100
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch():
    return torch.randn(times, batch_size, input_size).to(DEVICE)


gru = nn.GRUCell(input_size, hidden_size).to(DEVICE)
gru1 = nn.GRUCell(input_size, hidden_size).to(DEVICE)
gru2 = nn.GRUCell(input_size, hidden_size).to(DEVICE)
# for one epoch
for j in tqdm.tqdm(range(420)):  # one mini-batch
    batch = get_batch()
    hidden1 = None
    hidden2 = None
    hidden3 = None
    for t in range(times):
        hidden1 = gru(batch[t], hidden1)
        hidden2 = gru1(batch[t], hidden2)
        hidden3 = gru2(batch[t], hidden3)

for j in tqdm.tqdm(range(420)):  # one mini-batch
    batch = get_batch()
    hidden = None
    for t in range(times):
        # one data
        for i in batch[t].unsqueeze(dim=1):
            # i = torch.unsqueeze(i, dim=0)
            hidden = gru(i, hidden)
