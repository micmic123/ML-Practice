import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RMTL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']  # H
        self.item_num = config['item_num']  # I, 0 for padding and 1 for keeping, so start with 2 for each unique item
        self.behavior_num = config['behavior_num']  # N
        self.embedding = nn.Embedding(self.item_num + 2, self.hidden_size, padding_idx=0)
        self.rnns = nn.ModuleList()
        for i in range(self.behavior_num):
            self.rnns.append(nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True))
        self.linear1 = nn.Linear((self.behavior_num+1) * self.hidden_size, 3 * (self.behavior_num+1) * self.hidden_size)
        self.linear2 = nn.Linear(3 * (self.behavior_num+1) * self.hidden_size, self.item_num)

        """
        TODO: Note that output of forward() is (B, I), not (B, I+2). 
              Thus, needs to modify train.py - train_epoch and metric.py - filtering.
        """

    def forward(self, seqs, mask_lens, target_behavior):
        """
        :param
        - seqs: list of (B, T) with length = N
        - mask_lens:
        - target_behavior: (B, )
        :return
        - scores: (B, I)
        """
        embeddings = [self.item_embedding(seq) for seq in seqs]  # list of (B, T, H) with len = N
        accumulated_output = 0
        hiddens = []  # (N, B, H)
        for i, embedding in enumerate(embeddings):
            if i == 0:
                x = embedding  # (B, T, H)
            else:
                x = embedding + accumulated_output
            rnn_input = pack_padded_sequence(x, mask_lens[i], batch_first=True)
            output_packed, hidden = self.rnns[i](rnn_input)  # hidden: (1, B, H)
            output, output_lengths = pad_packed_sequence(output_packed, batch_first=True)  # output: (B, T, H)
            accumulated_output += output
            hiddens.append(hidden.squeeze())
        hiddens = torch.stack(hiddens, dim=1)  # (B, N, H)
        hidden_target = hiddens[range(target_behavior.size(0)), target_behavior]  # (B, H)
        context = self.attention(hidden_target, hiddens)  # (B, H)
        final = torch.cat([hiddens, context.unsqueeze(1)], dim=1)  # (B, N+1, H)
        final = final.view(final.size(0), -1)  # (B, (N+1)*H)
        x = F.relu(self.linear1(final))  # (B, 3*(N+1)*H)
        scores = self.linear2(x)  # (B, I)

        return scores

    def attention(self, query, key):
        """
        :param
        - query: (B, H)
        - key: (B, N, H)
        :return
        - context: (B, H)
        """
        query = query.unsqueeze(2)  # (B, H, 1)
        scores = torch.bmm(key, query)  # (B, N, 1),
        weights = F.softmax(scores, dim=1)  # (B, N, 1)
        context = torch.sum(weights * key, dim=1)  # (B, H)

        return context
