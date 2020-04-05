import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaLSTM(nn.Module):
    def __init__(self, config):
        super(VanillaLSTM, self).__init__()
        self.embed_size = config['embed_size']  # E
        self.hidden_size = config['hidden_size']  # H
        self.user_num = config['user_num']  # U
        self.item_num = config['item_num']  # I

        self.user_embedding = nn.Embedding(self.user_num, self.embed_size)
        self.item_embedding = nn.Embedding(self.item_num+1, self.embed_size, padding_idx=0)  # 0 is reserved for padding
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, dropout=0)
        self.linear1 = nn.Linear(self.hidden_size, self.embed_size)
        bias = torch.empty(1, self.item_num+1)
        torch.nn.init.xavier_normal_(bias)
        self.bias = nn.Parameter(bias)

    def forward(self, user, item, behavior, sample_len):
        """
        :param
        - user: (B, ) tensor of long
        - item: (B, T) tensor of long
        - sample_len: (B, ) list of int
        :return
        - scores: (B, I) tensor of double
        """
        item_embedding = F.dropout(self.item_embedding(item))  # (B, T, E)
        x_packed = pack_padded_sequence(item_embedding, sample_len, batch_first=True)
        output_packed, (h, c) = self.lstm(x_packed)  # h, c: (1, B, H)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)

        h = F.dropout(h.squeeze())  # (B, H)
        x = F.relu(self.linear1(h))  # (B, E)
        all_item = self.item_embedding.weight.t()  # (E, I+1)
        scores = torch.matmul(x, all_item) + self.bias  # (B, I+1)
        scores = scores[:, 1:]  # (E, I)

        return scores