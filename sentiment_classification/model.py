import torch
import torch.nn as nn
import torch.nn.functional as F
from data import DEVICE


class SimpleGRU(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embed_dim, class_num, dropout=0.5, num_layers=2):
        super(SimpleGRU, nn.Module).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        """
        :param
        - x.size(): B x T = Batch_size x length of the sentence
        """
        x = self.embed(x)  # B x T x (embed_dim * num_direction)
        h_0 = self._init_hidden(x.size()[0])
        _, h_t = self.gru(x, h_0)
        # [hidden_state_all] B x T x (num_directions * hidden_dim)
        # [hidden_state_last] B x (num_layers * num_directions) x hidden_dim

        h_t = x[:, 0, :]  # B x hidden_dim
        out = self.out(h_t)

        return out

    def _init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.num_layers, self.hidden_dim, device=DEVICE)
