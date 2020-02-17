import torch
import torch.nn as nn
import torch.nn.functional as F
from data import DEVICE


class BasicLSTM(nn.Module):
    def __init__(self, item_size, user_size, embed_dim, hidden_dim, num_lstm, embed_user_dim):
        super(BasicLSTM, self).__init__()
        self.item_size = item_size
        self.hidden_dim = hidden_dim
        self.embed_user_dim = embed_user_dim
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(item_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_lstm, batch_first=True)
        self.embed_user = nn.Embedding(user_size, embed_user_dim)
        self.embed_item = nn.Embedding(item_size, embed_dim)
        self.linear1 = nn.Linear(hidden_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim + embed_user_dim, embed_dim)

    def get_vec(self, x, user):
        embedding = self.embed(x)  # B x T x embed_dim
        _, h = self.lstm(embedding)
        h = h[0]  # num_lstm x B x hidden_dim
        h = h[-1, :, :]  # B x hidden_dim
        h = F.relu(self.linear1(h))  # B x embed_dim
        embedding_user = self.embed_user(user)  # B x embed_user_dim
        vec = torch.cat([h, embedding_user], dim=1)  # B x (embed_dim + embed_user_dim)
        vec = self.linear2(vec)  # B x embed_dim
        return vec

    def forward(self, x, user, sample_pos, sample_neg):
        """
        Calculates only scores of sample items
        All arguments should be Tensor and be set .to(DEVICE) before calling
        :param
        - x: B x T
        - user: (B, )
        - sample_pos: B x 3
        - sample_neg: B x 60
        :return
        - scores of B x 63
        """
        vec = self.get_vec(x, user)
        vec = torch.unsqueeze(vec, dim=1)  # B x 1 x embed_dim
        sample = torch.cat([sample_pos, sample_neg], dim=1)  # B x 63
        embedding_sample = self.embed_item(sample)  # B x 63 x embed_dim
        # B x embed_dim x 63
        embedding_sample = embedding_sample.permute((0, 2, 1))
        scores = torch.bmm(vec, embedding_sample)  # B x 1 x 63
        scores = torch.squeeze(scores, dim=1)  # B x 63

        return scores

    def predict(self, x, user):
        """
        Calculates all scores of items
        :param
        - x: B x T
        - user: (B, )
        :return
        - scores of B x item_size
        """
        vec = self.get_vec(x, user)  # B x embed_dim
        items = self.embed_item(torch.LongTensor(range(self.item_size)).to(DEVICE)).t()  # embed_dim x item_size
        scores = torch.matmul(vec, items)  # B x item_size

        return scores
