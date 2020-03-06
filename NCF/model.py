import torch
import torch.nn as nn
import torch.nn.functional as F


class GMF(nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.embed_dim = config['embed_dim']

        self.user_embed = nn.Embedding(self.user_num, self.embed_dim)
        self.item_embed = nn.Embedding(self.item_num, self.embed_dim)
        self.linear = nn.Linear(self.embed_dim, 1)

    def forward(self, user, item):
        """
        :param
            - user, item: (B, )
        :return
            - logits: scores of (B, )
        """
        user_embedding = self.user_embed(user)
        item_embedding = self.item_embed(item)
        logits = self.linear(user_embedding * item_embedding)

        return logits.squeeze()

    def predict(self, user, item):
        """
        :param
            - user: (B, )
            - item: (item_num, )
        :return
            - logits: scores of (B, item_num)
        """
        user_embedding = self.user_embed(user)
        user_embedding = user_embedding.unsqueeze(1)  # (B, 1, embed_dim)
        item_embedding = self.item_embed(item)  # (item_num, embed_dim)
        x = user_embedding * item_embedding  # (B, item_num, embed_dim)
        scores = self.linear(x)  # (B, item_num, 1)

        return scores.squeeze()


class MLP(nn.Module):
    pass


class NeuMF(nn.Module):
    pass
