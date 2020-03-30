import torch
import torch.nn as nn


class SimpleNeuMF(nn.Module):
    def __init__(self, embed_size, out_size):
        super(SimpleNeuMF, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size

        self.linear1 = nn.Linear(3 * self.embed_size, 2 * self.embed_size)
        self.linear2 = nn.Linear(2 * self.embed_size, self.out_size)
        self.activation = nn.ReLU()

    def forward(self, user_embed, item_embed):
        """
        :param
        - user_embed: (B, embed_size)
        - item_embed: (B, T, embed_size)
        :return (B, T, out_size)
        """
        user_embed = user_embed.unsqueeze(dim=1).expand_as(item_embed)  # (B, T, embed_size)
        x = torch.cat([user_embed * item_embed, user_embed, item_embed], dim=2)
        a = self.activation(self.linear1(x))
        z = self.linear2(a)

        return z
