import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.embed_dim = config['embed_dim']
        self.layer_num = config['layer_num']
        self.linear_dims = [2*self.embed_dim]
        for i in range(self.layer_num):
            self.linear_dims.append(self.linear_dims[-1]//2)

        self.user_embed = nn.Embedding(self.user_num, self.embed_dim)
        self.item_embed = nn.Embedding(self.item_num, self.embed_dim)
        self.linears = nn.ModuleList()
        for input, output in zip(self.linear_dims[:-1], self.linear_dims[1:]):
            self.linears.append(nn.Linear(input, output))
        self.activation = nn.ReLU()
        self.affine_out = nn.Linear(self.linear_dims[-1], 1)

    def forward(self, user, item):
        user_embedding = self.user_embed(user)
        item_embedding = self.item_embed(item)
        x = torch.cat((user_embedding, item_embedding), dim=1)
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)
        logits = self.affine_out(x).squeeze()

        return logits

    def predict(self, user, item):
        user_embedding = self.user_embed(user).unsqueeze(1).expand((-1, len(item), -1))  # B x item_num x embed_dim
        item_embedding = self.item_embed(item).expand_as(user_embedding)  # B x item_num x embed_dim
        x = torch.cat((user_embedding, item_embedding), dim=2)  # B x item_num x 2embed_dim
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)
        logits = self.affine_out(x).squeeze()

        return logits
