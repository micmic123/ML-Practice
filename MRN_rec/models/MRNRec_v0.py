import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MRN_masked import MRN4GRU
from models.NCF import SimpleNeuMF


class MRN4Rec(nn.Module):
    def __init__(self, config):
        super(MRN4Rec, self).__init__()
        self.embed_size = config['embed_size']  # E
        self.hidden_size = config['hidden_size']  # H
        self.mrn_in_size = config['mrn_in_size']  # MI
        self.fcl_size = config['fcl_size']  # F
        self.user_num = config['user_num']  # U
        self.item_num = config['item_num']  # I
        self.behavior_num = config['behavior_num']  # N
        self._device = config['device']

        self.user_embedding = nn.Embedding(self.user_num, self.embed_size)
        self.item_embedding = nn.Embedding(self.item_num+1, self.embed_size, padding_idx=0)  # 0 is reserved for padding
        self.simple_neumf = SimpleNeuMF(self.embed_size, self.mrn_in_size)
        self.mrn = MRN4GRU(self.mrn_in_size, self.hidden_size, self.behavior_num, batch_first=True)
        self.mrn._device = self._device
        self.linear1 = nn.Linear(self.hidden_size + self.embed_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size + self.embed_size, self.embed_size)
        self.linear3 = nn.Linear(self.embed_size, 1)

    def forward(self, user, item, behavior, mask_len, target=None):
        """

        :param
        - user: (B, ) tensor of long
        - item: (B, T) tensor of long
        - behavior: (B, T) tensor of int
        - mask_len: (T, ) tensor of bool
        - target: (B, # sample) tensor of long
        :return
        - scores: (B, # sample) tensor of double
        """
        if target is None:
            target = self.all_item(self._device)
            # print(target.size())
            target_embedding = self.item_embedding(target).unsqueeze(0)  # (1, # sample, E)
            # print(target_embedding.size())
            target_embedding = target_embedding.expand(user.size(0), -1, -1)  # (B, # sample, E)
        else:
            target_embedding = self.item_embedding(target)  # (B, # sample, E)
        user_embedding = self.user_embedding(user)  # (B, E)
        item_embedding = self.item_embedding(item)  # (B, T, E)
        embedding = self.simple_neumf(user_embedding, item_embedding)  # (B, T, MI)
        core = self.mrn(embedding, behavior, mask_len)  # (B, H)
        x = torch.cat([core, user_embedding], dim=1)  # (B, H + E)
        x = F.relu(self.linear1(x)).unsqueeze(dim=1)  # (B, 1, H)
        x = x.expand(-1, target_embedding.size(1), -1)  # (B, # sample, H)
        x = torch.cat([x, target_embedding], dim=2)  # (B, # sample, H + E)
        x = F.relu(self.linear2(x))  # (B, # sample, E)
        x = self.linear3(x).squeeze()  # (B, # sample)

        return x

    def all_item(self, device):
        return (torch.LongTensor(range(self.item_num)) + 1).to(device)
