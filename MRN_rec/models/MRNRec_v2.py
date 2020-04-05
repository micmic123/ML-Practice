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
        self.mrn = MRN4GRU(self.embed_size, self.hidden_size, self.behavior_num, batch_first=True)
        self.mrn._device = self._device
        self.linear1 = nn.Linear(self.hidden_size, self.embed_size)
        bias = torch.empty(1, self.item_num + 1)
        torch.nn.init.xavier_normal_(bias)
        self.bias = nn.Parameter(bias)

    def forward(self, user, item, behavior, mask_len):
        """
        TODO
        1. neumf 빼기 + user 빼보기
        2. dropout
        3. attention in core and out of gru
        4. loss: BPR, margin

        0. @score계산 시 item embedding 사용하기 + NCF를 이 부분에 어떻게 응용할지 생각
        :param
        - user: (B, ) tensor of long
        - item: (B, T) tensor of long
        - behavior: (B, T) tensor of int
        - mask_len: (T, ) list of int
        :return
        - scores: (B, I) tensor of double
        """
        item_embedding = F.dropout(self.item_embedding(item))  # (B, T, E)
        core = F.dropout(self.mrn(item_embedding, behavior, mask_len))  # (B, H)
        x = F.relu(self.linear1(core))  # (B, E)
        all_item = self.item_embedding.weight.t()  # (E, I+1)
        scores = torch.matmul(x, all_item) + self.bias  # (B, I+1)
        scores = scores[:, 1:]  # (E, I)

        return scores
