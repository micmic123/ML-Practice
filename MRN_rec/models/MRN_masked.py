import numpy as np
import torch
import torch.nn as nn


class MRN4GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, type_num):
        """
        :param
        - input_size
        - hidden_size
        - type_num
        """
        super(MRN4GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.type_num = type_num

        self.rnns = nn.ModuleList()
        for i in range(self.type_num):
            self.rnns.append(nn.GRUCell(self.input_size, self.hidden_size))
        self.core_gru = nn.GRUCell(self.type_num * self.hidden_size, self.hidden_size)
        self.core_linear1 = nn.Linear((self.type_num+1) * self.hidden_size, 2 * self.hidden_size)
        self.core_linear2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, input, type, c=None, reg=None):
        """
        :param
        - input: (batch_size, input_size)
        - type: (batch_size, ), 0, 1, ..., type_num-1
        - c: (batch_size, hidden_size), core state
        """
        # print('forward')
        batch_size = input.size(0)
        if reg is None:
            # registers: (batch_size, type_num, hidden_size), last hidden states of each gru cell
            c, reg = self.init_c_reg(batch_size, input.device)
        # print('reg is None finished')
        type = type.numpy().astype(np.int32)
        type_others = []
        # print('type fin')
        for idx in type:
            ls = list(range(self.type_num))
            ls.pop(idx)
            type_others.append(ls)
        type_others = np.array(type_others)
        # print('type_others')
        # (batch_size, type_num, hidden_size)
        # print(reg)
        hidden_tmp = torch.stack([rnn(input, reg[:, i, :]) for i, rnn in enumerate(self.rnns)], dim=1)
        # print('a')
        hidden = hidden_tmp[range(batch_size), type].unsqueeze(dim=1)  # (batch_size, 1, hidden_size)
        # print('hidden')
        # (batch_size, type_num-1, hidden_size)
        hidden_other = torch.stack([reg[range(batch_size), type_other] for type_other in type_others.T], dim=1)
        hidden_all = torch.cat([hidden, hidden_other], dim=1)  # (batch_size, type_num, hidden_size)
        # print('hidden_all')
        N = [[i] for i in range(batch_size)]
        I = np.concatenate((np.expand_dims(type, axis=1), type_others), axis=1)
        # print('N, I')
        reg = hidden_all[N, I]  # (batch_size, type_num, hidden_size)
        # print('reg')
        core = self.core_v2(c, reg)
        # print('core')

        return core, reg

    def core_v2(self, c=None, reg=None):
        x = reg
        x = x.view(x.size(0), x.size(1) * x.size(2))  # (batch_size, type_num * hidden_size)
        core = self.core_gru(x, c)

        return core

    def core(self, registers, c):
        x = torch.cat(registers.append(c), dim=1)  # (batch_size, (type_num+1) * hidden_size)
        x = self.activation(self.core_linear1(x))  # (batch_size, 2 * hidden_size)
        core = self.activation(self.core_linear2(x))  # (batch_size, hidden_size)

        return core

    def init_c_reg(self, batch_size, device):
        c = torch.zeros(batch_size, self.hidden_size).to(device)
        reg = torch.zeros(size=(batch_size, self.type_num, self.hidden_size)).to(device)

        return c, reg


class MRN4GRU(nn.Module):
    def __init__(self, input_size, hidden_size, type_num, batch_first=False):
        super(MRN4GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.type_num = type_num
        self.batch_first = batch_first
        self._device = None

        self.mrn = MRN4GRUCell(input_size, hidden_size, type_num)

    def forward(self, input, type, mask_len, core=None):
        """
        :param
        - input: (batch_size, time, input_size)
        - type: (batch_size, time)
        - mask_len: (time, )
        :return
        - cores: (times, batch_size, hidden_size)
        """

        if self.batch_first:
            seqs = torch.unbind(input, dim=1)
            types = torch.unbind(type, dim=1)
            batch_size = input.size(0)
        else:
            seqs = input
            types = type
            batch_size = input.size(1)

        c, reg = self.mrn.init_c_reg(batch_size, self._device)
        for x, behavior, m in zip(seqs, types, mask_len):
            if not m:
                break

            core, reg = self.mrn(x[:m], behavior[:m], c[:m], reg[:m])
            c = torch.cat([core, c[m:]])

        return c
