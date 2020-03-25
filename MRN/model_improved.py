import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


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

        self.registers = None
        self.rnns = nn.ModuleList()
        for i in range(self.type_num):
            self.rnns.append(nn.GRUCell(self.input_size, self.hidden_size))
        self.core_gru = nn.GRUCell(self.type_num * self.hidden_size, self.hidden_size)
        self.core_linear1 = nn.Linear((self.type_num+1) * self.hidden_size, 2 * self.hidden_size)
        self.core_linear2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, input, type, c=None):
        """
        :param
        - input: (batch_size, input_size)
        - type: (batch_size, ), 0, 1, ..., type_num-1
        - c: (batch_size, hidden_size), last core state
        """
        batch_size = input.size(0)
        if c is None:
            # registers: (batch_size, type_num, hidden_size), last hidden states of each gru cell
            self.registers = self.init_registers(batch_size, input.device)


        type = type.numpy().astype(np.int32)
        type_others = []
        for idx in type:
            ls = list(range(self.type_num))
            ls.pop(idx)
            type_others.append(ls)
        type_others = np.array(type_others)

        # (batch_size, type_num, hidden_size)
        hidden_tmp = torch.stack([rnn(input, self.registers[:, i, :]) for i, rnn in enumerate(self.rnns)], dim=1)
        hidden = hidden_tmp[range(batch_size), type].unsqueeze(dim=1)  # (batch_size, 1, hidden_size)
        # (batch_size, type_num-1, hidden_size)
        hidden_other = torch.stack([self.registers[range(batch_size), type_other] for type_other in type_others.T], dim=1)
        hidden_all = torch.cat([hidden, hidden_other], dim=1)  # (batch_size, type_num, hidden_size)
        N = [[i] for i in range(batch_size)]
        I = np.concatenate((np.expand_dims(type, axis=1), type_others), axis=1)
        self.registers = hidden_all[N, I]  # (batch_size, type_num, hidden_size)

        core = self.core_v2(c)

        return core

    def core_v2(self, c=None):
        x = self.registers
        x = x.view(x.size(0), x.size(1) * x.size(2))  # (batch_size, type_num * hidden_size)
        core = self.core_gru(x, c)

        return core

    def core(self, registers, c):
        # registers = torch.stack(registers, dim=0)   # (type_num, batch_size, hidden_size)
        # registers = registers.permute((1, 0, 2))  # (batch_size, type_num, hidden_size)
        x = torch.cat(registers.append(c), dim=1)  # (batch_size, (type_num+1) * hidden_size)
        x = self.activation(self.core_linear1(x))  # (batch_size, 2 * hidden_size)
        core = self.activation(self.core_linear2(x))  # (batch_size, hidden_size)

        return core

    def init_registers(self, batch_size, device):
        registers = torch.zeros(size=(batch_size, self.type_num, self.hidden_size)).to(device)

        return registers


class MRN4GRU(nn.Module):
    def __init__(self, input_size, hidden_size, type_num, batch_first=True):
        super(MRN4GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.type_num = type_num
        self.batch_first = batch_first

        self.mrn = MRN4GRUCell(input_size, hidden_size, type_num)

    # 0.25s: about x10 faster
    def forward(self, input, type, core=None):
        """
        :param
        - input: (batch_size, time, input_size)
        - type: (batch_size, time)
        :return
        - cores: (times, batch_size, hidden_size)
        """

        if self.batch_first:
            seqs = torch.unbind(input, dim=1)
            types = torch.unbind(type, dim=1)
        else:
            seqs = torch.unbind(input, dim=0)
            types = torch.unbind(type, dim=0)

        cores = []
        for x, behavior in zip(seqs, types):
            core = self.mrn(x, behavior, core)
            cores.append(core.clone())
        cores = torch.stack(cores)

        return cores
