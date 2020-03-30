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

        self.rnns = nn.ModuleList()
        for i in range(self.type_num):
            self.rnns.append(nn.GRUCell(self.input_size, self.hidden_size))
        self.core_gru = nn.GRUCell(self.type_num * self.hidden_size, self.hidden_size)
        self.core_linear1 = nn.Linear((self.type_num+1) * self.hidden_size, 2 * self.hidden_size)
        self.core_linear2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, input, type, registers=None, c=None):
        """
        :param
        - input: (batch_size, input_size)
        - type: (batch_size, ), 0, 1, ..., type_num-1
        - registers: (batch_size, type_num, hidden_size), last hidden states of each gru cell, 2d-array of (batch_size, type_num)
        - c: (batch_size, hidden_size), last core state
        """
        hiddens = [None] * self.type_num
        ### 병목 10%  ###
        if registers is None:
            registers = self.init(input.size(0), input.device)
            hiddens = [None] * self.type_num
        else:
            for t in range(self.type_num):
                hiddens[t] = torch.stack([batch[t] for batch in registers])  # registers의 type=t인 열 tensor로 가져옴
        ################

        type = list(map(int, type.tolist()))

        for i, rnn in enumerate(self.rnns):
            hiddens[i] = rnn(input, hiddens[i])

        ### 병목 10%  ###
        for i, t in enumerate(type):
            registers[i][t] = hiddens[t][i]
        ################


        ### 병목 70%  ###
        core = self.core_v2(registers, c)
        ################
        return registers, core

    def core_v2(self, registers, c=None):
        # x = torch.cat(registers, dim=1)  # (batch_size, type_num * hidden_size)
        ### 병목 99% ###
        x = torch.stack([torch.stack(i) for i in registers])  # (batch_size, type_num, hidden_size)
        ###############
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

    def init(self, batch_size, device):
        # (batch_size, type_num)
        registers = [[torch.zeros(size=(self.hidden_size,)).to(device) for j in range(self.type_num)] for i in range(batch_size)]
        # registers = [torch.zeros(size=(batch_size, self.hidden_size)).to(device) for i in range(self.type_num)]
        # c = torch.zeros(size=(batch_size, self.hidden_size)).to(device)

        return registers


class MRN4GRU(nn.Module):
    def __init__(self, input_size, hidden_size, type_num, batch_first=True):
        super(MRN4GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.type_num = type_num
        self.batch_first = batch_first

        self.mrn = MRN4GRUCell(input_size, hidden_size, type_num)

    # 2.75s
    def forward(self, input, type, hidden=None, core=None):
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
            hidden, core = self.mrn(x, behavior, hidden, core)
            cores.append(core.clone())
        cores = torch.stack(cores)

        return cores
