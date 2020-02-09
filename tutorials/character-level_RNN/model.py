import torch
import torch.nn as nn


# only for one time point.
# refer to https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        :param
        - input: 1 x input_size
        - hidden: 1 x hidden_size
        """
        combined = torch.cat((input, hidden), dim=1)  # treat as bunch
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.log_softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)




