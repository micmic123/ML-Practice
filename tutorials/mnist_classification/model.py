import torch.nn as nn
import torch.nn.functional as F


# Custom model should inherit nn.Module.
# Thanks to Autograd, what you need to do is just defining forward() that includes some layers,
# operates some logic and returns scores.
# Also, you could add some needed layers in __init__() and using them on forward().
# Calculating derivative of each parameter used in forward() is automatically carried out by Autograd.
""" 
** NOTE **
torch.nn only supports mini-batches. The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
"""
class ConvNet(nn.Module):
    """
    Input -> Conv -> pool -> ReLU -> Conv -> Dropout -> pool -> ReLU -> FC -> ReLU -> Dropout(train) -> FC -> log-softmax
    N * 1 * 28 * 28 -> N * 10 * 24 * 24 -> N * 10 * 12 * 12 -> N * 20 * 8 * 8 -> N * 20 * 4 * 4 -> N * 320 -> N * 50 -> N * 10
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        # X.shape: N * 1 * 28 * 28
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #        dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # number of 1*5*5 filters = 10, N * 10 * 24 * 24
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # N * 20 * 8 * 8
        self.conv2_drop = nn.Dropout2d()

        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.max_pool = nn.MaxPool2d(2)  # N * 10 * 12 * 12, N * 20 * 4 * 4

        # ReLU(inplace=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Note: the following two ways for max pooling / relu are equivalent.
        # 1) with torch.nn.functional:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 2) with torch.nn:
        x = self.relu(self.max_pool(self.conv2_drop(self.conv2(x))))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # NOTE: It is perfectly safe to reuse the same Module many times when defining a computational graph.
        # The state of the network is held in the graph and not in the layers,
        # you can simply create a module object and reuse it over and over again for the recurrence.
        return F.log_softmax(x, dim=1)



