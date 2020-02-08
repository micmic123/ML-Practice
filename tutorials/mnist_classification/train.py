import os
from time import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import MNIST
from model import ConvNet


# Create the MNIST dataset.
# transforms.ToTensor() automatically converts PIL images to torch tensors with range [0, 1]
train_set = MNIST(os.path.expanduser('~/dataset/mnist_png/training'), preload=True, transform=transforms.ToTensor())
# Use the torch dataloader to iterate through the dataset
# We want the dataset to be shuffled during training.
train_set_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1)
test_set = MNIST(os.path.expanduser('~/dataset/mnist_png/testing'), preload=True, transform=transforms.ToTensor())
test_set_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=1)

# Use GPU if available, otherwise stick with cpu
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = ConvNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(epoch, log_interval=100):
    model.train()  # set training mode
    iteration = 0

    for ep in range(epoch):
        start = time()
        for batch_idx, (data, target) in enumerate(train_set_loader):
            # bring data to the computing device, e.g. GPU
            data, target = data.to(device), target.to(device)
            # data: torch.Size([64, 1, 28, 28])
            # target: torch.Size([64])

            # forward pass
            output = model(data)  # In __call__ in nn.Module, it calls self.forward(x) and returns it.
            # compute loss: negative log-likelihood and it is equivalent to softmax with cross-entropy
            loss = F.nll_loss(output, target)

            # backward pass
            # clear the gradients of all tensors being optimized since they are accumulated.
            optimizer.zero_grad()  # All gradients of parameters become zero and they will not be traced.
            # accumulate (i.e. add) the gradients from this forward pass
            loss.backward()  # Any argument is not needed since loss is scalar tensor.
            # performs a single optimization step (parameter update)
            optimizer.step()

            if iteration % log_interval == 0:
                print(f'Epoch: {ep} [{batch_idx * len(data)}/{len(train_set_loader.dataset)}'
                      f'({(100. * batch_idx / len(train_set_loader)):.2f})]\tLoss: {loss.item():.6f}]')
            iteration += 1

        end = time()
        print(f'Time: {(end-start):.2f}s')
        test()  # evaluate at the end of epoch


def test():
    model.eval()  # set evaluation mode
    test_loss = 0
    correct = 0

    # do not calculate any gradient in this context.
    with torch.no_grad():
        for data, target in test_set_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(dim=1, keepdim=True)[1] # get the index of the max log-probability. i.e. argmax
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_set_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_set_loader.dataset)} '
          f'({(100. * correct / len(test_set_loader.dataset)):.2f}%)\n')


if __name__ == '__main__':
    train()