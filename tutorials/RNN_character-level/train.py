import os, random, time
import torch.nn as nn
import torch.optim as optim
from .data import *
from .model import RNN

all_categories, category_lines, n_letters = get_data(os.path.expanduser('~/dataset/names/*.txt'))
torch.manual_seed(123)
n_hidden = 128
rnn = RNN(n_letters, n_hidden, len(all_categories))
optimizer = optim.SGD(rnn.parameters(), lr=0.005, momentum=0.9)


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_choice(ls):
    return ls[random.randint(0, len(ls) - 1)]


def random_training_example():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def train(print_every=5000):
    rnn.train()
    criterion = nn.NLLLoss()
    n_iters = 100000
    start = time.time()

    for iter in range(n_iters):
        category, line, category_tensor, line_tensor = random_training_example()
        hidden = rnn.init_hidden()

        for t in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[t], hidden)
        loss = criterion(output, category_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, time.time() - start, loss, line, guess, correct))


if __name__ == '__main__':
    train()
