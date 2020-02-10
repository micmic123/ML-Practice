import glob
import os
import unicodedata
import string
import torch

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor that consists of one-hot vector for a character
# 1 means that the size of batch = 1
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters> Tensor that consists of time series one-hot vector with batch_size=1,
# In Pytorch RNN, shape of input tensor is T x B x D = time-steps x batch size x len(vector of one data)
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


# Read a file and split into lines
def read_lines(filename):
    lines = open(filename, 'r', encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


# return the category_lines dictionary, a list of names per language
def get_data(path):
    category_lines = {}
    all_categories = []

    for filename in glob.glob(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return all_categories, category_lines, n_letters


if __name__ == '__main__':
    print(unicode_to_ascii('Ślusàrski'))  # Slusarski
    print(letter_to_tensor('J'))
    print(line_to_tensor('Jones').size())  # torch.Size([5, 1, 57])
