import torch.nn as nn
import torch.optim as optim
from model import *
from data import get_data, prepare_sequence


training_data, word_to_ix, tag_to_ix = get_data()
# These will usually be more like 32 or 64 dimensional.


def train(epoch):
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.NLLLoss()

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
