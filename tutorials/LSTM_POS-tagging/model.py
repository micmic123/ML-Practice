import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))  # hiddens, (h_final, c_final)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class LSTMTagger2(nn.Module):
    def __init__(self, embed_word_dim, embed_char_dim, hidden_dim, hidden_dim2, vocab_size, char_size, tagset_size):
        super(LSTMTagger2, self).__init__()
        self.embedding_word = nn.Embedding(vocab_size, embed_word_dim)
        self.embedding_char = nn.Embedding(char_size, embed_char_dim)
        self.lstm_final = nn.LSTM(embed_word_dim + hidden_dim2, hidden_dim)
        self.lstm_char = nn.LSTM(embed_char_dim, hidden_dim2)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, seq_chars):
        """

        :param
        - sentence: (T, )
        - seq_chars: (T, len(w_t))
        :return:
        """
        embed_word = self.embedding_word(sentence)  # T x embed_word_dim
        embed_word = embed_word.view(len(sentence), 1, -1)  # T x 1 x embed_word_dim
        embed_chars = [self.embedding_char(chars) for chars in seq_chars]  # T x len(w_t) x embed_char_dim
        hidden_chars = [self.lstm_char(embed_char.view(len(embed_char), 1, -1))[1][0] for embed_char in embed_chars]  # T x 1 x hidden_dim2
        input_lstm_final = torch.cat((embed_word, hidden_chars), dim=2)  # T x 1 x (embed_word_dim + hidden_dim2)
        lstm_out, _ = self.lstm_final(input_lstm_final)  # T x 1 x (hidden_dim)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores
