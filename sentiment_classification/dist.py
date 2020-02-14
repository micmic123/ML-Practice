import dill
import torch
import torch.nn.functional as F
from konlpy.tag import Mecab
from model import *


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
tokenizer = Mecab()
model, vocab = None, None


def get_saved_state(model_name):
    saved_state = torch.load(f'snapshot/{model_name}.pt')
    with open(f'snapshot/{model_name}.hp') as f:
        saved_hp = dill.load(f)
    with open('snapshot/fields.Field', 'rb') as f:
        saved_fields = dill.load(f)

    return saved_state, saved_hp, saved_fields


def get_model_vocab(model_cls, target='best'):
    saved_state, saved_hp, saved_fields = get_saved_state(target)
    model = model_cls(saved_hp)
    model.load_state_dict(saved_state)
    model.to(DEVICE)
    vocab = saved_fields['TEXT'].vocab

    return model, vocab


def preprocessing(sentence):
    sentence = sentence.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '').strip()
    sentence = tokenizer.morphs(sentence)

    if sentence == '':
        raise Exception()
    print(f'[INFO] tokenized sentence: {sentence}')
    sentence = [vocab.stoi[word] for word in sentence]

    return torch.tensor(sentence).view(1, -1).to(DEVICE)


def predict(sentence):
    input = preprocessing(sentence)
    score = model(input)
    pred = score.max(dim=1, keepdim=True)[1].item()
    prob = F.softmax(score, dim=1)[0][pred]
    result = 'positive' if pred == 1 else 'negative'
    print(f'The sentiment of the statement is [{result}] with p={prob:.4f}')


if __name__ == '__main__':
    model, vocab = get_model_vocab(SimpleGRU)
    flag = 'exit'
    while True:
        s = input()
        if s == flag:
            break

        predict(input())
