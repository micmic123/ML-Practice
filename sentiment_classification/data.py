import requests, os, re, random
import pandas as pd
import numpy as np
import dill
import torch
from torchtext import data
from torchtext.data import TabularDataset, BucketIterator
from konlpy.tag import Mecab


path_base = os.path.expanduser('~/dataset/naver_movie_review')
if not os.path.isdir(path_base):
    os.makedirs(path_base)
data_ls = os.listdir(path_base)
stopwords = {'의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다'}
tokenizer = Mecab()
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f'{DEVICE} will be used')


def get_raw_data():
    if not os.path.isdir(path_base):
        os.mkdir(path_base)

    # download raw data only when there are no data in path_base
    if 'train.txt' not in data_ls:
        with open(os.path.join(path_base, 'train.txt'), 'wb') as f:
            f.write(requests.get('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt').content)
    if 'test.txt' not in data_ls:
        with open(os.path.join(path_base, 'test.txt'), 'wb') as f:
            f.write(requests.get('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt').content)

    train_df = pd.read_table(os.path.join(path_base, 'train.txt'))
    test_df = pd.read_table(os.path.join(path_base, 'test.txt'))

    return train_df, test_df


def raw2pre(train_df, test_df):
    if 'train_pre.csv' in data_ls and 'test_pre.csv' in data_ls:
        return

    # leave only Korean characters
    train_df['document'] = train_df['document'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '').str.strip().replace(r'^\s*$', np.nan, regex=True)
    test_df['document'] = test_df['document'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '').str.strip().replace(r'^\s*$', np.nan, regex=True)

    # remove row with null
    train_df = train_df.dropna(how='any')
    test_df = test_df.dropna(how='any')

    # save as csv file
    train_df.to_csv(os.path.join(path_base, 'train_pre.csv'), index=False)
    test_df.to_csv(os.path.join(path_base, 'test_pre.csv'), index=False)


def preprocessing(sentence):
    # sentence = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', sentence)
    sentence = tokenizer.morphs(sentence)
    # sentence = [word for word in sentence if word not in stopwords]
    return sentence


# return Dataloader and Fields
def get_data_helper(batch_size=100, fix_length=None, min_req=10, max_size=10000):
    # define Field
    ID = data.Field(sequential=False, use_vocab=False)
    # fix_length of TEXT is not necessary: refer to BucketIterator
    TEXT = data.Field(sequential=True, use_vocab=True, tokenize=preprocessing, lower=True, batch_first=True,
                      fix_length=fix_length)
    LABEL = data.Field(sequential=False, use_vocab=False, is_target=True, batch_first=True)
    fields = {
        'ID': ID,
        'TEXT': TEXT,
        'LABEL': LABEL
    }

    # make Dataset
    trainset, testset = TabularDataset.splits(path=path_base, train='train_pre.csv', test='test_pre.csv',
                                              format='csv', fields=[('id', ID), ('text', TEXT), ('label', LABEL)],
                                              skip_header=True)
    trainset, valset = trainset.split(split_ratio=0.8)
    print(f'{"="*10} Loading data succeeded {"="*10}')
    print(f'# train: {len(trainset)}\n# val: {len(valset)}\n# test:{len(testset)}')

    # make Vocab
    TEXT.build_vocab(trainset, min_freq=min_req, max_size=max_size)
    # TEXT.build_vocab(valset, min_freq=min_req, max_size=max_size)
    print(f'# vocab: {len(TEXT.vocab)}')

    # make Dataloader
    train_loader, val_loader, test_loader = BucketIterator.splits((trainset, valset, testset), batch_size=batch_size,
                                                                  sort_key=lambda x: len(x.text), shuffle=True, repeat=False)
    print(f'# train_batch: {len(train_loader)}\n# val_batch: {len(val_loader)}\n# test_batch:{len(test_loader)}')

    return train_loader, val_loader, test_loader, fields


# return Dataloader and vocab_size
def get_data(batch_size=64, fix_length=None, min_req=10, max_size=10000):
    train, test = get_raw_data()
    raw2pre(train, test)
    train_loader, val_loader, test_loader, fields = get_data_helper(batch_size, fix_length, min_req, max_size)
    vocab_size = len(fields['TEXT'].vocab)
    save_fields(fields)

    return train_loader, val_loader, test_loader, vocab_size


def save_fields(fields):
    if not os.path.isdir('snapshot'):
        os.mkdir('snapshot')
    with open('snapshot/fields.Field', 'wb') as f:
        dill.dump(fields, f)


if __name__ == '__main__':
    train, test = get_raw_data()
    raw2pre(train, test)
    # _, _, _, fields = get_data_helper()
    # save_fields(fields)
