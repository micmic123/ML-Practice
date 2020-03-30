import numpy as np
import torch


def filtering(users, scores, train_dict, k):
    """ filter out items in train_dict for each user and return intended real-recommended top-k items"""
    scores = (torch.topk(scores, k=(k+35), dim=1).indices + 1).tolist()
    scores = [[i for i in line if i not in train_dict[user]][:k] for user, line in zip(users, scores)]

    return scores


def hit_ratio(scores, labels):
    """ return average HR@k """
    total = [y in scores[i] for i, y in enumerate(labels)]
    # hr_avg = sum(total) / len(total)
    #
    # return round(hr_avg, 4)
    return sum(total)


def ndcg(scores, labels):
    """ return average NDCG@k """
    total = [1/(scores[i].index(y) + 1) for i, y in enumerate(labels) if y in scores[i]]
    # ndcg_avg = sum(total) / len(labels)
    #
    # return round(ndcg_avg, 4)
    return sum(total)


def eval(users, scores, labels, train_dict, k_ls):
    """ return average HR@k and NDCG@k """
    # top_k = torch.topk(scores, k=k, dim=1).indices
    top_k = np.array(filtering(users, scores, train_dict, k=k_ls[-1]))
    HR_ls, NDCG_ls = [], []
    for k in k_ls:
        scores = top_k[:, :k].tolist()
        HR_ls.append(hit_ratio(scores, labels))
        NDCG_ls.append(ndcg(scores, labels))

    return HR_ls, NDCG_ls