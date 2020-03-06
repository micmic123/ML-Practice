# Metrics
import torch


def filtering(scores, train_dict, k):
    """ filter out items in train_dict for each user and return intended real-recommended top-k items"""
    scores = scores.argsort(dim=1, descending=True).tolist()
    scores = [[i for i in line[:len(train_dict[user]) + k] if i not in train_dict[user]][:k] for user, line in enumerate(scores)]

    return scores


def hit_ratio(scores, labels):
    """ return average HR@k """
    total = [y in scores[i] for i, y in enumerate(labels)]
    hr_avg = sum(total) / len(total)

    return hr_avg


def ndcg(scores, labels):
    """ return average NDCG@k """

    return None


def eval(scores, labels, train_dict, k=50):
    """ return average HR@k and NDCG@k """
    # top_k = torch.topk(scores, k=k, dim=1).indices
    top_k = filtering(scores, train_dict, k=k)
    HR, NDCG = hit_ratio(top_k, labels), ndcg(top_k, labels)

    return HR, NDCG