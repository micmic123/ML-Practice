# (정밀도) scores의 top-k를 고른 뒤, 그들 중 targets에 있는 것의 비율
# 의미: 이 모델이 예측을 하면 그 예측이 얼마나 정확한지
def compute_precision(predictions, targets, k):
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    return float(num_hit) / len(pred)


# (재현율) scores의 top-k를 고른 뒤, 전체 target들 중 맞춘 것 비율
# 의미: 이 모델이 정답을 얼마나 빠뜨리지 않는지
def compute_recall(predictions, targets, k):
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    return float(num_hit) / len(targets)


def compute_ap(predictions, targets, k):
    if len(predictions) > k:
        predictions = predictions[:k]
    score = 0.
    num_hits = 0.
    for i, p in enumerate(predictions):
        if p in targets:
            num_hits += 1.
            score += num_hits / (i+1)
    return score / min(len(targets), k)


def evaluate(preds, gts, k=10, map_k=50):
    preds = (-preds).argsort()
    preds = preds.tolist()
    precs = [compute_precision(p, t, k=k) for (p, t) in zip(preds, gts)]
    recalls = [compute_recall(p, t, k=k) for (p, t) in zip(preds, gts)]
    aps = [compute_ap(p, t, k=map_k) for (p, t) in zip(preds, gts)]
    return float(sum(precs) / len(precs)), \
            float(sum(recalls) / len(recalls)), \
            float(sum(aps) / len(aps))
