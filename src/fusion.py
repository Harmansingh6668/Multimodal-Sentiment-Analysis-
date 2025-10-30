from typing import Dict, List
LABELS = ["negative", "neutral", "positive"]

def fuse_many(distributions: List[Dict[str, float]], weights: List[float]) -> Dict[str, float]:
    assert len(distributions) == len(weights)
    # If a distribution is all zeros, set its weight to zero
    adj_weights = []
    for d, w in zip(distributions, weights):
        if sum(d.values()) == 0.0:
            adj_weights.append(0.0)
        else:
            adj_weights.append(max(0.0, w))
    s = sum(adj_weights)
    if s == 0:
        return {k: 0.0 for k in LABELS}
    adj_weights = [w/s for w in adj_weights]

    fused = {k: 0.0 for k in LABELS}
    for d, w in zip(distributions, adj_weights):
        for k in LABELS:
            fused[k] += w * d.get(k, 0.0)

    total = sum(fused.values())
    if total > 0:
        for k in fused:
            fused[k] /= total
    return fused

def argmax_label(proba: Dict[str, float]) -> str:
    return max(proba.items(), key=lambda x: x[1])[0]


def fuse_distributions(dists, weights):
    fused = {}
    total_weight = sum(weights)
    for dist, w in zip(dists, weights):
        for k, v in dist.items():
            fused[k] = fused.get(k, 0.0) + v * (w / total_weight)
    return fused

def argmax_label(dist):
    return max(dist, key=dist.get)
