"""
Per-batch metric computation used by the streaming LightGCN experiment.
"""

import numpy as np


def compute_metrics_at_ks(scores: np.ndarray, gt_items: set, user_history: set,
                          ks=(10, 20)) -> dict:
    """
    Compute Recall, NDCG, HR, MRR at each k in ks for one user.
    Masks already-seen items before ranking. Single forward pass.
    """
    scores = scores.copy()
    seen = [i for i in user_history if i < len(scores)]
    if seen:
        scores[seen] = -np.inf

    max_k = max(ks)
    top_max = list(np.argpartition(scores, -max_k)[-max_k:])
    top_max_sorted = sorted(top_max, key=lambda x: scores[x], reverse=True)

    # MRR is k-independent (rank of first hit in full sorted list)
    mrr = 0.0
    for rank, item in enumerate(top_max_sorted):
        if item in gt_items:
            mrr = 1.0 / (rank + 1)
            break

    out = {"mrr": mrr}
    for k in ks:
        top_k = top_max_sorted[:k]
        hits = set(top_k) & gt_items
        recall = len(hits) / max(len(gt_items), 1)
        hr = 1.0 if hits else 0.0
        dcg = sum(1.0 / np.log2(rank + 2)
                  for rank, item in enumerate(top_k) if item in gt_items)
        ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(gt_items), k)))
        ndcg = dcg / ideal if ideal > 0 else 0.0
        out[f"recall@{k}"] = recall
        out[f"ndcg@{k}"]   = ndcg
        out[f"hr@{k}"]     = hr
    return out


def _avg(results):
    if not results:
        keys = ["recall@10", "ndcg@10", "hr@10", "recall@20", "ndcg@20", "hr@20", "mrr"]
        return {k: 0.0 for k in keys}
    return {m: float(np.mean([r[m] for r in results])) for m in results[0]}


def batch_metrics_lgcn(model, user_ids, item_ids, user_history, k=10, ks=(10, 20)):
    """Compute averaged metrics for a batch using a LightGCN model."""
    import torch
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.forward()

    user_gt = {}
    for uid, iid in zip(user_ids, item_ids):
        user_gt.setdefault(uid, set()).add(iid)

    results = [compute_metrics_at_ks(
                   torch.matmul(user_emb[uid], item_emb.T).cpu().numpy(),
                   gt, user_history.get(uid, set()), ks)
               for uid, gt in user_gt.items() if uid < user_emb.shape[0]]

    return _avg(results)
