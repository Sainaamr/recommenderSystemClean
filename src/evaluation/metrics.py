"""
Per-batch metric computation used by the streaming LightGCN experiment.
"""

import numpy as np

"""
scores: array of one number per item how much a user like eahc item
gt_items: set of row numbers of items user interacted with in the batch counts as correct responses
user_history: set of row numbers of items user interacted overall used to mask out old recommendations
ks: values of K
"""

def compute_metrics_at_ks(scores: np.ndarray, gt_items: set, user_history: set,
                          ks=(10, 20)) -> dict:
    """
    Compute Recall, NDCG, HR, MRR at each k in ks for one user.
    """

    # Masks already-seen items before ranking.
    scores = scores.copy()
    seen = [i for i in user_history if i < len(scores)]
    if seen:
        scores[seen] = -np.inf
    # give the top max_K items score wise ordered
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
        # N_rs = len(hits) (Herlocker et al. 2004, Table I). Recall's
        # denominator (N_r) is this batch's ground truth size — a prequential
        # approximation, not the full true relevant set (impractical to know
        # online) — so these numbers are comparative across strategies/batches,
        # not absolute. Precision's denominator (N_s) is just k, since exactly
        # k items are always selected.
        recall = len(hits) / max(len(gt_items), 1)
        precision = len(hits) / k
        hr = 1.0 if hits else 0.0
        dcg = sum(1.0 / np.log2(rank + 2)
                  for rank, item in enumerate(top_k) if item in gt_items)
        ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(gt_items), k)))
        ndcg = dcg / ideal if ideal > 0 else 0.0
        out[f"recall@{k}"]    = recall
        out[f"precision@{k}"] = precision
        out[f"ndcg@{k}"]      = ndcg
        out[f"hr@{k}"]        = hr
    return out


def _avg(results):
    if not results:
        keys = ["recall@10", "precision@10", "ndcg@10", "hr@10",
                "recall@20", "precision@20", "ndcg@20", "hr@20", "mrr"]
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


#  Recall@K computation (unused — superseded by batch_metrics_lgcn above,
#  kept for the worked example in its docstring/commentary below)
#
#   Batch 20 — NO UPDATE:
#
#   Model embeddings: unchanged from historical training.
#
#   Model's top-10 scores (before masking):
#   italian_1 → 0.95  ← already visited, MASKED to -inf
#   italian_2 → 0.90  ← already visited, MASKED to -inf
#   italian_3 → 0.85
#   italian_4 → 0.80
#   sushi_2   → 0.75
#   italian_5 → 0.70
#   burger_1  → 0.65
#   pizza_1   → 0.60
#   pizza_2   → 0.55
#   pizza_3   → 0.50
#   pizza_4   → 0.45
#
#   After masking, top-10:
#   {italian_3, italian_4, sushi_2, italian_5, burger_1, pizza_1, pizza_2, pizza_3, pizza_4, italian_6}
#
#   Ground truth (batch 20): {sushi_2, sushi_9}
#
#   Hits: {sushi_2} → 1 hit
#
#   Recall@10 = 1/2 = 0.5
#
#   Model never updates → embeddings stay the same forever.
#
#   ---
#   Batch 40 — NO UPDATE:
#
#   Model embeddings: still unchanged from historical training.
#
#   Model's top-10 scores (before masking):
#   italian_1 → 0.95  ← already visited, MASKED to -inf
#   italian_2 → 0.90  ← already visited, MASKED to -inf
#   italian_3 → 0.85
#   italian_4 → 0.80
#   sushi_2   → 0.75  ← now also visited (from batch 20), MASKED to -inf
#   italian_5 → 0.70
#   burger_1  → 0.65
#   pizza_1   → 0.60
#   pizza_2   → 0.55
#   pizza_3   → 0.50
#   pizza_4   → 0.45
#   italian_6 → 0.40
#
#   After masking, top-10:
#   {italian_3, italian_4, italian_5, burger_1, pizza_1, pizza_2, pizza_3, pizza_4, italian_6, italian_7}
#
#   Ground truth (batch 40): {sushi_5, sushi_9} ← user has shifted to sushi
#
#   Hits: {} → 0 hits
#
#   Recall@10 = 0/2 = 0.0
#
#   The scores didn't change at all — but the ground truth did because the user's
#   real behaviour shifted. The model keeps recommending Italian while the user now wants sushi.
#
#   ---
#   Batch 20 — INCREMENTAL:
#
#   Same as no_update at this point — recall measured before update:
#   Recall@10 = 1/2 = 0.5
#
#   Then update triggers → model sees 20,000 recent interactions, lots of sushi visits → embeddings shift toward sushi.
#
#   ---
#   Batch 40 — INCREMENTAL:
#
#   Model embeddings: updated — now leans toward sushi.
#
#   Model's top-10 scores (before masking):
#   italian_1 → 0.95  ← already visited, MASKED to -inf
#   italian_2 → 0.90  ← already visited, MASKED to -inf
#   sushi_2   → 0.88  ← already visited (from batch 20), MASKED to -inf
#   sushi_5   → 0.85  ← score went up after update
#   sushi_9   → 0.82  ← score went up after update
#   italian_3 → 0.75
#   sushi_3   → 0.70
#   burger_1  → 0.60
#   pizza_1   → 0.55
#   pizza_2   → 0.50
#   pizza_3   → 0.45
#
#   After masking, top-10:
#   {sushi_5, sushi_9, italian_3, sushi_3, burger_1, pizza_1, pizza_2, pizza_3, italian_4, italian_5}
#
#   Ground truth (batch 40): {sushi_5, sushi_9}
#
#   Hits: {sushi_5, sushi_9} → 2 hits
#   Recall@10 = 2/2 = 1.0
#
def recall_at_k(model, user_ids: list, item_ids: list,
                user_history: dict, k: int = 10) -> float:
    """
    For each user in user_ids, recommend top-K items (excluding seen history),
    then compute Recall@K against item_ids (ground truth for this batch).
    Returns average Recall@K across users.
    """
    import torch
    model.eval()
    # don't save gradients not needed
    with torch.no_grad():
        user_all_emb, item_all_emb = model.forward()

    # Group ground truth items by user
    # user_gt = {1: {10, 30}, 2: {20}, 3: {40}}
    user_gt = {}
    for uid, iid in zip(user_ids, item_ids):
        if uid not in user_gt:
            user_gt[uid] = set()
        user_gt[uid].add(iid)

    recalls = []

    for uid, gt_items in user_gt.items():

        # Safeguard in case the user id is bigger than total number of users
        if uid >= user_all_emb.shape[0]:
            continue

        # multiply one user's embedding with all item embeddings → how much this user likes each item
        scores = torch.matmul(user_all_emb[uid], item_all_emb.T).cpu().numpy()

        # Mask already-seen items so they can never appear in recommendations
        # scores     = [0.9, 0.3, 0.8, 0.95, 0.1]
        # seen       = {0, 2}  ← user already saw items 0 and 2
        # after mask = [-inf, 0.3, -inf, 0.95, 0.1]
        seen = user_history.get(uid, set())
        if seen:
            scores[list(seen)] = -np.inf

        # top_k    = {1, 3}    ← model recommended these
        # gt_items = {3, 7}    ← user actually liked these in this batch
        # hits     = 1         ← only item 3 was correctly recommended
        top_k = set(np.argpartition(scores, -k)[-k:])
        hits = len(top_k & gt_items)
        recalls.append(hits / max(len(gt_items), 1))

    return float(np.mean(recalls)) if recalls else 0.0
