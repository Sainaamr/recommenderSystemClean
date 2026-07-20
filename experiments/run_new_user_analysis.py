"""
New user drift analysis.

Isolates the impact of new users on LightGCN recall over the streaming period.
Model never retrains. New users receive mean-initialised embeddings (cold-start fallback).

Three groups scored per batch:
  existing : uid < n_users_trained
  new_user : uid >= n_users_trained (mean embedding)
  overall  : all users combined

Usage:
  python experiments/run_new_user_analysis.py --dataset yelp
  python experiments/run_new_user_analysis.py --dataset ml-1m
  python experiments/run_new_user_analysis.py --dataset yelp --csv results/existing.csv
"""

import os, sys, argparse, torch
from pathlib import Path
from datetime import datetime

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics_at_ks, _avg
from src.models.incremental_lightgcn import IncrementalLightGCN
from experiments.run_incremental_lightgcn import (
    DATASET_CONFIGS, RESULTS_DIR, BATCH_SIZE,
    load_id_mappings, build_user_history, train_historical,
)
from tools.plot_utils import plot_new_user_analysis


#  Per-batch scoring

def score_batch(lgcn, existing_gt, new_user_gt, history, k=10):
    """
    Single forward pass. Scores an already-split existing/new_user ground
    truth dict pair separately. Returns (metrics_existing, metrics_new_user,
    metrics_overall) — same shape batch_metrics_lgcn returns, just computed
    separately per group instead of for everyone combined. Classifying which
    uid is "existing" vs "new" is the caller's job (run_new_user_analysis) —
    this function only scores whatever split it's handed.
    """
    lgcn.eval()
    with torch.no_grad():
        user_emb, item_emb = lgcn.forward()

    # works similar to batch_metrics_lgcn without the forward call
    def score_group(user_gt):
        results = []
        for uid, gt in user_gt.items():
            if uid >= user_emb.shape[0]:
                continue
            scores = torch.matmul(user_emb[uid], item_emb.T).cpu().numpy()
            results.append(compute_metrics_at_ks(scores, gt, history.get(uid, set()), (k,)))
        return results

    existing_results = score_group(existing_gt)
    new_user_results = score_group(new_user_gt)
    all_results       = existing_results + new_user_results

    return (
        _avg(existing_results),
        _avg(new_user_results),
        _avg(all_results),
    )


#Streaming loop

def run_new_user_analysis(cfg: dict, ckpt: str) -> pd.DataFrame:
    user2id, item2id, config, dataset = load_id_mappings(cfg)
    lgcn = IncrementalLightGCN.from_checkpoint(ckpt, config, dataset)
    history = build_user_history(user2id, item2id, cfg)

    # number of users trained on
    n_users_trained = lgcn.n_users

    # realtime_path is pre-filtered to rating>=3 by tools/split_dataset.py
    df_rt = pd.read_csv(cfg["realtime_path"], sep="\t")

    id_cast  = cfg["id_cast"]
    next_uid = [lgcn.n_users]
    next_iid = [lgcn.n_items]

    def get_uid(x):
        key = id_cast(x)
        if key not in user2id:
            user2id[key] = next_uid[0]; next_uid[0] += 1
        return user2id[key]

    def get_iid(x):
        key = id_cast(x)
        if key not in item2id:
            item2id[key] = next_iid[0]; next_iid[0] += 1
        return item2id[key]

    df_rt["uid"] = df_rt["user_id:token"].apply(get_uid).astype(int)
    df_rt["iid"] = df_rt["item_id:token"].apply(get_iid).astype(int)

    n_batches = len(df_rt) // BATCH_SIZE
    records   = []

    # Tracks every uid ever classified as "new" so far, so a user whose
    # interactions straddle a batch boundary
    seen_as_new = set()

    print(f"  Streaming {n_batches} batches (no retraining)...")

    for i in range(n_batches):
        batch       = df_rt.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_users = batch["uid"].tolist()
        batch_items = batch["iid"].tolist()

        # Expand embeddings so new users get mean embeddings and can be scored
        max_u = max(batch_users, default=0)
        max_i = max(batch_items, default=0)
        if max_u >= lgcn.n_users or max_i >= lgcn.n_items:
            lgcn.expand_embeddings(
                max(max_u + 1, lgcn.n_users),
                max(max_i + 1, lgcn.n_items),
            )

        # Split this batch's ground truth by existing vs new user — the
        # classification itself lives here, not inside score_batch, so that
        # function only has to worry about scoring whatever split it's given.
        existing_gt = {}
        new_user_gt = {}
        for uid, iid in zip(batch_users, batch_items):
            if uid >= n_users_trained:
                new_user_gt.setdefault(uid, set()).add(iid)
            else:
                existing_gt.setdefault(uid, set()).add(iid)

        m_existing, m_new_user, m_overall = score_batch(
            lgcn, existing_gt, new_user_gt, history,
        )

        new_user_set = set(new_user_gt.keys())
        first_time_new = new_user_set - seen_as_new
        seen_as_new |= new_user_set
        n_new_users = len(first_time_new)

        pct_new_user = n_new_users / max(len(set(batch_users)), 1)

        for uid, iid in zip(batch_users, batch_items):
            history.setdefault(uid, set()).add(iid)

        records.append({
            "batch":              i + 1,
            "interactions":       (i + 1) * BATCH_SIZE,
            "recall_existing":    m_existing["recall@10"],
            "precision_existing": m_existing["precision@10"],
            "ndcg_existing":      m_existing["ndcg@10"],
            "recall_new_user":    m_new_user["recall@10"],
            "precision_new_user": m_new_user["precision@10"],
            "ndcg_new_user":      m_new_user["ndcg@10"],
            "recall_overall":     m_overall["recall@10"],
            "precision_overall":  m_overall["precision@10"],
            "ndcg_overall":       m_overall["ndcg@10"],
            "pct_new_user":       pct_new_user,
            "n_new_users":        n_new_users,
        })

        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1:>3}/{n_batches}  "
                  f"existing={m_existing['recall@10']:.4f}  "
                  f"new_user={m_new_user['recall@10']:.4f}  "
                  f"overall={m_overall['recall@10']:.4f}  "
                  f"pct_new={pct_new_user:.2f}  "
                  f"n_new={n_new_users}")

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-1m", "yelp"], required=True)
    parser.add_argument("--csv", type=Path, default=None,
                        help="Existing results CSV — skip streaming, just re-plot")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg = DATASET_CONFIGS[args.dataset].copy()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix  = args.dataset.replace("-", "")
    out_csv = results_dir / f"{prefix}_new_user_analysis_{ts}.csv"
    out_png = results_dir / f"{prefix}_new_user_analysis_{ts}.png"

    if args.csv:
        df = pd.read_csv(args.csv)
        out_png = Path(str(args.csv).replace(".csv", "_replot.png"))
        print(f"Loaded existing results from {args.csv}")
    else:
        print("\n── Loading LightGCN checkpoint ──────────────────────────────────────")
        ckpt, _ = train_historical(cfg)

        print("\n── Running new user analysis (no retraining) ────────────────────────")
        df = run_new_user_analysis(cfg, ckpt)

        df.to_csv(out_csv, index=False)
        print(f"\nResults saved → {out_csv}")

    plot_new_user_analysis(df, out_png,
                           f"New User Drift Analysis — {args.dataset} (no retraining)",
                           batch_size=BATCH_SIZE)

    print(f"\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Avg recall — existing users: {df['recall_existing'].mean():.4f}")
    print(f"  Avg recall — new users:      {df['recall_new_user'].mean():.4f}")
    print(f"  Avg recall — overall:        {df['recall_overall'].mean():.4f}")
    print(f"  Avg % new users per batch:   {df['pct_new_user'].mean():.2%}")
    print(f"  Total unique new users seen: {df['n_new_users'].sum():.0f}")


if __name__ == "__main__":
    main()
