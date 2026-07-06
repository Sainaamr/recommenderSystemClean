"""
Split a raw dataset into historical and real-time portions using RecBole's
dataset loader for filtering (rating>=3, min interactions, etc.), then a
per-user time-ordered split on the filtered data.

  Historical (80%)  — first 80% of each user's interactions → train LightGCN
  Real-time  (20%)  — last 20% of each user's interactions  → simulate streaming

Every user appears in both splits — no user gets dropped.

Usage:
  python tools/split_dataset.py --dataset ml-1m
  python tools/split_dataset.py --dataset yelp --split 0.8
"""

import argparse
import sys
import os
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset

# ── Per-dataset settings ──────────────────────────────────────────────────────
SPLIT_CONFIGS = {
    "ml-1m": {
        "dataset":      "ml-1m",
        "config_files": ["configs/dataset.yaml", "configs/historical_eval.yaml", "configs/lightgcn.yaml"],
        "hist_dir":      Path("dataset/ml-1m-historical"),
        "realtime_dir":  Path("dataset/ml-1m-realtime"),
        "hist_file":     "ml-1m-historical.inter",
        "realtime_file": "ml-1m-realtime.inter",
    },
    "yelp": {
        "dataset":      "yelp",
        "config_files": ["configs/yelp_dataset.yaml", "configs/yelp_historical_eval.yaml", "configs/lightgcn.yaml"],
        "hist_dir":      Path("dataset/yelp-historical"),
        "realtime_dir":  Path("dataset/yelp-realtime"),
        "hist_file":     "yelp-historical.inter",
        "realtime_file": "yelp-realtime.inter",
    },
}
# ─────────────────────────────────────────────────────────────────────────────


def split(dataset_key: str, ratio: float = 0.8):
    cfg = SPLIT_CONFIGS[dataset_key]

    print(f"Loading {cfg['dataset']} via RecBole (applies same filtering as training)...")
    config = Config(
        model="LightGCN",
        dataset=cfg["dataset"],
        config_file_list=cfg["config_files"],
    )
    dataset = create_dataset(config)

    # inter_feat is a DataFrame with internal integer IDs
    df = dataset.inter_feat.copy()
    print(f"  Total interactions after filtering: {len(df)}")
    print(f"  Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")

    # Convert internal IDs back to original tokens
    user_tokens = dataset.field2id_token["user_id"]  # index=internal_id, value=token
    item_tokens = dataset.field2id_token["item_id"]
    df["user_id"] = [user_tokens[i] for i in df["user_id"]]
    df["item_id"] = [item_tokens[i] for i in df["item_id"]]

    # Sort each user's interactions by timestamp
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Per-user split
    historical_rows = []
    realtime_rows   = []

    for uid, group in df.groupby("user_id", sort=False):
        cutoff = max(1, int(len(group) * ratio))
        historical_rows.append(group.iloc[:cutoff])
        realtime_rows.append(group.iloc[cutoff:])

    historical = pd.concat(historical_rows).reset_index(drop=True)
    realtime   = pd.concat(realtime_rows).reset_index(drop=True)

    # Rename columns to RecBole .inter format
    historical = historical.rename(columns={
        "user_id": "user_id:token", "item_id": "item_id:token",
        "rating": "rating:float", "timestamp": "timestamp:float",
    })
    realtime = realtime.rename(columns={
        "user_id": "user_id:token", "item_id": "item_id:token",
        "rating": "rating:float", "timestamp": "timestamp:float",
    })

    print(f"\nPer-user {ratio*100:.0f}/{(1-ratio)*100:.0f} split:")
    print(f"  Historical : {len(historical):>7} interactions  "
          f"({historical['user_id:token'].nunique()} users, "
          f"{historical['item_id:token'].nunique()} items)")
    print(f"  Real-time  : {len(realtime):>7} interactions  "
          f"({realtime['user_id:token'].nunique()} users, "
          f"{realtime['item_id:token'].nunique()} items)")

    # Save
    cfg["hist_dir"].mkdir(parents=True, exist_ok=True)
    hist_path = cfg["hist_dir"] / cfg["hist_file"]
    historical.to_csv(hist_path, sep="\t", index=False)
    print(f"\nSaved → {hist_path}")

    cfg["realtime_dir"].mkdir(parents=True, exist_ok=True)
    rt_path = cfg["realtime_dir"] / cfg["realtime_file"]
    realtime.to_csv(rt_path, sep="\t", index=False)
    print(f"Saved → {rt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(SPLIT_CONFIGS), required=True,
                   help="Which dataset to split")
    p.add_argument("--split", type=float, default=0.8,
                   help="Fraction of each user's interactions for historical (default: 0.8)")
    args = p.parse_args()
    split(args.dataset, args.split)


if __name__ == "__main__":
    main()
