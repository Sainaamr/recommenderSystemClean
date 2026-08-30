"""
Split a raw dataset into historical and real-time portions with a GLOBAL TIME
CUT, so that no training interaction post-dates any streamed interaction.

  Historical — every interaction on or before the cut date → train LightGCN
  Real-time  — every interaction after the cut date        → simulate streaming

This differs from tools/split_dataset.py, which splits each user's own timeline
80/20. That per-user split leaves the historical set spanning the entire period,
so a chronologically-ordered stream is evaluated by a model that has already
seen the future: at the start of the stream ~99% of the training data post-dates
the interaction being predicted. A global cut removes that leakage entirely, at
the cost that users who only became active after the cut are absent from
training — which is the realistic cold-start population.

The cut is placed at the timestamp quantile that puts `ratio` of all
interactions in the historical portion.

Usage:
  python tools/split_dataset_timecut.py --dataset yelp
  python tools/split_dataset_timecut.py --dataset yelp --split 0.8
"""

import argparse
import sys
import os
from datetime import datetime, timezone
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
        "dataset":       "ml-1m",
        "config_files":  ["configs/dataset.yaml", "configs/historical_eval.yaml", "configs/lightgcn.yaml"],
        "hist_dir":      Path("dataset/ml-1m-historical-timecut"),
        "realtime_dir":  Path("dataset/ml-1m-realtime-timecut"),
        "hist_file":     "ml-1m-historical-timecut.inter",
        "realtime_file": "ml-1m-realtime-timecut.inter",
    },
    "yelp": {
        "dataset":       "yelp",
        # yelp_dataset_split.yaml keeps users from 3 interactions instead of 10,
        # so users with 3-9 interactions survive into the stream. Training later
        # loads the historical file with yelp_dataset.yaml ("[10, inf)"), which
        # excludes them from the model — making them cold-start users.
        "config_files":  ["configs/yelp_dataset_split.yaml", "configs/yelp_historical_eval.yaml", "configs/lightgcn.yaml"],
        "hist_dir":      Path("dataset/yelp-historical-timecut"),
        "realtime_dir":  Path("dataset/yelp-realtime-timecut"),
        "hist_file":     "yelp-historical-timecut.inter",
        "realtime_file": "yelp-realtime-timecut.inter",
    },
}
# ─────────────────────────────────────────────────────────────────────────────

def _d(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def split(dataset_key: str, ratio: float = 0.8):
    cfg = SPLIT_CONFIGS[dataset_key]

    print(f"Loading {cfg['dataset']} via RecBole (same filtering as training)...")
    config = Config(model="LightGCN", dataset=cfg["dataset"],
                    config_file_list=cfg["config_files"])
    dataset = create_dataset(config)

    df = dataset.inter_feat.copy()
    print(f"  Total interactions after filtering: {len(df):,}")
    print(f"  Users: {df['user_id'].nunique():,}, Items: {df['item_id'].nunique():,}")

    user_tokens = dataset.field2id_token["user_id"]
    item_tokens = dataset.field2id_token["item_id"]
    df["user_id"] = [user_tokens[i] for i in df["user_id"]]
    df["item_id"] = [item_tokens[i] for i in df["item_id"]]

    # ── the only structural difference: one global cut, in time ──────────────
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cut = df["timestamp"].quantile(ratio)
    historical = df[df["timestamp"] <= cut].reset_index(drop=True)
    realtime   = df[df["timestamp"] >  cut].reset_index(drop=True)

    print(f"\nGlobal time cut at {_d(cut)} (timestamp {int(cut)})")
    print(f"  Historical : {_d(historical['timestamp'].min())} .. {_d(historical['timestamp'].max())}")
    print(f"  Real-time  : {_d(realtime['timestamp'].min())} .. {_d(realtime['timestamp'].max())}")
    print("  No training interaction post-dates any streamed interaction.")

    ren = {"user_id": "user_id:token", "item_id": "item_id:token",
           "rating": "rating:float", "timestamp": "timestamp:float"}
    historical = historical.rename(columns=ren)
    realtime   = realtime.rename(columns=ren)

    hu = set(historical["user_id:token"]); ru = set(realtime["user_id:token"])
    hi = set(historical["item_id:token"]); ri = set(realtime["item_id:token"])
    print(f"\n  Historical : {len(historical):>9,} interactions  "
          f"({len(hu):,} users, {len(hi):,} items)")
    print(f"  Real-time  : {len(realtime):>9,} interactions  "
          f"({len(ru):,} users, {len(ri):,} items)")
    print(f"  Stream users never seen in training : {len(ru - hu):,} "
          f"({len(ru - hu) / len(ru) * 100:.1f}% of stream users)")
    print(f"  Stream items never seen in training : {len(ri - hi):,} "
          f"({len(ri - hi) / len(ri) * 100:.1f}% of stream items)")

    cfg["hist_dir"].mkdir(parents=True, exist_ok=True)
    cfg["realtime_dir"].mkdir(parents=True, exist_ok=True)
    hp = cfg["hist_dir"] / cfg["hist_file"]
    rp = cfg["realtime_dir"] / cfg["realtime_file"]
    historical.to_csv(hp, sep="\t", index=False)
    realtime.to_csv(rp, sep="\t", index=False)
    print(f"\nSaved → {hp}")
    print(f"Saved → {rp}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(SPLIT_CONFIGS), required=True)
    p.add_argument("--split", type=float, default=0.8,
                   help="Fraction of all interactions placed before the cut (default: 0.8)")
    args = p.parse_args()
    split(args.dataset, args.split)


if __name__ == "__main__":
    main()
