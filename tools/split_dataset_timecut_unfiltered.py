"""
Usage:
  python tools/split_dataset_timecut_unfiltered.py --dataset yelp
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

CONFIGS = {
    "yelp": {
        "raw":          Path("dataset/yelp/yelp.inter"),
        "reference":    Path("dataset/yelp-realtime-timecut/yelp-realtime-timecut.inter"),
        "out_dir":      Path("dataset/yelp-realtime-timecut-unfiltered"),
        "out_file":     "yelp-realtime-timecut-unfiltered.inter",
        "min_rating":   3.0,
    },
    "ml-1m": {
        "raw":          Path("dataset/ml-1m/ml-1m.inter"),
        "reference":    Path("dataset/ml-1m-realtime-timecut/ml-1m-realtime-timecut.inter"),
        "out_dir":      Path("dataset/ml-1m-realtime-timecut-unfiltered"),
        "out_file":     "ml-1m-realtime-timecut-unfiltered.inter",
        "min_rating":   3.0,
    },
}

U, I, R, T = "user_id:token", "item_id:token", "rating:float", "timestamp:float"


def _d(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def split(dataset_key: str):
    cfg = CONFIGS[dataset_key]

    ref = pd.read_csv(cfg["reference"], sep="\t")
    cut_ts = float(ref[T].iloc[0])
    first_user = str(ref[U].iloc[0])
    print(f"Reference stream : {cfg['reference']}")
    print(f"  starts at {_d(cut_ts)} (timestamp {cut_ts:.0f}), user {first_user}")
    print(f"  {len(ref):,} interactions, {ref[U].nunique():,} users\n")

    raw = pd.read_csv(cfg["raw"], sep="\t")
    print(f"Raw dataset      : {cfg['raw']}")
    print(f"  {len(raw):,} interactions, {raw[U].nunique():,} users, {raw[I].nunique():,} items")

    kept = raw[raw[R] >= cfg["min_rating"]]
    print(f"  after rating >= {cfg['min_rating']:g}: {len(kept):,} interactions, "
          f"{kept[U].nunique():,} users, {kept[I].nunique():,} items")


    stream = kept[kept[T] >= cut_ts].sort_values(T, kind="mergesort").reset_index(drop=True)

    print(f"\nUnfiltered stream (no minimum-interaction filter)")
    print(f"  {len(stream):,} interactions, {stream[U].nunique():,} users, {stream[I].nunique():,} items")
    print(f"  {_d(stream[T].min())} .. {_d(stream[T].max())}")
    print(f"  {len(stream) / len(ref):.1f}x the interactions and "
          f"{stream[U].nunique() / ref[U].nunique():.1f}x the users of the filtered stream")

    got = str(stream[U].iloc[0])
    if got == first_user:
        print(f"  first interaction matches the filtered stream (user {got})")
    else:
        print(f"  NOTE: first row is user {got}, not {first_user} — same timestamp, "
              f"different tie order; batch 1 still starts at the same instant")

    per_user = stream[U].value_counts()
    print(f"\n  interactions per user: median {per_user.median():.0f}, "
          f"mean {per_user.mean():.2f}, max {per_user.max():,}")
    print(f"  users with a single interaction: {(per_user == 1).sum():,} "
          f"({(per_user == 1).mean() * 100:.1f}%)")

    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    out = cfg["out_dir"] / cfg["out_file"]
    stream.to_csv(out, sep="\t", index=False)
    print(f"\nSaved → {out}")
    print("Historical portion unchanged — the existing checkpoint still applies.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(CONFIGS), required=True)
    args = p.parse_args()
    split(args.dataset)


if __name__ == "__main__":
    main()
