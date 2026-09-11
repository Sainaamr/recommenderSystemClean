"""
Write an UNFILTERED real-time stream for an existing time-cut split.

Same global time cut and the same rating filter as tools/split_dataset_timecut.py,
but WITHOUT the minimum-interaction filter. Users and items with only a handful
of interactions therefore survive into the stream instead of being removed.

Why: in the filtered stream, first-time users decline steadily over the run. The
filter is a plausible cause — a user arriving late has little time left to reach
10 lifetime interactions before the data ends, so late arrivals are
right-censored (survival through the >=10 filter falls from 1.82% for 2018-Q2
arrivals to 0.24% for 2019-Q4). Streaming the unfiltered data answers whether
the arrival rate is genuinely constant once that censoring is removed.

Only the stream is written. The historical portion is deliberately left alone,
so the existing checkpoint stays valid and nothing has to be retrained: the
question is who the trained model has never seen, which needs no new training.

The cut is taken from the first interaction of the existing real-time file, so
batch 1 of the unfiltered stream begins at exactly the same moment as batch 1 of
the filtered one and the two arrival curves are directly comparable.

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

    # mergesort is stable, so interactions sharing a timestamp keep their
    # original relative order and the first row is reproducible.
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
