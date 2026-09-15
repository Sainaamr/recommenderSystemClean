"""
Usage:
    python3 tools/new_user_share.py                          
    python3 tools/new_user_share.py --dataset yelp-timecut
    python3 tools/new_user_share.py --out share.csv --latex share.tex

Flag:
 --dataset --batch-size  --update-every --out --latex
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from experiments.run_incremental_lightgcn import (
    DATASET_CONFIGS, RESULTS_DIR, BATCH_SIZE, UPDATE_EVERY, load_id_mappings,
)

DEFAULT_DATASETS = ["yelp-timecut", "ml-1m-timecut"]


def new_user_share(dataset_key: str, batch_size: int = BATCH_SIZE,
                   update_every: int = UPDATE_EVERY) -> pd.DataFrame:

    cfg = DATASET_CONFIGS[dataset_key]
    user2id, _, _, _ = load_id_mappings(cfg)

    df = pd.read_csv(cfg["realtime_path"], sep="\t")
    cast = cfg["id_cast"]
    users = df["user_id:token"].apply(cast)


    untrained = ~users.isin(user2id.keys())

    n_batches = len(users) // batch_size
    kept = n_batches * batch_size            # trailing partial batch is not streamed
    u, unt = users.iloc[:kept], untrained.iloc[:kept]


    n_trained = len(user2id) - 1
    print(f"  Trained users:  {n_trained:,}")
    print(f"  Stream rows:    {len(df):,} ({n_batches} batches of {batch_size}, "
          f"{len(df) - kept} trailing rows unused)")


    rows = [
        {"scope": "stream", "stat": "count", "definition": "untrained", "unit": "users",
         "n_new": u[unt].nunique(), "n_total": u.nunique()},
        {"scope": "stream", "stat": "count", "definition": "untrained", "unit": "interactions",
         "n_new": int(unt.sum()), "n_total": len(u)},
    ]

    seen, per_window = set(), []
    for start in range(0, kept, update_every * batch_size):
        wu = u.iloc[start:start + update_every * batch_size]
        wn = unt.iloc[start:start + update_every * batch_size]
        n_first, inter_first = 0, 0
        for b in range(0, len(wu), batch_size):
            bu, bn = wu.iloc[b:b + batch_size], wn.iloc[b:b + batch_size]
            firsts = set(bu[bn]) - seen
            seen |= firsts
            n_first += len(firsts)
            inter_first += int(bu.isin(firsts).sum())
        cold = set(wu[wn])
        seen |= cold
        per_window.append({
            "users_first_appearance":        n_first,
            "users_untrained":               len(cold),
            "users_total":                   wu.nunique(),
            "inter_first_appearance":        inter_first,
            "inter_untrained":               int(wn.sum()),
            "inter_total":                   len(wu),
        })
    w = pd.DataFrame(per_window)
    print(f"  Windows:        {len(w)} of {update_every} batches "
          f"(last holds {w['inter_total'].iloc[-1] // batch_size})")

    for unit, total in [("users", "users_total"), ("interactions", "inter_total")]:
        for definition in ["first_appearance", "untrained"]:
            col = f"{'users' if unit == 'users' else 'inter'}_{definition}"
            rows.append({"scope": "window", "stat": "mean", "definition": definition,
                         "unit": unit, "n_new": w[col].mean(), "n_total": w[total].mean()})


    per_user = pd.DataFrame({"u": u[unt], "batch": pd.Series(range(kept))[unt.values] // batch_size})
    first_batch = per_user.groupby("u").batch.min()
    total_each = per_user.groupby("u").size()
    in_first = (per_user[per_user.batch == per_user.u.map(first_batch)]
                .groupby("u").size())
    share_first = (in_first / total_each * 100)
    n_batches_active = per_user.groupby("u").batch.nunique()
    one_shot = int((n_batches_active == 1).sum())

    rows += [
        {"scope": "per_user", "stat": "median", "definition": "untrained",
         "unit": "interactions_in_first_batch",
         "n_new": in_first.median(), "n_total": total_each.median(),
         "pct_override": share_first.median()},
        {"scope": "per_user", "stat": "count", "definition": "untrained",
         "unit": "users_never_returning",
         "n_new": one_shot, "n_total": len(total_each),
         "pct_override": one_shot / len(total_each) * 100},
    ]

    out = pd.DataFrame(rows)

    pct = []
    for r in rows:
        if "pct_override" in r:
            pct.append(r.pop("pct_override"))
        elif r["scope"] == "stream":
            pct.append(r["n_new"] / r["n_total"] * 100)
        else:
            col = f"{'users' if r['unit'] == 'users' else 'inter'}_{r['definition']}"
            tot = "users_total" if r["unit"] == "users" else "inter_total"
            pct.append((w[col] / w[tot] * 100).mean())
    out["pct_new"] = pct
    out = out.drop(columns="pct_override", errors="ignore")
    out.insert(0, "dataset", dataset_key)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASET_CONFIGS), action="append",
                    help="repeatable; defaults to both time-cut datasets")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--update-every", type=int, default=UPDATE_EVERY)
    ap.add_argument("--out", type=Path, help="write the combined table as CSV")
    ap.add_argument("--latex", type=Path, help="write the combined table as LaTeX")
    args = ap.parse_args()

    frames = []
    for key in (args.dataset or DEFAULT_DATASETS):
        print(f"\n{key}")
        frames.append(new_user_share(key, args.batch_size, args.update_every))
    combined = pd.concat(frames, ignore_index=True)

    print()
    print(combined.to_string(index=False,
                             formatters={"n_new": "{:,.2f}".format,
                                         "n_total": "{:,.2f}".format,
                                         "pct_new": "{:.2f}%".format}))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out, index=False)
        print(f"\nCSV saved → {args.out}")
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        fmt = combined.copy()
        for c in ["n_new", "n_total"]:
            fmt[c] = fmt[c].map("{:,.2f}".format)
        fmt["pct_new"] = fmt["pct_new"].map("{:.2f}\\%".format)
        args.latex.write_text(fmt.to_latex(index=False))
        print(f"LaTeX table saved → {args.latex}")


if __name__ == "__main__":
    main()
