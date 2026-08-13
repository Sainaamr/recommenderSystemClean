"""
Content-aware cold-start experiment for new users.

Compares two initialization strategies for new users arriving during streaming:
  mean_init    : global mean of all trained user embeddings (current baseline)
  content_init : weighted average of geo+category similar trained users

LightGCN is never retrained.

Usage:
  python experiments/run_content_coldstart.py --dataset yelp
  python experiments/run_content_coldstart.py --dataset yelp --csv results/existing.csv

Note: only works meaningfully for yelp (requires yelp.item metadata).
"""

import os, sys, argparse, torch
from pathlib import Path
from datetime import datetime

from codecarbon import EmissionsTracker

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from src.evaluation.metrics import compute_metrics_at_ks, _avg
from src.models.incremental_lightgcn import IncrementalLightGCN
from src.models.content_user_init import ContentUserInitializer
from experiments.run_incremental_lightgcn import (
    DATASET_CONFIGS, RESULTS_DIR, BATCH_SIZE,
    load_id_mappings, build_user_history, train_historical,
)
from tools.plot_utils import plot_content_vs_no_update, _CONTENT_COLDSTART_METRICS

ITEM_META_PATH = "dataset/yelp/yelp.item"


# Per-batch scoring 

def score_batch(user_emb, item_emb, existing_gt, new_user_gt, history,
                content_init: ContentUserInitializer, new_user_first_items: dict,
                k: int = 10):
    """
    Scores existing users once, and new users once under content-aware
    init — built from their first interactions so far.

    Combine this script's output with a run_new_user_analysis.py

    new_user_first_items : uid → list of item identifiers seen so far
    (updated externally) — each entry is either an iid (int, a trained item)
    or a raw item token (str, an item excluded from training but still
    content-matchable); see ContentUserInitializer.get_embedding().

    Returns (m_existing, m_new_content, m_overall_content) — each an
    _avg()-averaged metrics dict (recall@k, precision@k, ndcg@k, ...), same
    shape batch_metrics_lgcn returns.
    """
    def score_uid(uid, gt, user_vec):
        #    translate to torch from numpy for content user init
        user_vec_t = torch.as_tensor(user_vec, dtype=item_emb.dtype, device=item_emb.device)
        scores = torch.matmul(user_vec_t, item_emb.T).cpu().numpy()
        return compute_metrics_at_ks(scores, gt, history.get(uid, set()), (k,))

    # scoring existing users
    existing_results = [
        score_uid(uid, gt, user_emb[uid])
        for uid, gt in existing_gt.items() if uid < user_emb.shape[0]
    ]

    # scoring new users
    new_content_results = []
    for uid, gt in new_user_gt.items():
        if uid >= user_emb.shape[0]:
            continue
        iids_so_far = new_user_first_items.get(uid, [])
        content_emb = content_init.get_embedding(iids_so_far)
        new_content_results.append(score_uid(uid, gt, content_emb))

    return (
        _avg(existing_results),
        _avg(new_content_results),
        _avg(existing_results + new_content_results),
    )


# Streaming loop
"""
cfg: dataset configuration
ckpt: trained lightgcn checkpoint
"""
def run_content_coldstart(cfg: dict, ckpt: str) -> pd.DataFrame:
    user2id, item2id, config, dataset = load_id_mappings(cfg)
    lgcn = IncrementalLightGCN.from_checkpoint(ckpt, config, dataset)
    history = build_user_history(user2id, item2id, cfg)

    n_users_trained = lgcn.n_users

    # Build content index from frozen embeddings
    print("\n── Building content-aware user index ────────────────────────────────")
    lgcn.eval()
    with torch.no_grad():
        user_emb, item_emb = lgcn.forward()

    geo_precision = 4
    ckpt_path = Path(cfg["checkpoint"])
    cache_path = ckpt_path.with_name(f"{ckpt_path.stem}-content_init-geo{geo_precision}.pkl")

    if cache_path.exists():
        print(f"Found existing content-index cache: {cache_path}")
        content_init = ContentUserInitializer.load(str(cache_path))
    else:
        content_init = ContentUserInitializer(alpha=0.7, top_k=20, geo_precision=geo_precision)
        content_init.build(
            item_meta_path        = ITEM_META_PATH,
            historical_inter_path = cfg["historical_path"],
            user2id               = user2id,
            item2id               = item2id,
            # One-time NumPy snapshot for the geo/category index
            user_emb              = user_emb.cpu().numpy(),
        )
        content_init.save(str(cache_path))

    # realtime_path
    df_rt = pd.read_csv(cfg["realtime_path"], sep="\t")

    id_cast  = cfg["id_cast"]
    next_uid = [lgcn.n_users]
    next_iid = [lgcn.n_items]

    # uid → list of iids seen so far, used to build each new user's content
    # embedding.
    new_user_first_items: dict = {}
    # to asignnew id for new users
    def get_uid(x):
        key = id_cast(x)
        if key not in user2id:
            new_uid = next_uid[0]
            user2id[key] = new_uid
            next_uid[0] += 1
            # check if this user has any prior history that model wasnt trained on
            hist_items = content_init.get_excluded_history(key)
            if hist_items:
                new_user_first_items[new_uid] = list(hist_items)
        return user2id[key]
    # to asign id for new items
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
    # interactions straddle a batch boundary only counts toward n_new_users
    # once
    seen_as_new = set()

    print(f"\n── Streaming {n_batches} batches (no retraining) ────────────────────")

    for i in range(n_batches):
        batch       = df_rt.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_users = batch["uid"].tolist()
        batch_items = batch["iid"].tolist()

        # Expand so new users get mean embeddings (baseline path)
        max_u = max(batch_users, default=0)
        max_i = max(batch_items, default=0)
        if max_u >= lgcn.n_users or max_i >= lgcn.n_items:
            lgcn.expand_embeddings(
                max(max_u + 1, lgcn.n_users),
                max(max_i + 1, lgcn.n_items),
            )
            # Refresh embeddings after expansion
            lgcn.eval()
            with torch.no_grad():
                user_emb, item_emb = lgcn.forward()

        # Split this batch's ground truth by existing vs new user
        existing_gt = {}
        new_user_gt = {}
        for uid, iid in zip(batch_users, batch_items):
            if uid >= n_users_trained:
                new_user_gt.setdefault(uid, set()).add(iid)
            else:
                existing_gt.setdefault(uid, set()).add(iid)

        m_existing, m_new_content, m_overall_content = score_batch(
            user_emb, item_emb, existing_gt, new_user_gt,
            history, content_init, new_user_first_items,
        )

        new_user_set = set(new_user_gt.keys())
        first_time_new = new_user_set - seen_as_new
        seen_as_new |= new_user_set
        n_new_users = len(first_time_new)
        pct_new_users = n_new_users / max(len(set(batch_users)), 1)

        # Accumulate this batch's items into each new user's content profile,
        # and update history
        for uid, iid in zip(batch_users, batch_items):
            if uid >= n_users_trained:
                new_user_first_items.setdefault(uid, [])
                if iid not in new_user_first_items[uid]:
                    new_user_first_items[uid].append(iid)
            history.setdefault(uid, set()).add(iid)

        records.append({
            "batch":                     i + 1,
            "interactions":              (i + 1) * BATCH_SIZE,
            "recall_existing":           m_existing["recall@10"],
            "precision_existing":        m_existing["precision@10"],
            "ndcg_existing":             m_existing["ndcg@10"],
            "hr_existing":               m_existing["hr@10"],
            "mrr_existing":              m_existing["mrr"],
            "recall_new_content":        m_new_content["recall@10"],
            "precision_new_content":     m_new_content["precision@10"],
            "ndcg_new_content":          m_new_content["ndcg@10"],
            "hr_new_content":            m_new_content["hr@10"],
            "mrr_new_content":           m_new_content["mrr"],
            "recall_overall_content":    m_overall_content["recall@10"],
            "precision_overall_content": m_overall_content["precision@10"],
            "ndcg_overall_content":      m_overall_content["ndcg@10"],
            "hr_overall_content":        m_overall_content["hr@10"],
            "mrr_overall_content":       m_overall_content["mrr"],
            "n_new_users":               n_new_users,
            "pct_new_users":             pct_new_users,
            "n_users_trained":           n_users_trained,
        })

        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1:>3}/{n_batches}  "
                  f"new_content={m_new_content['recall@10']:.4f}  "
                  f"n_new={n_new_users}")

    return pd.DataFrame(records)


def merge_new_user_baseline(df: pd.DataFrame, new_user_csv: Path) -> pd.DataFrame:
    """
    Merge in the mean-init baseline
    """
    baseline = pd.read_csv(new_user_csv)[[
        "batch",
        "recall_new_user", "precision_new_user", "ndcg_new_user",
        "recall_overall", "precision_overall", "ndcg_overall",
    ]]
    baseline = baseline.rename(columns={
        "recall_new_user":    "recall_new_mean",
        "precision_new_user": "precision_new_mean",
        "ndcg_new_user":      "ndcg_new_mean",
        "recall_overall":     "recall_overall_mean",
        "precision_overall":  "precision_overall_mean",
        "ndcg_overall":       "ndcg_overall_mean",
    })
    return df.merge(baseline, on="batch", how="left")


def merge_no_update_overall(df: pd.DataFrame, no_update_csv: Path) -> pd.DataFrame:
    """
    Merge in the overall recall/precision/ndcg from a
    run_incremental_lightgcn.py no_update-strategy results CSV, joined on
    batch. That script scores everyone together (no existing/new-user
    split), so this only supplies an aggregate "no update" baseline — use
    merge_new_user_baseline instead for the existing/new/overall breakdown.
    """
    baseline = pd.read_csv(no_update_csv)[[
        "batch", "recall_at_10", "precision_at_10", "ndcg_at_10",
    ]]
    baseline = baseline.rename(columns={
        "recall_at_10":    "recall_no_update",
        "precision_at_10": "precision_no_update",
        "ndcg_at_10":      "ndcg_no_update",
    })
    return df.merge(baseline, on="batch", how="left")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["yelp"], default="yelp",
                        help="Only yelp supported (requires yelp.item metadata)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Existing results CSV — skip streaming, just re-plot")
    parser.add_argument("--new-user-csv", type=Path, default=None,
                        help="A run_new_user_analysis.py results CSV, for the same "
                             "dataset/checkpoint — merged in on 'batch' to provide the "
                             "mean-init baseline for comparison, without recomputing it")
    parser.add_argument("--no-update-csv", type=Path, default=None,
                        help="A run_incremental_lightgcn.py no_update-strategy results "
                             "CSV — merged in on 'batch' to provide the overall no-update "
                             "baseline for comparison")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg = DATASET_CONFIGS[args.dataset].copy()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"yelp_content_coldstart_{ts}.csv"
    out_png = results_dir / f"yelp_content_coldstart_{ts}.png"

    if args.csv:
        df = pd.read_csv(args.csv)
        out_png = Path(str(args.csv).replace(".csv", "_replot.png"))
        print(f"Loaded existing results from {args.csv}")

        energy_path = Path(str(args.csv).replace(".csv", "_energy.csv"))
        if energy_path.exists():
            emissions = pd.read_csv(energy_path).iloc[0]
            training_emissions_mg  = emissions.get("training_emissions_mg", 0.0)
            streaming_emissions_mg = emissions.get("streaming_emissions_mg", 0.0)
        else:
            print("  (no emissions sidecar found for this CSV — energy unknown)")
            training_emissions_mg = streaming_emissions_mg = 0.0
    else:
        print("\n── Loading LightGCN checkpoint ──────────────────────────────────────")
        ckpt, training_emissions_mg = train_historical(cfg)

        print("\n── Running content cold-start experiment ────────────────────────────")
        # Measures the cost of the whole cold-start run (content-index
        # build/load + per-batch scoring over all streaming batches)
        tracker = EmissionsTracker(
            project_name=f"content_coldstart_{args.dataset}",
            output_dir=str(results_dir),
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        df = run_content_coldstart(cfg, ckpt)
        # tracker.stop() returns CO2 emissions in kg, not energy; *1e6
        kg_co2 = tracker.stop() or 0.0
        streaming_emissions_mg = kg_co2 * 1e6
        print(f"  Content cold-start streaming emissions: {streaming_emissions_mg:.4f} mg CO2eq")

        df.to_csv(out_csv, index=False)
        print(f"\nResults saved → {out_csv}")

        energy_path = Path(str(out_csv).replace(".csv", "_energy.csv"))
        pd.DataFrame([{
            "training_emissions_mg":  training_emissions_mg,
            "streaming_emissions_mg": streaming_emissions_mg,
        }]).to_csv(energy_path, index=False)
        print(f"Emissions saved → {energy_path}")

    plot_df = df
    if args.new_user_csv:
        plot_df = merge_new_user_baseline(plot_df, args.new_user_csv)
        print(f"Merged mean-init baseline from {args.new_user_csv}")
    if args.no_update_csv:
        plot_df = merge_no_update_overall(plot_df, args.no_update_csv)
        print(f"Merged no-update overall baseline from {args.no_update_csv}")


    if args.csv:
        plot_content_vs_no_update(plot_df, out_png,
                                  "Content-Aware Cold Start — yelp (no retraining)",
                                  subtitle="Yelp Dataset")

    print(f"\n── Summary ──────────────────────────────────────────────────────────")
    new_mask = plot_df["n_new_users"] > 0
    any_mean_baseline = False
    for metric, label, has_baseline_col in _CONTENT_COLDSTART_METRICS:
        new_content = plot_df.loc[new_mask, f"{metric}_new_content"].mean()
        overall_content = plot_df[f"{metric}_overall_content"].mean()
        print(f"  Avg {label:<11} new users (content init): {new_content:.4f}")
        print(f"  Avg {label:<11} overall   (content init): {overall_content:.4f}")
        mean_col = f"{metric}_new_mean"
        if has_baseline_col and mean_col in plot_df.columns:
            any_mean_baseline = True
            new_mean = plot_df.loc[new_mask, mean_col].mean()
            overall_mean = plot_df[f"{metric}_overall_mean"].mean()
            print(f"  Avg {label:<11} new users (mean init):    {new_mean:.4f}")
            print(f"  Avg {label:<11} overall   (mean init):    {overall_mean:.4f}")
            print(f"  {label:<11} gain on new users:            {new_content - new_mean:+.4f}")
    if not any_mean_baseline:
        print("  (pass --new-user-csv <run_new_user_analysis.py CSV> for a mean-init comparison)")

    print(f"  Historical training emissions:       {training_emissions_mg:.4f} mg CO2eq")
    print(f"  Content cold-start emissions:        {streaming_emissions_mg:.4f} mg CO2eq")
    print(f"  Total emissions:                     {training_emissions_mg + streaming_emissions_mg:.4f} mg CO2eq")


if __name__ == "__main__":
    main()
