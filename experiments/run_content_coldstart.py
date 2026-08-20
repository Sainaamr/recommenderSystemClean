"""
Content-aware cold-start experiment for new users.

New users get a content-based embedding (ContentUserInitializer) written
once into a persistent row in the embedding table — the same mechanism
run_content_incremental.py uses, minus the periodic gradient update. Once
seeded, a user's row is never touched again for the rest of the run.
Existing and new users are scored identically, by reading directly from
that table.

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
    load_id_mappings, build_user_history, train_historical, task_mg,
)
from tools.plot_utils import plot_content_vs_no_update, _CONTENT_COLDSTART_METRICS

ITEM_META_PATH = "dataset/yelp/yelp.item"


# Per-batch scoring

def score_batch(user_emb, item_emb, existing_gt, new_user_gt, history, k: int = 10):
    """
    Scores existing AND new users the same way — reading directly from the
    embedding table row, matching run_content_incremental.py's score_batch.
    No inline recomputation here; whatever was last written to a user's row
    (mean-init or content-seeded) is what gets scored.

    Returns (m_existing, m_new_content, m_overall_content) — each an
    _avg()-averaged metrics dict (recall@k, precision@k, ndcg@k, ...).
    """
    def score_uid(uid, gt):
        scores = torch.matmul(user_emb[uid], item_emb.T).cpu().numpy()
        return compute_metrics_at_ks(scores, gt, history.get(uid, set()), (k,))

    existing_results = [score_uid(uid, gt) for uid, gt in existing_gt.items() if uid < user_emb.shape[0]]
    new_results      = [score_uid(uid, gt) for uid, gt in new_user_gt.items() if uid < user_emb.shape[0]]
    return (
        _avg(existing_results),
        _avg(new_results),
        _avg(existing_results + new_results),
    )


def _apply_content_seeds_once(lgcn: IncrementalLightGCN, content_init: ContentUserInitializer,
                              accumulated_items: dict, content_seeded: set) -> int:
    """
    One-time content seed: for every still-unseeded uid with accumulated
    history, write a content-based embedding into their row exactly once.
    No gradient training exists in this script, so once seeded, a row is
    never touched again for the rest of the run.
    """
    seed = {}
    for uid, items in accumulated_items.items():
        if uid in content_seeded:
            continue
        if not items:
            continue
        seed[uid] = content_init.get_embedding(items)
        content_seeded.add(uid)
    if seed:
        lgcn.set_user_embeddings(seed)
    return len(seed)


# Streaming loop
"""
cfg: dataset configuration
ckpt: trained lightgcn checkpoint
"""
def run_content_coldstart(cfg: dict, ckpt: str) -> tuple[pd.DataFrame, float, float, dict]:
    tracker = EmissionsTracker(
        project_name="content_coldstart",
        output_dir=str(RESULTS_DIR),
        log_level="error",
        save_to_file=False,
    )
    tracker.start()

    tracker.start_task("load_id_mappings")
    user2id, item2id, config, dataset = load_id_mappings(cfg)
    id_mapping_emissions_mg = task_mg(tracker.stop_task())

    tracker.start_task("load_checkpoint")
    lgcn = IncrementalLightGCN.from_checkpoint(ckpt, config, dataset)
    checkpoint_load_emissions_mg = task_mg(tracker.stop_task())

    tracker.start_task("build_user_history")
    history = build_user_history(user2id, item2id, cfg)
    build_user_history_emissions_mg = task_mg(tracker.stop_task())

    n_users_trained = lgcn.n_users

    # Build content index from frozen embeddings
    print("\n── Building content-aware user index ────────────────────────────────")
    tracker.start_task("initial_embedding_snapshot")
    lgcn.eval()
    with torch.no_grad():
        user_emb, item_emb = lgcn.forward()
    embedding_snapshot_emissions_mg = task_mg(tracker.stop_task())

    section_emissions_mg = {
        "id_mapping_emissions_mg":            id_mapping_emissions_mg,
        "checkpoint_load_emissions_mg":       checkpoint_load_emissions_mg,
        "build_user_history_emissions_mg":    build_user_history_emissions_mg,
        "embedding_snapshot_emissions_mg":    embedding_snapshot_emissions_mg,
    }
    streaming_emissions_mg = sum(section_emissions_mg.values())

    geo_precision = 4
    ckpt_path = Path(cfg["checkpoint"])
    cache_path = ckpt_path.with_name(f"{ckpt_path.stem}-content_init-geo{geo_precision}.pkl")

    if cache_path.exists():
        print(f"Found existing content-index cache: {cache_path}")
        content_init = ContentUserInitializer.load(str(cache_path))
        # No build happened — nothing to measure.
        content_build_emissions_mg = 0.0
    else:
        tracker.start_task("content_build")
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
        content_build_emissions_mg = task_mg(tracker.stop_task())
        streaming_emissions_mg += content_build_emissions_mg
        print(f"  Content index build emissions: {content_build_emissions_mg:.4f} mg CO2eq")

    id_cast = cfg["id_cast"]
    next_uid = [lgcn.n_users]
    next_iid = [lgcn.n_items]
    df_rt = pd.read_csv(cfg["realtime_path"], sep="\t")
    n_batches = len(df_rt) // BATCH_SIZE

    # uid -> accumulated item list, in ContentUserInitializer.get_embedding()'s
    # native mixed format (int iid for trained items, str token for items
    # that were themselves excluded from training)
    accumulated_items: dict = {}
    content_seeded: set = set()   # uids that have already received their one-time content seed
    seen_as_new: set = set()      # every uid ever classified as "new" so far
    records = []

    print(f"\n── Streaming {n_batches} batches (content-init, no retraining) ───────")

    for i in range(n_batches):

        # Batch body split into named, back-to-back sub-tasks (no gaps
        # between stop_task() and the next start_task()), so each phase's
        # own cost is saved separately in the CSV, and batch_emissions_mg
        # (their sum) is still the accurate whole-batch total.
        tracker.start_task(f"batch_{i}_id_resolution")

        batch = df_rt.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].copy()

        newly_arrived = []  # (uid, raw_token)

        def get_uid(x):
            key = id_cast(x)
            if key not in user2id:
                uid = next_uid[0]
                user2id[key] = uid
                next_uid[0] += 1
                newly_arrived.append((uid, key))
            return user2id[key]

        def get_iid(x):
            key = id_cast(x)
            if key not in item2id:
                item2id[key] = next_iid[0]
                next_iid[0] += 1
            return item2id[key]

        batch["uid"] = batch["user_id:token"].apply(get_uid).astype(int)
        batch["iid"] = batch["item_id:token"].apply(get_iid).astype(int)
        batch_users = batch["uid"].tolist()
        batch_items = batch["iid"].tolist()

        id_resolution_emissions_mg = task_mg(tracker.stop_task())

        tracker.start_task(f"batch_{i}_recovered_history_seed")

        content_seed_this_batch = {}
        promoted_iids_this_batch = []  # excluded-history items promoted to real iids

        for uid, u_tok in newly_arrived:
            hist_items = content_init.get_excluded_history(u_tok)
            if not hist_items:
                continue  # no recovered history -> mean-init fallback

            promoted = []
            for item_key in hist_items:
                if isinstance(item_key, str):
                    if item_key not in item2id:
                        item2id[item_key] = next_iid[0]
                        next_iid[0] += 1
                    promoted.append(item2id[item_key])
                else:
                    promoted.append(item_key)
            promoted_iids_this_batch.extend(promoted)

            content_seed_this_batch[uid] = content_init.get_embedding(hist_items)
            accumulated_items[uid] = list(hist_items)
            content_seeded.add(uid)

        recovered_history_seed_emissions_mg = task_mg(tracker.stop_task())

        tracker.start_task(f"batch_{i}_expand_embeddings")

        max_u = max(batch_users, default=0)
        max_i = max(batch_items + promoted_iids_this_batch, default=0)
        if max_u >= lgcn.n_users or max_i >= lgcn.n_items:
            lgcn.expand_embeddings_content(
                max(max_u + 1, lgcn.n_users),
                max(max_i + 1, lgcn.n_items),
                content_seed_this_batch,
            )
            lgcn.eval()
            with torch.no_grad():
                user_emb, item_emb = lgcn.forward()

        expand_embeddings_emissions_mg = task_mg(tracker.stop_task())

        tracker.start_task(f"batch_{i}_gt_split")

        # Split this batch's ground truth by existing vs new user
        existing_gt = {}
        new_user_gt = {}
        for uid, iid in zip(batch_users, batch_items):
            if uid >= n_users_trained:
                new_user_gt.setdefault(uid, set()).add(iid)
            else:
                existing_gt.setdefault(uid, set()).add(iid)

        gt_split_emissions_mg = task_mg(tracker.stop_task())

        tracker.start_task(f"batch_{i}_scoring")

        m_existing, m_new_content, m_overall_content = score_batch(
            user_emb, item_emb, existing_gt, new_user_gt, history)

        scoring_emissions_mg = task_mg(tracker.stop_task())

        tracker.start_task(f"batch_{i}_history_update")

        new_user_set = set(new_user_gt.keys())
        n_new_users = len(new_user_set - seen_as_new)
        seen_as_new |= new_user_set
        pct_new_users = n_new_users / max(len(set(batch_users)), 1)

        # After scoring only: update history, and grow accumulated_items for
        # new users (their content embedding, if not yet seeded, gets built
        # from this once they're picked up by _apply_content_seeds_once below).
        for uid, iid in zip(batch_users, batch_items):
            history.setdefault(uid, set()).add(iid)
            if uid >= n_users_trained:
                accumulated_items.setdefault(uid, [])
                if iid not in accumulated_items[uid]:
                    accumulated_items[uid].append(iid)

        history_update_emissions_mg = task_mg(tracker.stop_task())

        tracker.start_task(f"batch_{i}_apply_content_seeds")

        _apply_content_seeds_once(lgcn, content_init, accumulated_items, content_seeded)

        apply_content_seeds_emissions_mg = task_mg(tracker.stop_task())

        batch_emissions_mg = (
            id_resolution_emissions_mg
            + recovered_history_seed_emissions_mg
            + expand_embeddings_emissions_mg
            + gt_split_emissions_mg
            + scoring_emissions_mg
            + history_update_emissions_mg
            + apply_content_seeds_emissions_mg
        )
        streaming_emissions_mg += batch_emissions_mg

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
            "id_resolution_emissions_mg":          id_resolution_emissions_mg,
            "recovered_history_seed_emissions_mg": recovered_history_seed_emissions_mg,
            "expand_embeddings_emissions_mg":      expand_embeddings_emissions_mg,
            "gt_split_emissions_mg":               gt_split_emissions_mg,
            "scoring_emissions_mg":                scoring_emissions_mg,
            "history_update_emissions_mg":         history_update_emissions_mg,
            "apply_content_seeds_emissions_mg":    apply_content_seeds_emissions_mg,
            "batch_emissions_mg":        batch_emissions_mg,
        })

        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1:>3}/{n_batches}  "
                  f"new_content={m_new_content['recall@10']:.4f}  "
                  f"n_new={n_new_users}")

    tracker.stop()
    return pd.DataFrame(records), content_build_emissions_mg, streaming_emissions_mg, section_emissions_mg


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
            training_emissions_mg      = emissions.get("training_emissions_mg", 0.0)
            streaming_emissions_mg     = emissions.get("streaming_emissions_mg", 0.0)
            content_build_emissions_mg = emissions.get("content_build_emissions_mg", 0.0)
        else:
            print("  (no emissions sidecar found for this CSV — energy unknown)")
            training_emissions_mg = streaming_emissions_mg = content_build_emissions_mg = 0.0
        total_batch_emissions = df["batch_emissions_mg"].sum() if "batch_emissions_mg" in df.columns else 0.0
    else:
        print("\n── Loading LightGCN checkpoint ──────────────────────────────────────")
        ckpt, training_emissions_mg = train_historical(cfg)

        print("\n── Running content cold-start experiment ────────────────────────────")

        df, content_build_emissions_mg, streaming_emissions_mg, section_emissions_mg = run_content_coldstart(cfg, ckpt)
        print(f"  Content cold-start streaming emissions: {streaming_emissions_mg:.4f} mg CO2eq")

        df.to_csv(out_csv, index=False)
        print(f"\nResults saved → {out_csv}")

        total_batch_emissions = df["batch_emissions_mg"].sum()

        energy_path = Path(str(out_csv).replace(".csv", "_energy.csv"))
        pd.DataFrame([{
            "training_emissions_mg":      training_emissions_mg,
            **section_emissions_mg,
            "content_build_emissions_mg": content_build_emissions_mg,
            "total_batch_emissions_mg":   total_batch_emissions,
            "streaming_emissions_mg":     streaming_emissions_mg,
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

    print("\n── Summary ──────────────────────────────────────────────────────────")
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
    print(f"  Content index build emissions:       {content_build_emissions_mg:.4f} mg CO2eq  (one-time)")
    print(f"  Total batch emissions (scoring+seed):{total_batch_emissions:.4f} mg CO2eq  (every batch)")
    print(f"  Content cold-start emissions:        {streaming_emissions_mg:.4f} mg CO2eq")
    print(f"  Total emissions:                     {training_emissions_mg + streaming_emissions_mg:.4f} mg CO2eq")


if __name__ == "__main__":
    main()
