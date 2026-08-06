"""
Combined content-aware cold-start + incremental update experiment.

New users get a content-based cold-start embedding (ContentUserInitializer)
instead of a plain mean embedding, AND the model itself keeps periodically
fine-tuning via real BPR gradient steps (IncrementalLightGCN.incremental_update),


Recovered historical interactions for users excluded from training (real data
that exists in the historical file but was filtered out by RecBole's
interaction-count threshold) are used two ways:
  1. to compute that user's initial content-based embedding immediately, and
  2. fed as real graph edges into the next incremental_update, so this
     otherwise-discarded signal actually gets learned via gradient descent,
     not just used as a passive similarity heuristic.

Usage:
  python experiments/run_content_incremental.py --dataset yelp
  python experiments/run_content_incremental.py --dataset yelp --csv results/existing.csv

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

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics_at_ks, _avg
from src.models.incremental_lightgcn import IncrementalLightGCN
from src.models.content_user_init import ContentUserInitializer
from experiments.run_incremental_lightgcn import (
    DATASET_CONFIGS, RESULTS_DIR, BATCH_SIZE, UPDATE_EVERY, UPDATE_EPOCHS,
    load_id_mappings, build_user_history, train_historical,
)
from experiments.run_content_coldstart import ITEM_META_PATH, merge_new_user_baseline
from tools.plot_utils import plot_content_coldstart, _CONTENT_COLDSTART_METRICS


# Per-batch scoring
"""
user_emb: User embeddings table
item_emb: Item embeddings table
existing_gt: list of items user interacted with in this batch loop  
uid -> set of item ids that user actually interacted with in this batch for existing user
new_user_gt: list of items user interacted with in this batch loop  
uid -> set of item ids that user actually interacted with in this batch for new user
history: dict build up over the experimnet. it gives what items user has interacted with so far
"""
def score_batch(user_emb, item_emb, existing_gt, new_user_gt, history, k: int = 10):
    """
    Scores existing AND new users
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

"""
lgcn: model
content init: content index object
accumulated_items: dict a  of items user interacted with data that content embeddings computed from
uid -> iids and str
content_seeded: a set of uid of user already got content embeddings
gradient_touched: a set of uid of uid already had incremental update on 
"""
def _apply_content_seeds_once(lgcn: IncrementalLightGCN, content_init: ContentUserInitializer,
                               accumulated_items: dict, content_seeded: set, gradient_touched: set) -> int:
    """
    Content-init is a one-time INITIALIZATION,
    For every still-new uid that has never yet been content-seeded, seed it
    exactly once from whatever's accumulated in accumulated_items so far
    their later interactions simply accumulate into the update buffer for incremental_update
    """
    seed = {}
    # Loop over the users tracked so far.
    for uid, items in accumulated_items.items():
        # skip if embedding is result of gradient or the content based initialization already took place
        if uid in gradient_touched or uid in content_seeded:
            continue
        if not items:
            continue
        # for new users get content based embeddings  and mark as seeded so  it won't happen again
        seed[uid] = content_init.get_embedding(items)
        content_seeded.add(uid)
    if seed:
        # if there are some new content based embeddings updated those rows
        lgcn.set_user_embeddings(seed)
    return len(seed)


# Streaming loop

def run_content_incremental(cfg: dict, ckpt: str) -> tuple[pd.DataFrame, float]:
    """
    1. Load the trained model and mappings
    2. Build history: used later to mask already-seen items out of every ranking
    3. Run one forward pass to get the current user_emb/item_emb tensors
    4. Build or load the content index
    5. Initialize the tracking containers: accumulated_items (per-user interaction history for content-seeding),
        content_seeded/gradient_touched (one-time-seed and one-time-gradient tracking sets),
        seen_as_new (dedupe new-user counting), buffer_users/buffer_items (queued interactions for the next update),
        records (the output accumulator).
    per batch
    6. Resolve this batch's uids/iids
    7. Handle newly-arrived users with recovered history
    8. grow the model's tables if needed
    9.  Split ground truth
    10. Score the batch
    11. Update new-user counting
    12. Post-scoring accumulation update history,
        grow accumulated_items for new users, and append every interaction into the buffer.
    13. Apply the one-time content seed for anyone now eligible
    14. incremental update
    """
    user2id, item2id, config, dataset = load_id_mappings(cfg)
    lgcn = IncrementalLightGCN.from_checkpoint(ckpt, config, dataset)
    history = build_user_history(user2id, item2id, cfg)
    n_users_trained = lgcn.n_users

    lgcn.eval()
    with torch.no_grad():
        user_emb, item_emb = lgcn.forward()

    print("\n── Building content-aware user index ────────────────────────────────")
    geo_precision = 4
    ckpt_path = Path(cfg["checkpoint"])
    cache_path = ckpt_path.with_name(f"{ckpt_path.stem}-content_init-geo{geo_precision}.pkl")
    if cache_path.exists():
        print(f"Found existing content-index cache: {cache_path}")
        content_init = ContentUserInitializer.load(str(cache_path))
        # No build happened — nothing to measure.
        content_build_emissions_mg = 0.0
    else:
        build_tracker = EmissionsTracker(
            project_name="content_index_build",
            output_dir=str(RESULTS_DIR),
            log_level="error",
            save_to_file=False,
        )
        build_tracker.start()
        content_init = ContentUserInitializer(alpha=0.7, top_k=20, geo_precision=geo_precision)
        content_init.build(
            item_meta_path        = ITEM_META_PATH, # path to yelp.item
            historical_inter_path = cfg["historical_path"], # historical interaction file
            user2id               = user2id, # token to row
            item2id               = item2id,
            user_emb              = user_emb.cpu().numpy(), # frozen embedding from the checkpoint
        )
        content_init.save(str(cache_path))
        kg_co2_build = build_tracker.stop() or 0.0
        content_build_emissions_mg = kg_co2_build * 1e6
        print(f"  Content index build emissions: {content_build_emissions_mg:.4f} mg CO2eq")

    id_cast = cfg["id_cast"]
    next_uid = [lgcn.n_users]
    next_iid = [lgcn.n_items]
    df_rt = pd.read_csv(cfg["realtime_path"], sep="\t")
    n_batches = len(df_rt) // BATCH_SIZE
    update_every = cfg.get("update_every", UPDATE_EVERY)

    # uid -> accumulated item list, in ContentUserInitializer.get_embedding()'s
    # native mixed format (int iid for trained items, str token for items
    # that were themselves excluded from training)
    accumulated_items: dict = {} # uid -> list of items that user has interacted with so far (
    content_seeded: set = set()     # uids that have already received their one-time content seed
    gradient_touched: set = set()   # uids that have been through >=1 incremental_update
    seen_as_new: set = set() # every uid that has ever been classified as "new" a
    buffer_users: list = []
    buffer_items: list = []
    # buffer_users[k] interacted with buffer_items[k] since last incremental update
    records = []

    print(f"\n── Streaming {n_batches} batches "
          f"(content-init + incremental update every {update_every} batches) ──")

    for i in range(n_batches):
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

        content_seed_this_batch = {}
        promoted_iids_this_batch = [] # collect ids that get generated for recovered item
                                      # from excluded history


        recovered_seed_tracker = EmissionsTracker(
            project_name=f"content_seed_recovered_batch_{i}",
            output_dir=str(RESULTS_DIR),
            log_level="error",
            save_to_file=False,
        )
        recovered_seed_tracker.start()
        for uid, u_tok in newly_arrived:
            hist_items = content_init.get_excluded_history(u_tok) # user recovered history
            if not hist_items:
                continue  # no recovered history -> mean-init fallback


            promoted = [] #  include normal trained item id and ids of new assignees. hist_items  includes new assignees as str
            for item_key in hist_items:
                if isinstance(item_key, str):
                    if item_key not in item2id:
                        item2id[item_key] = next_iid[0]
                        next_iid[0] += 1
                    promoted.append(item2id[item_key])
                else:
                    promoted.append(item_key)
            promoted_iids_this_batch.extend(promoted)

            # These parallel lists are used later for add interactions
            buffer_users.extend([uid] * len(promoted))
            buffer_items.extend(promoted)

            content_seed_this_batch[uid] = content_init.get_embedding(hist_items) # Content seed computed from the recovered history itself
            accumulated_items[uid] = list(hist_items)
            content_seeded.add(uid) # mark as seeded so it's not recomputed
        kg_co2_recovered_seed = recovered_seed_tracker.stop() or 0.0


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

        # Split ground truth
        existing_gt, new_user_gt = {}, {}
        for uid, iid in zip(batch_users, batch_items):
            (new_user_gt if uid >= n_users_trained else existing_gt).setdefault(uid, set()).add(iid)

        # SCORE
        m_existing, m_new_content, m_overall_content = score_batch(
            user_emb, item_emb, existing_gt, new_user_gt, history)
        # filtering out the preciously new users from previous batches form this batch current user
        new_user_set = set(new_user_gt.keys())
        n_new_users = len(new_user_set - seen_as_new)
        seen_as_new |= new_user_set # now add then to seen as new for future batches filtering
        pct_new_users = n_new_users / max(len(set(batch_users)), 1) # what fraction of this batch is new users

        # after scoring update history and accumulated items
        for uid, iid in zip(batch_users, batch_items):
            history.setdefault(uid, set()).add(iid)
            if uid >= n_users_trained:
                accumulated_items.setdefault(uid, [])
                if iid not in accumulated_items[uid]:
                    accumulated_items[uid].append(iid)
            buffer_users.append(uid)
            buffer_items.append(iid)


        seed_tracker = EmissionsTracker(
            project_name=f"content_seed_batch_{i}",
            output_dir=str(RESULTS_DIR),
            log_level="error",
            save_to_file=False,
        )
        seed_tracker.start()
        _apply_content_seeds_once(lgcn, content_init, accumulated_items, content_seeded, gradient_touched)
        kg_co2_seed = seed_tracker.stop() or 0.0
        # Sum both content-seeding paths (recovered-history-at-creation +
        # first-interaction-fallback) into one figure for this batch.
        content_seed_emissions_mg = (kg_co2_recovered_seed + kg_co2_seed) * 1e6

        # ---- Periodic incremental update ----
        update_emissions_mg = 0.0
        updated = False
        if (i + 1) % update_every == 0:
            new_users_arr = np.array(buffer_users)
            new_items_arr = np.array(buffer_items)

            tracker = EmissionsTracker(
                project_name=f"content_incremental_update_batch_{i}",
                output_dir=str(RESULTS_DIR),
                log_level="error",
                save_to_file=False,
            )
            tracker.start()
            lgcn.add_interactions(new_users_arr, new_items_arr)
            lgcn.incremental_update(new_users_arr, new_items_arr, n_epochs=UPDATE_EPOCHS)
            # tracker.stop() returns CO2 emissions in kg, not energy; *1e6
            # converts kg -> mg, matching the mg CO2eq convention used elsewhere.
            kg_co2 = tracker.stop() or 0.0
            update_emissions_mg = kg_co2 * 1e6
            updated = True

            gradient_touched.update(buffer_users)
            buffer_users, buffer_items = [], []

            lgcn.eval()
            with torch.no_grad():
                user_emb, item_emb = lgcn.forward()

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
            "content_seed_emissions_mg": content_seed_emissions_mg,
            "update_emissions_mg":       update_emissions_mg,
            "updated":                   updated,
        })

        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1:>3}/{n_batches}  "
                  f"new_content={m_new_content['recall@10']:.4f}  "
                  f"n_new={n_new_users}  updated={updated}")

    return pd.DataFrame(records), content_build_emissions_mg


# Main

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
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg = DATASET_CONFIGS[args.dataset].copy()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"yelp_content_incremental_{ts}.csv"
    out_png = results_dir / f"yelp_content_incremental_{ts}.png"

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
    else:
        print("\n── Loading LightGCN checkpoint ──────────────────────────────────────")
        ckpt, training_emissions_mg = train_historical(cfg)

        print("\n── Running content-init + incremental update experiment ─────────────")
        # Aggregate figure for the whole run (content-index build/load + all
        # batches' scoring/refresh/updates). The per-batch update_emissions_mg
        # column already isolates just the incremental_update cost, matching
        # run_incremental_lightgcn.py's convention exactly for direct
        # comparability against the pure "incremental" strategy's results.
        tracker = EmissionsTracker(
            project_name=f"content_incremental_{args.dataset}",
            output_dir=str(results_dir),
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        df, content_build_emissions_mg = run_content_incremental(cfg, ckpt)
        kg_co2 = tracker.stop() or 0.0
        streaming_emissions_mg = kg_co2 * 1e6
        print(f"  Content-incremental streaming emissions: {streaming_emissions_mg:.4f} mg CO2eq")

        df.to_csv(out_csv, index=False)
        print(f"\nResults saved → {out_csv}")

        energy_path = Path(str(out_csv).replace(".csv", "_energy.csv"))
        pd.DataFrame([{
            "training_emissions_mg":      training_emissions_mg,
            "streaming_emissions_mg":     streaming_emissions_mg,
            "content_build_emissions_mg": content_build_emissions_mg,
        }]).to_csv(energy_path, index=False)
        print(f"Emissions saved → {energy_path}")

    plot_df = df
    if args.new_user_csv:
        plot_df = merge_new_user_baseline(df, args.new_user_csv)
        print(f"Merged mean-init baseline from {args.new_user_csv}")

    plot_content_coldstart(plot_df, out_png,
                           "Content-Aware Init + Incremental Update — yelp",
                           batch_size=BATCH_SIZE)

    print(f"\n── Summary ──────────────────────────────────────────────────────────")
    new_mask = plot_df["n_new_users"] > 0
    any_mean_baseline = False
    for metric, label, has_baseline_col in _CONTENT_COLDSTART_METRICS:
        new_content = plot_df.loc[new_mask, f"{metric}_new_content"].mean()
        overall_content = plot_df[f"{metric}_overall_content"].mean()
        print(f"  Avg {label:<11} new users (content+incr): {new_content:.4f}")
        print(f"  Avg {label:<11} overall   (content+incr): {overall_content:.4f}")
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

    n_updates = int(plot_df["updated"].sum()) if "updated" in plot_df.columns else 0
    total_update_emissions = plot_df["update_emissions_mg"].sum() if "update_emissions_mg" in plot_df.columns else 0.0
    total_seed_emissions = plot_df["content_seed_emissions_mg"].sum() if "content_seed_emissions_mg" in plot_df.columns else 0.0
    print(f"  Historical training emissions:       {training_emissions_mg:.4f} mg CO2eq")
    print(f"  Content index build emissions:       {content_build_emissions_mg:.4f} mg CO2eq  (one-time)")
    print(f"  Streaming (whole run) emissions:     {streaming_emissions_mg:.4f} mg CO2eq")
    print(f"  Total content-seeding emissions:     {total_seed_emissions:.4f} mg CO2eq  (every batch)")
    print(f"  Total incremental-update emissions:  {total_update_emissions:.4f} mg CO2eq  ({n_updates} updates)")
    print(f"  Total emissions:                     {training_emissions_mg + streaming_emissions_mg:.4f} mg CO2eq")


if __name__ == "__main__":
    main()
