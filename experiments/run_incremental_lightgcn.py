
"""
Incremental LightGCN streaming experiment.

Supports both ml-1m and yelp datasets via --dataset argument.

Step 1: Train LightGCN on historical split (skip if checkpoint exists)
Step 2: Stream realtime split in batches of 1000 interactions
Step 3: Every 20 batches, run incremental_update() on accumulated interactions
Step 4: Compare two strategies:
          - no_update:    model never updates → quality degrades over time
          - incremental:  model updates every 20 batches → quality stays stable
Step 5: Save Recall@10 and energy per batch to results/{dataset}_hybrid_results.csv
Step 6: Plot quality degradation vs recovery curve

Usage:
  python experiments/run_incremental_lightgcn.py --dataset ml-1m
  python experiments/run_incremental_lightgcn.py --dataset yelp

Outputs (timestamped to avoid overwriting previous runs):
  results/ml1m_hybrid_results_YYYYMMDD_HHMMSS.csv
  results/ml1m_hybrid_curves_YYYYMMDD_HHMMSS.png
"""

import os, sys, glob, argparse, json
from pathlib import Path
from datetime import datetime

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset

from src.models.incremental_lightgcn import IncrementalLightGCN
from src.evaluation.metrics import batch_metrics_lgcn
from tools.plot_utils import plot_streaming_results

# ── Parameters ───────────────────────────────────────────────────────────────
BATCH_SIZE    = 1000
UPDATE_EVERY  = 20    # run incremental_update every N batches
UPDATE_EPOCHS = 30    # fixed gradient steps per incremental update, no early stopping
RESULTS_DIR   = Path("results")
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset configs ───────────────────────────────────────────────────────────
# Each dataset has different paths, config files, ID types, and output files.
# ml-1m uses numeric IDs (cast with int), Yelp uses alphanumeric (keep as str).
DATASET_CONFIGS = {
    "ml-1m": {
        "dataset":          "ml-1m-historical",
        "historical_path":  "dataset/ml-1m-historical/ml-1m-historical.inter",
        "realtime_path":    "dataset/ml-1m-realtime/ml-1m-realtime.inter",
        "config_files":     ["configs/dataset.yaml", "configs/historical_eval.yaml", "configs/lightgcn.yaml"],
        "checkpoint":           "saved/compatible/LightGCN-ml1m-historical.pth",
        "itemknn_checkpoint":   "saved/compatible/ItemKNN-ml1m-historical.pkl",
        "results_csv":          RESULTS_DIR / "ml1m_hybrid_results.csv",
        "results_png":          RESULTS_DIR / "ml1m_hybrid_curves.png",
        "id_cast":              lambda x: str(int(x)),
        "plot_title":           "Incremental LightGCN vs No Update\n(ml-1m streaming, batch size=1000)",
    },
    "yelp": {
        "dataset":              "yelp-historical",
        "historical_path":      "dataset/yelp-historical/yelp-historical.inter",
        "realtime_path":        "dataset/yelp-realtime/yelp-realtime.inter",
        "config_files":         ["configs/yelp_dataset.yaml", "configs/yelp_historical_eval.yaml", "configs/lightgcn.yaml"],
        "checkpoint":           "saved/compatible/LightGCN-yelp-historical.pth",
        "itemknn_checkpoint":   "saved/compatible/ItemKNN-yelp-historical.pkl",
        "results_csv":          RESULTS_DIR / "yelp_hybrid_results.csv",
        "results_png":          RESULTS_DIR / "yelp_hybrid_curves.png",
        "id_cast":              lambda x: str(x),
        "plot_title":           "Incremental LightGCN vs No Update\n(yelp streaming, batch size=1000)",
    },
}
# ─────────────────────────────────────────────────────────────────────────────


# Step 1: Train on historical data
"""
cfg: dataset configuration
"""

def train_historical(cfg: dict) -> tuple[str, float]:
    """
    Train LightGCN on historical dataset. Returns (checkpoint_path, training_emissions_mg).

    Training emissions are measured with codecarbon and saved to a sidecar
    "<checkpoint>.energy.json" file next to the checkpoint, so the one-time
    training cost survives across runs that later reuse the checkpoint
    (in that case training_emissions_mg is read back from the sidecar, or
    0.0 if no sidecar exists — e.g. a checkpoint trained before this was
    added).
    """
    # If a specific checkpoint is configured, use it directly
    if cfg["checkpoint"] and Path(cfg["checkpoint"]).exists():
        print(f"Found existing checkpoint: {cfg['checkpoint']}")
        energy_file = Path(cfg["checkpoint"]).with_suffix(".energy.json")
        if energy_file.exists():
            training_emissions_mg = json.loads(energy_file.read_text())["training_emissions_mg"]
        else:
            print("  (no emissions sidecar found — historical training emissions unknown for this checkpoint)")
            training_emissions_mg = 0.0
        return cfg["checkpoint"], training_emissions_mg

    print(f"Training LightGCN on {cfg['dataset']}...")
    tracker = EmissionsTracker(
        project_name=f"historical_training_{cfg['dataset']}",
        output_dir=str(RESULTS_DIR),
        log_level="error",
        save_to_file=False,
    )
    tracker.start()
    run_recbole(
        model="LightGCN",
        dataset=cfg["dataset"],
        config_file_list=cfg["config_files"],
    )
    # tracker.stop() returns CO2 emissions in kg (codecarbon's own
    # EmissionsTracker.stop() docstring: "return: CO2 emissions in kgs"),
    # not energy — the *1e6 here converts kg to mg (same multiplication as
    # before, just correctly labeled now instead of being called "µWh").
    kg_co2 = tracker.stop() or 0.0
    training_emissions_mg = kg_co2 * 1e6
    print(f"  Historical training emissions: {training_emissions_mg:.4f} mg CO2eq")

    checkpoints = sorted(glob.glob("saved/LightGCN-*.pth"))
    assert checkpoints, "No LightGCN checkpoint found after training"
    checkpoint_path = checkpoints[-1]

    energy_file = Path(checkpoint_path).with_suffix(".energy.json")
    energy_file.write_text(json.dumps({"training_emissions_mg": training_emissions_mg}))

    return checkpoint_path, training_emissions_mg


# Step 2: Load model and ID mappings

def load_id_mappings(cfg: dict):
    """
    Build the RecBole config/dataset and token→internal_id mappings.
    Returns (user2id, item2id, config, dataset) — config/dataset are handed
    to IncrementalLightGCN.from_checkpoint by the caller to build the model.
    """
    config = Config(
        model="LightGCN",
        dataset=cfg["dataset"],
        config_file_list=cfg["config_files"],
    )
    dataset = create_dataset(config)
    # RecBole token arrays: index = internal ID, value = original token string
    # user_tokens: user_tokens[57] gives the real user ID. 57 is the row number assigned
    user_tokens = dataset.field2id_token["user_id"]
    item_tokens = dataset.field2id_token["item_id"]

    # Reverse the Array and turn it into dict
    # user2id: lets say 4231 is the user id  user2id["4231"] gives 57 which is row
    user2id = {str(tok): idx for idx, tok in enumerate(user_tokens)}
    item2id = {str(tok): idx for idx, tok in enumerate(item_tokens)}
    print(f"  Users in dataset: {dataset.user_num}")
    print(f"  Items in dataset: {dataset.item_num}")
    return user2id, item2id, config, dataset


#  Step 3: Build historical user profiles

def build_user_history(user2id: dict, item2id: dict, cfg: dict) -> dict:
    """
    Build user_internal_id (row number) → set of item_internal_ids (row numbers), from historical data.
    Only keeps users and items that exist in the trained model.
    """
    # historical_path is produced by tools/split_dataset.py, which already
    # filters to rating>=3 via RecBole (configs/*.yaml val_interval), so no
    # re-filtering is needed here.
    df = pd.read_csv(cfg["historical_path"], sep="\t")

    # turn id raw id into string because mapping table built by load id mappings
    # is based on string
    id_cast = cfg["id_cast"]

    #  build history row to row and not id to id
    history = {}
    for _, row in df.iterrows():
        uid = user2id.get(id_cast(row["user_id:token"]))
        iid = item2id.get(id_cast(row["item_id:token"]))
        if uid is None or iid is None:
            continue
        if uid not in history:
            history[uid] = set()
        history[uid].add(iid)

    print(f"  Historical profiles built for {len(history)} users")
    return history


#Step 5: Main streaming loop

"""""
- model — the IncrementalLightGCN instance, already loaded from a checkpoint before this function was called. Holds the actual embedding tables and does the scoring/training.
- user2id — a dictionary mapping real-world user IDs (strings, e.g. "4231") to their row number in the model's embedding table. Built once by load_id_mappings, passed in here, and (for no_update/incremental) grows in place as new users appear in the stream.
- item2id — same idea as user2id, but for items.
- user_history — a dictionary of {uid: {set of item ids this user already interacted with}}, built from the historical data before streaming starts (via build_user_history)
- strategy — a plain string, one of "no_update", "incremental", "full_retrain" — controls which branch of the function runs.
- cfg — the settings dictionary for whichever dataset is running (paths, id_cast, batch_size, etc.) — explained in detail a few messages back.
"""
def run_streaming(model: IncrementalLightGCN, user2id: dict, item2id: dict,
                  user_history: dict, strategy: str, cfg: dict) -> pd.DataFrame:
    """
    Stream realtime data in batches of BATCH_SIZE.
    strategy: 'no_update', 'incremental', or 'full_retrain'

    New users and items that appear in the stream but were not in historical
    training are assigned fresh internal IDs and added to the model on the fly:
      - incremental:  added to graph + embeddings trained via incremental_update
      - no_update:    embeddings expanded with mean init (so they can be scored)
                      but graph and weights never updated
      - full_retrain: every update_every batches, retrains LightGCN from
                      scratch (not warm-started) via RecBole (run_recbole),
                      on all historical + realtime interactions consumed so
                      far — the recall/energy "ceiling" comparison point.
                      Unlike the other two, IDs are looked up per-batch
                      (not grown/precomputed upfront), because a RecBole
                      retrain can renumber IDs — its user_inter_num_interval
                      / item_inter_num_interval filters re-evaluate against
                      the whole combined dataset each retrain, so which IDs
                      even qualify can change. New users/items not yet
                      incorporated by the most recent retrain are skipped
                      from evaluation until the next retrain absorbs them.
    """
    print(f"\nRunning strategy: {strategy}")
    # get info from config file
    id_cast = cfg["id_cast"]
    batch_size = cfg.get("batch_size", BATCH_SIZE)
    update_every = cfg.get("update_every", UPDATE_EVERY)


    records = [] # to keep track

    # Deep copy history so strategies don't interfere with each other
    history = {uid: set(items) for uid, items in user_history.items()}

    if strategy == "full_retrain":
        # both paths are pre-filtered to rating>=3 by tools/split_dataset.py
        realtime_df    = pd.read_csv(cfg["realtime_path"], sep="\t")
        historical_raw = pd.read_csv(cfg["historical_path"], sep="\t")

        df = realtime_df.copy()

        # hold original index
        # the different between this and iid / uid is that interactions of a user is not in one row rather
        # is it distrubtuted all over the raw data so inex is jsut the index of the row
        # of one user and one item interaction
        # we need this because
        df["orig_idx"] = df.index

        #reset so the holes in index goe away
        df = df.reset_index(drop=True)
        n_batches = len(df) // batch_size

        # create the file for the new combined dataset including the config file
        # combined dataset will incude history  + realtime until that point
        combined_name = f"{cfg['dataset']}-combined"
        combined_dir = Path("dataset") / combined_name
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / f"{combined_name}.inter"
        combined_cfg = {**cfg, "dataset": combined_name, "historical_path": str(combined_path)}
        consumed_raw_upto = -1 # mark how far along in realtime data we are

    else:
        # realtime_path is pre-filtered to rating>=3 by tools/split_dataset.py
        df = pd.read_csv(cfg["realtime_path"], sep="\t")

        # Growing ID mappings — new users/items get assigned the next available ID
        # instead of being dropped. user2id and item2id are mutated in place.
        # this is different compare to the full retrain
        next_user_id = [model.n_users]   # list so it's mutable inside lambda
        next_item_id = [model.n_items]

        # based on user id return the row number of the original one or give me row if new user
        # each row represent a user
        def get_or_create_uid(x):
            key = id_cast(x)
            if key not in user2id:
                user2id[key] = next_user_id[0]
                next_user_id[0] += 1
            return user2id[key]

        def get_or_create_iid(x):
            key = id_cast(x)
            if key not in item2id:
                item2id[key] = next_item_id[0]
                next_item_id[0] += 1
            return item2id[key]

        # new column with original row number
        df["uid"] = df["user_id:token"].apply(get_or_create_uid)
        df["iid"] = df["item_id:token"].apply(get_or_create_iid)
        df["uid"] = df["uid"].astype(int)
        df["iid"] = df["iid"].astype(int)

        # Expand model embeddings upfront for all new entities seen in the stream.
        # New rows are initialized with mean of existing embeddings.
        # For incremental: add_interactions will also expand as new IDs arrive per batch.
        # For no_update: this is not important since it is never actually used
        max_uid = int(df["uid"].max()) + 1
        max_iid = int(df["iid"].max()) + 1
        if max_uid > model.n_users or max_iid > model.n_items:
            model.expand_embeddings(max(max_uid, model.n_users), max(max_iid, model.n_items))

        n_batches = len(df) // batch_size

        # Buffer to accumulate new interactions since last update
        buffer_users = []
        buffer_items = []

    for i in range(n_batches):
        batch = df.iloc[i * batch_size: (i + 1) * batch_size]

        if strategy == "full_retrain":
            batch_users, batch_items = [], []
            for _, row in batch.iterrows():
                u  = user2id.get(id_cast(row["user_id:token"]))
                it = item2id.get(id_cast(row["item_id:token"]))
                if u is not None and it is not None and u < model.n_users and it < model.n_items:
                    batch_users.append(u)
                    batch_items.append(it)
        else:
            batch_users = batch["uid"].tolist()
            batch_items = batch["iid"].tolist()

        # Measure metrics before update (reflects current model state, no data leakage)
        if batch_users:
            m = batch_metrics_lgcn(model, batch_users, batch_items, history)
        else:
            m = {"recall@10": 0.0, "precision@10": 0.0, "ndcg@10": 0.0, "hr@10": 0.0,
                 "recall@20": 0.0, "precision@20": 0.0, "ndcg@20": 0.0, "hr@20": 0.0, "mrr": 0.0}
        recall = m["recall@10"]

        update_emissions_mg = 0.0
        updated = False

        if strategy == "incremental":
            # Accumulate interactions into buffer
            buffer_users.extend(batch_users)
            buffer_items.extend(batch_items)

            if (i + 1) % update_every == 0:
                # Use all accumulated interactions since last update (not just current batch)
                # This gives a stronger gradient signal: 20 batches × 1000 = 20,000 interactions
                new_users = np.array(buffer_users)
                new_items = np.array(buffer_items)

                tracker = EmissionsTracker(
                    project_name=f"incremental_update_batch_{i}",
                    output_dir=str(RESULTS_DIR),
                    log_level="error",
                    save_to_file=False,
                )
                tracker.start()
                model.add_interactions(new_users, new_items)
                model.incremental_update(new_users, new_items, n_epochs=UPDATE_EPOCHS)
                # tracker.stop() returns CO2 emissions in kg, not energy;
                # *1e6 converts kg to mg (see train_historical for details).
                kg_co2 = tracker.stop() or 0.0
                update_emissions_mg = kg_co2 * 1e6
                updated = True

                # Clear buffer after update
                buffer_users = []
                buffer_items = []

        elif strategy == "full_retrain":
            consumed_raw_upto = max(consumed_raw_upto, int(batch["orig_idx"].max()))

            if (i + 1) % update_every == 0:
                #
                consumed_realtime = realtime_df.iloc[: consumed_raw_upto + 1]
                combined = pd.concat([historical_raw, consumed_realtime], ignore_index=True)
                combined.to_csv(combined_path, sep="\t", index=False)

                tracker = EmissionsTracker(
                    project_name=f"full_retrain_batch_{i}",
                    output_dir=str(RESULTS_DIR),
                    log_level="error",
                    save_to_file=False,
                )
                tracker.start()
                run_recbole(model="LightGCN", dataset=combined_name, config_file_list=cfg["config_files"])
                # tracker.stop() returns CO2 emissions in kg, not energy;
                # *1e6 converts kg to mg (see train_historical for details).
                kg_co2 = tracker.stop() or 0.0
                update_emissions_mg = kg_co2 * 1e6

                checkpoints = sorted(glob.glob("saved/LightGCN-*.pth"))
                assert checkpoints, "No LightGCN checkpoint found after full retrain"
                raw_ckpt_path = checkpoints[-1]

                # Move out of the generic saved/LightGCN-*.pth pool (shared
                # with historical training) into a dedicated, clearly-labeled
                # location — so full_retrain's periodic snapshots never get
                # confused with the historical starting checkpoint, and stay
                # organized per-dataset and per-batch instead of piling
                retrain_dir = Path("saved/full_retrain") / cfg["dataset"]
                retrain_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = retrain_dir / f"batch_{i + 1:04d}.pth"
                Path(raw_ckpt_path).rename(ckpt_path)
                ckpt_path = str(ckpt_path)

                user2id, item2id, config, dataset = load_id_mappings(combined_cfg)
                model = IncrementalLightGCN.from_checkpoint(ckpt_path, config, dataset)
                history = build_user_history(user2id, item2id, combined_cfg)
                updated = True

        # Add batch to running history
        for uid, iid in zip(batch_users, batch_items):
            if uid not in history:
                history[uid] = set()
            history[uid].add(iid)

        records.append({
            "batch":             i + 1,
            "interactions":      (i + 1) * BATCH_SIZE,
            "strategy":          strategy,
            "recall_at_10":      m["recall@10"],
            "precision_at_10":   m["precision@10"],
            "ndcg_at_10":        m["ndcg@10"],
            "hr_at_10":          m["hr@10"],
            "recall_at_20":      m["recall@20"],
            "precision_at_20":   m["precision@20"],
            "ndcg_at_20":        m["ndcg@20"],
            "hr_at_20":          m["hr@20"],
            "mrr":               m["mrr"],
            "update_emissions_mg": update_emissions_mg,
            "updated":           updated,
        })

        if (i + 1) % 20 == 0 or updated:
            print(f"  Batch {i+1:>3}/{n_batches}  Recall@10={recall:.4f}"
                  + (f"  ← updated ({update_emissions_mg:.4f} mg CO2eq)" if updated else ""))

    return pd.DataFrame(records)


# Step 6: Plot results

def plot_results(df: pd.DataFrame, cfg: dict, training_emissions_mg: float = None):
    plot_streaming_results(df, cfg["results_png"], cfg["plot_title"], UPDATE_EVERY,
                           training_emissions_mg=training_emissions_mg)


# Main

STRATEGIES = ["no_update", "incremental", "full_retrain"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-1m", "yelp"], required=True,
                        help="Dataset to run the experiment on")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Directory to save results (default: results/)")
    parser.add_argument("--no-update", action="store_true",
                        help="Run the no_update baseline strategy (model never updates)")
    parser.add_argument("--incremental", action="store_true",
                        help="Run the incremental update strategy")
    parser.add_argument("--full-retrain", action="store_true",
                        help="Run the full-retrain baseline (expensive — "
                             "retrains LightGCN from scratch every update_every batches)")
    args = parser.parse_args()

    run_flags = {"no_update": args.no_update, "incremental": args.incremental,
                 "full_retrain": args.full_retrain}
    if not any(run_flags.values()):
        parser.error("at least one of --no-update, --incremental, --full-retrain is required")

    results_dir = args.results_dir
    cfg = DATASET_CONFIGS[args.dataset].copy()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Stamp output filenames with the requested strategies + a timestamp, so
    # runs never overwrite each other AND the filename alone tells you which
    # strategy/strategies it holds (useful since strategies are often run as
    # separate invocations rather than all together).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_prefix = args.dataset.replace("-", "")
    strategy_tag = "_".join(s for s in STRATEGIES if run_flags[s])
    cfg["results_csv"] = results_dir / f"{dataset_prefix}_hybrid_results_{strategy_tag}_{ts}.csv"
    cfg["results_png"] = results_dir / f"{dataset_prefix}_hybrid_curves_{strategy_tag}_{ts}.png"
    print(f"Run timestamp: {ts}")

    # Step 1: Train or load checkpoint
    checkpoint, training_emissions_mg = train_historical(cfg)

    # Step 2: Run each requested strategy, each starting from a fresh copy
    # of the same historical checkpoint (so strategies are directly
    # comparable, none of them chain off another's updated weights).
    dfs = {}
    for strategy in STRATEGIES:
        if not run_flags[strategy]:
            continue
        print(f"\nLoading model for {strategy} strategy...")
        user2id, item2id, config, dataset = load_id_mappings(cfg)
        model = IncrementalLightGCN.from_checkpoint(checkpoint, config, dataset)
        user_history = build_user_history(user2id, item2id, cfg)
        dfs[strategy] = run_streaming(model, user2id, item2id, user_history,
                                      strategy=strategy, cfg=cfg)

    # Step 3: Save and plot
    results = pd.concat(list(dfs.values()), ignore_index=True)
    results.to_csv(cfg["results_csv"], index=False)
    print(f"\nResults saved → {cfg['results_csv']}")

    # Historical training emissions are only otherwise persisted in the
    # gitignored saved/*.energy.json sidecar — mirror it into results/
    # (git-tracked) so it survives independently of the checkpoint.
    energy_summary_path = Path(str(cfg["results_csv"]).replace(".csv", "_training_energy.json"))
    energy_summary_path.write_text(json.dumps({"training_emissions_mg": training_emissions_mg}))
    print(f"Training emissions saved → {energy_summary_path}")

    plot_results(results, cfg, training_emissions_mg=training_emissions_mg)

    # Summary
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  Historical training emissions:  {training_emissions_mg:.4f} mg CO2eq")
    for strategy, df in dfs.items():
        avg_recall = df["recall_at_10"].mean()
        print(f"  Avg Recall@10 ({strategy}): {avg_recall:.4f}")
        if strategy == "no_update":
            continue
        total_emissions = df["update_emissions_mg"].sum()
        n_updates    = int(df["updated"].sum())
        print(f"  Total {strategy} emissions:      {total_emissions:.4f} mg CO2eq")
        print(f"  Number of {strategy} updates: {n_updates}")
        print(f"  Avg emissions per update:       {total_emissions / max(n_updates, 1):.4f} mg CO2eq")


if __name__ == "__main__":
    main()
