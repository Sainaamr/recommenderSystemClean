
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
        "checkpoint":       "saved/compatible/LightGCN-ml1m-historical.pth",
        "id_cast":          lambda x: str(int(x)),
    },
    "yelp": {
        "dataset":          "yelp-historical",
        "historical_path":  "dataset/yelp-historical/yelp-historical.inter",
        "realtime_path":    "dataset/yelp-realtime/yelp-realtime.inter",
        "config_files":     ["configs/yelp_dataset.yaml", "configs/yelp_historical_eval.yaml", "configs/lightgcn.yaml"],
        "checkpoint":       "saved/compatible/LightGCN-yelp-historical.pth",
        "id_cast":          lambda x: str(x),
    },
}
# ─────────────────────────────────────────────────────────────────────────────


def task_mg(task_result) -> float:
    """codecarbon stop_task() result (kg CO2, or None) -> mg CO2eq."""
    return (task_result.emissions if task_result else 0.0) * 1e6


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

    # Vectorized map + groupby instead of iterrows() — ~17x faster at
    # ml-1m/yelp scale, verified identical output on real data.
    uid_col = df["user_id:token"].map(lambda x: user2id.get(id_cast(x)))
    iid_col = df["item_id:token"].map(lambda x: item2id.get(id_cast(x)))
    mask = uid_col.notna() & iid_col.notna()
    uid_valid = uid_col[mask].astype(int)
    iid_valid = iid_col[mask].astype(int)
    history = pd.DataFrame({"uid": uid_valid, "iid": iid_valid}) \
                .groupby("uid")["iid"].apply(set).to_dict()

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
                  user_history: dict, strategy: str, cfg: dict,
                  tracker: EmissionsTracker) -> tuple[pd.DataFrame, float]:
    """
    Stream realtime data in batches of BATCH_SIZE.
    strategy: 'no_update', 'incremental', or 'full_retrain'

    New users and items that appear in the stream but were not in historical
    training are assigned fresh internal IDs and added to the model on the fly:
      - incremental:  added to graph + embeddings trained via incremental_update
      - no_update:    embeddings expanded with mean init (so they can be scored)
                      but graph and weights never updated
      - full_retrain: every update_every batches, retrains LightGCN from
                      scratch  via RecBole
                      on all historical + realtime interactions consumed so
                      far — the recall/energy "ceiling" comparison point.
    """
    print(f"\nRunning strategy: {strategy}")
    # get info from config file
    id_cast = cfg["id_cast"]
    batch_size = cfg.get("batch_size", BATCH_SIZE)
    update_every = cfg.get("update_every", UPDATE_EVERY)


    records = [] # to keep track
    batch_loop_emissions_mg = 0.0

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

        n_batches = len(df) // batch_size

        # Buffer to accumulate new interactions since last update
        buffer_users = []
        buffer_items = []

    for i in range(n_batches):
        tracker.start_task(f"batch_{i}")
        batch = df.iloc[i * batch_size: (i + 1) * batch_size].copy()

        if strategy == "full_retrain":
            batch_users, batch_items = [], []
            for _, row in batch.iterrows():
                u  = user2id.get(id_cast(row["user_id:token"]))
                it = item2id.get(id_cast(row["item_id:token"]))
                if u is not None and it is not None and u < model.n_users and it < model.n_items:
                    batch_users.append(u)
                    batch_items.append(it)
        else:
            # per-batch ID resolution
            batch["uid"] = batch["user_id:token"].apply(get_or_create_uid).astype(int)
            batch["iid"] = batch["item_id:token"].apply(get_or_create_iid).astype(int)
            batch_users = batch["uid"].tolist()
            batch_items = batch["iid"].tolist()

            # expansion — only grow the table if THIS batch actually
            # introduced a uid/iid beyond the model's current size
            max_u = max(batch_users, default=0)
            max_i = max(batch_items, default=0)
            if max_u >= model.n_users or max_i >= model.n_items:
                model.expand_embeddings(
                    max(max_u + 1, model.n_users),
                    max(max_i + 1, model.n_items),
                )

        # Measure metrics before update (reflects current model state, no data leakage)
        if batch_users:
            m = batch_metrics_lgcn(model, batch_users, batch_items, history)
        else:
            m = {"recall@10": 0.0, "precision@10": 0.0, "ndcg@10": 0.0, "hr@10": 0.0,
                 "recall@20": 0.0, "precision@20": 0.0, "ndcg@20": 0.0, "hr@20": 0.0, "mrr": 0.0}
        recall = m["recall@10"]

        needs_update = False

        if strategy == "incremental":
            # Accumulate interactions into buffer
            buffer_users.extend(batch_users)
            buffer_items.extend(batch_items)
            needs_update = (i + 1) % update_every == 0
            if needs_update:
                # Use all accumulated interactions since last update (not just current batch)
                # This gives a stronger gradient signal: 20 batches × 1000 = 20,000 interactions
                new_users = np.array(buffer_users)
                new_items = np.array(buffer_items)

        elif strategy == "full_retrain":
            consumed_raw_upto = max(consumed_raw_upto, int(batch["orig_idx"].max()))
            needs_update = (i + 1) % update_every == 0

        # Add batch to running history
        for uid, iid in zip(batch_users, batch_items):
            if uid not in history:
                history[uid] = set()
            history[uid].add(iid)

        batch_emissions_mg = task_mg(tracker.stop_task())
        batch_loop_emissions_mg += batch_emissions_mg

        update_emissions_mg = 0.0
        updated = False

        if strategy == "incremental" and needs_update:

            tracker.start_task(f"batch_{i}_update")
            model.add_interactions(new_users, new_items)
            model.incremental_update(new_users, new_items, n_epochs=UPDATE_EPOCHS)
            update_emissions_mg = task_mg(tracker.stop_task())
            batch_loop_emissions_mg += update_emissions_mg
            updated = True

            # Clear buffer after update
            buffer_users = []
            buffer_items = []

        elif strategy == "full_retrain" and needs_update:
            tracker.start_task(f"batch_{i}_update")
            consumed_realtime = realtime_df.iloc[: consumed_raw_upto + 1]
            combined = pd.concat([historical_raw, consumed_realtime], ignore_index=True)
            combined.to_csv(combined_path, sep="\t", index=False)

            run_recbole(model="LightGCN", dataset=combined_name, config_file_list=cfg["config_files"])

            checkpoints = sorted(glob.glob("saved/LightGCN-*.pth"))
            assert checkpoints, "No LightGCN checkpoint found after full retrain"
            raw_ckpt_path = checkpoints[-1]

            retrain_dir = Path("saved/full_retrain") / cfg["dataset"]
            retrain_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = retrain_dir / f"batch_{i + 1:04d}.pth"
            Path(raw_ckpt_path).rename(ckpt_path)
            ckpt_path = str(ckpt_path)

            user2id, item2id, config, dataset = load_id_mappings(combined_cfg)
            model = IncrementalLightGCN.from_checkpoint(ckpt_path, config, dataset)
            history = build_user_history(user2id, item2id, combined_cfg)

            update_emissions_mg = task_mg(tracker.stop_task())
            batch_loop_emissions_mg += update_emissions_mg
            updated = True

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
            "batch_emissions_mg":   batch_emissions_mg,
            "update_emissions_mg": update_emissions_mg,
            "updated":           updated,
        })

        if (i + 1) % 20 == 0 or updated:
            print(f"  Batch {i+1:>3}/{n_batches}  Recall@10={recall:.4f}"
                  + (f"  ← updated ({update_emissions_mg:.4f} mg CO2eq)" if updated else ""))

    return pd.DataFrame(records), batch_loop_emissions_mg


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

    # Stamp output filenames with a shared timestamp, so runs never overwrite
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_prefix = args.dataset.replace("-", "")
    print(f"Run timestamp: {ts}")

    # Step 1: Train or load checkpoint
    checkpoint, training_emissions_mg = train_historical(cfg)

    # Step 2: Run each requested strategy, each starting from a fresh copy
    # of the same historical checkpoint
    dfs = {}
    csv_paths = {}
    streaming_emissions_mg = {}
    section_emissions_mg = {}   # strategy -> {section_name: mg}, saved separately below
    for strategy in STRATEGIES:
        if not run_flags[strategy]:
            continue
        print(f"\nLoading model for {strategy} strategy...")


        tracker = EmissionsTracker(
            project_name=f"streaming_{strategy}_{cfg['dataset']}",
            output_dir=str(RESULTS_DIR),
            log_level="error",
            save_to_file=False,
        )
        tracker.start()

        tracker.start_task("load_id_mappings")
        user2id, item2id, config, dataset = load_id_mappings(cfg)
        id_mapping_emissions_mg = task_mg(tracker.stop_task())

        tracker.start_task("load_checkpoint")
        model = IncrementalLightGCN.from_checkpoint(checkpoint, config, dataset)
        checkpoint_load_emissions_mg = task_mg(tracker.stop_task())

        tracker.start_task("build_user_history")
        user_history = build_user_history(user2id, item2id, cfg)
        build_user_history_emissions_mg = task_mg(tracker.stop_task())

        df, batch_loop_emissions_mg = run_streaming(model, user2id, item2id, user_history,
                           strategy=strategy, cfg=cfg, tracker=tracker)
        tracker.stop()

        sections = {
            "id_mapping_emissions_mg":         id_mapping_emissions_mg,
            "checkpoint_load_emissions_mg":    checkpoint_load_emissions_mg,
            "build_user_history_emissions_mg": build_user_history_emissions_mg,
            "batch_scoring_emissions_mg":      df["batch_emissions_mg"].sum(),
            "update_total_emissions_mg":       df["update_emissions_mg"].sum(),
        }
        section_emissions_mg[strategy] = sections
        # Whole-run emissions = sum of every separately-measured section —
        # matches the convention in run_content_coldstart.py /
        # run_content_incremental.py, so streaming_emissions_mg is directly
        # comparable across all three scripts.
        streaming_emissions_mg[strategy] = sum(sections.values())
        dfs[strategy] = df

        # Step 3: Save each strategy's results to its own CSV immediately
        strategy_csv = results_dir / f"{dataset_prefix}_hybrid_results_{strategy}_{ts}.csv"
        df.to_csv(strategy_csv, index=False)
        csv_paths[strategy] = strategy_csv
        print(f"Results saved → {strategy_csv}")

    energy_row = {"training_emissions_mg": training_emissions_mg}
    for strategy, df in dfs.items():
        for section_name, section_mg in section_emissions_mg[strategy].items():
            energy_row[f"{strategy}_{section_name}"] = section_mg
        energy_row[f"{strategy}_streaming_emissions_mg"] = streaming_emissions_mg[strategy]
        if strategy == "no_update":
            continue
        total_emissions = section_emissions_mg[strategy]["update_total_emissions_mg"]
        n_updates = int(df["updated"].sum())
        energy_row[f"{strategy}_n_updates"] = n_updates
        energy_row[f"{strategy}_avg_emissions_per_update_mg"] = total_emissions / max(n_updates, 1)

    energy_summary_path = results_dir / f"{dataset_prefix}_hybrid_emissions_summary_{ts}.csv"
    pd.DataFrame([energy_row]).to_csv(energy_summary_path, index=False)
    print(f"Emissions summary saved → {energy_summary_path}")

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
