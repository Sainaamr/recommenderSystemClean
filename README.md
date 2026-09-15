# Incremental LightGCN — Streaming Recommender Experiment

Compares strategies for keeping a LightGCN recommender up to date as new
interactions stream in, tracking both recommendation quality
(Recall/NDCG/HR/MRR) and energy cost per update.

| Strategy | Script | What it does |
|---|---|---|
| `no_update` | `run_incremental_lightgcn.py --no-update` | model frozen; new users get a mean embedding |
| `incremental` | `run_incremental_lightgcn.py --incremental` | warm-started BPR steps every `update_every` batches |
| `full_retrain` | `run_incremental_lightgcn.py --full-retrain` | retrains from scratch via RecBole every `update_every` batches |
| content-init | `run_content_coldstart.py` | new users seeded from item metadata instead of the mean; no retraining |
| content-incremental | `run_content_incremental.py` | content seeding **and** incremental updates |

`run_new_user_analysis.py` it freezes
the model and scores existing vs. never-trained users separately, to show how
much of the quality decay is cold start.

Comparison tables come from `tools/compare_results.py`, figures from
`tools/plot_utils.py`.

## Setup

### 1. Clone the repo

```bash
git clone git@github.com:Sainaamr/recommenderSystemClean.git
cd recommenderSystemClean
```

### 2. Create the virtual environment

Use a modern Python (3.10+) — some servers default `python3` to an older
version (e.g. 3.8), which causes dependency resolution failures against the
pinned versions below.

```bash
which python3.12 python3.11 python3.10 2>/dev/null   # check what's available
python3.12 -m venv .venv                              # use whichever you found
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -r requirements-frozen.txt
```

### 4. Patch RecBole

RecBole 1.2.0 predates NumPy 2.0 and recent PyTorch versions, and needs a few
lines patched in the installed package:

```bash
RECBOLE_PATH=$(python3 -c "import recbole, os; print(os.path.dirname(recbole.__file__))")

# NumPy 2.0 removed these old aliases
sed -i 's/np.float = np.float_/np.float = np.float64/' "$RECBOLE_PATH/config/configurator.py"
sed -i 's/np.complex = np.complex_/np.complex = np.complex128/' "$RECBOLE_PATH/config/configurator.py"
sed -i 's/np.object = np.object_/np.object = object/' "$RECBOLE_PATH/config/configurator.py"
sed -i 's/np.unicode = np.unicode_/np.unicode = np.str_/' "$RECBOLE_PATH/config/configurator.py"

# PyTorch 2.6+ changed torch.load's default to weights_only=True, which
# breaks loading RecBole checkpoints (they pickle a full Config object)
sed -i 's/torch.load(model_file)/torch.load(model_file, weights_only=False)/' "$RECBOLE_PATH/quick_start/quick_start.py"
sed -i 's/torch.load(resume_file, map_location=self.device)/torch.load(resume_file, map_location=self.device, weights_only=False)/' "$RECBOLE_PATH/trainer/trainer.py"
sed -i 's/torch.load(checkpoint_file, map_location=self.device)/torch.load(checkpoint_file, map_location=self.device, weights_only=False)/' "$RECBOLE_PATH/trainer/trainer.py"

# Verify
grep -n "np.float\|np.complex\|np.object\|np.unicode" "$RECBOLE_PATH/config/configurator.py"
grep -n "weights_only" "$RECBOLE_PATH/quick_start/quick_start.py" "$RECBOLE_PATH/trainer/trainer.py"
```

Do **not** patch `lightgcn.py`'s `A.update(data_dict)` line — it's already
correct for this project's pinned scipy version.

### 5. Get the raw datasets

`dataset/` is gitignored (not committed — raw/split data can be large and
shouldn't live in the repo). Neither dataset is bundled; both need to be
obtained locally before the split step below.

**ml-1m** — no manual download needed. RecBole auto-downloads it the first
time it's referenced (triggered automatically by Step 6 below). Watch for a
`"Prepare to download dataset [ml-1m] from..."` message confirming it worked.

**Yelp** — download from the official source, not RecBole's built-in version
(RecBole's hosted Yelp is a different snapshot and won't match results
produced against this project's data):

1. Download the Yelp Academic Dataset from <https://www.yelp.com/dataset>
   (requires accepting Yelp's data agreement).
2. Convert the downloaded JSON files into RecBole's atomic-file format using
   [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets)' conversion
   tool — see their
   [Yelp conversion guide](https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/Yelp.md):
   ```bash
   git clone https://github.com/RUCAIBox/RecSysDatasets.git
   cd RecSysDatasets/conversion_tools
   python run.py --dataset yelp --input_path <path_to_decompressed_yelp_download> \
       --output_path output_data/yelp --convert_inter
   ```
3. Place the resulting `yelp.inter` at `dataset/yelp/yelp.inter` in this repo.

### 6. Split into historical/realtime

Two splits exist. **Use the time cut** — it is what every current result is
based on. Either one also triggers ml-1m's auto-download if it hasn't happened
yet.

**Global time cut (`split_dataset_timecut.py`)** — one cut date at the 80th
percentile of the timestamp distribution: everything on or before it trains,
everything after streams. No training interaction post-dates a streamed one, so
the stream can be read chronologically without the model having seen the future.

```bash
cd ~/recommenderSystemClean
source .venv/bin/activate
python3 tools/split_dataset_timecut.py --dataset ml-1m
python3 tools/split_dataset_timecut.py --dataset yelp
```

```bash
ls dataset/ml-1m-historical-timecut/ dataset/ml-1m-realtime-timecut/ \
   dataset/yelp-historical-timecut/ dataset/yelp-realtime-timecut/
```

**Per-user 80/20 (`split_dataset.py`)** — splits each user's own timeline, so
the historical set spans the whole period. Kept for reference only: it leaks,
because at the start of the stream almost all training data post-dates the
interaction being predicted, and its batches are user blocks rather than time
slices.

```bash
python3 tools/split_dataset.py --dataset ml-1m
python3 tools/split_dataset.py --dataset yelp
```

**Unfiltered stream (`split_dataset_timecut_unfiltered.py`)** — optional. Same
cut and rating filter, but no minimum-interaction filter, so light and
late-arriving users survive into the stream. Writes the stream only; the
historical portion and its checkpoint are unchanged. Used to test whether
new-user arrivals are constant once the filter's right-censoring is removed.

```bash
python3 tools/split_dataset_timecut_unfiltered.py --dataset yelp
```

Each output directory should contain one `.inter` file.

## Running an experiment

Long-running — use `screen` (or `tmux`) so it survives disconnecting:

```bash
screen -S recsys
cd ~/recommenderSystemClean
source .venv/bin/activate
python3 experiments/run_incremental_lightgcn.py --dataset yelp-timecut --no-update --incremental --full-retrain
```

Each strategy is opt-in via its own flag — `--no-update`, `--incremental`,
`--full-retrain` — and at least one is required. Mix and match to run only
what you need (e.g. just `--incremental` while iterating on its
hyperparameters). `--full-retrain` is the expensive one (retrains LightGCN
from scratch every `update_every` batches via RecBole). Detach with `Ctrl+A`
then `D`; reattach later with `screen -r recsys`.

`--dataset` selects an entry from `DATASET_CONFIGS` in
`experiments/run_incremental_lightgcn.py`: `yelp-timecut` / `ml-1m-timecut`
(the time cut), `yelp` / `ml-1m` (the per-user split), plus
`yelp-timecut-unfiltered` and `yelp-timeorder`. `--results-dir` overrides the
default `results/`; give concurrent runs their own directory, since every run
in one invocation shares a single start-of-process timestamp and would
otherwise overwrite.

Outputs, where `{prefix}` is the dataset key without hyphens and `{ts}` is
`YYYYMMDD_HHMMSS`:

```
{prefix}_hybrid_results_{strategy}_{ts}.csv    per-batch metrics + per-phase energy, one file per strategy
{prefix}_hybrid_emissions_summary_{ts}.csv     per-phase energy totals for the run
```

No figures are written; plot from the CSVs with `tools/plot_utils.py`. The
content scripts follow the same pattern and additionally write a
`*_energy.csv` sidecar; they only produce figures in replot mode (`--csv`).
