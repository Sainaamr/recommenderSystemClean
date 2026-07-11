# Incremental LightGCN — Streaming Recommender Experiment

Compares three strategies for keeping a LightGCN recommender up to date as new
interactions stream in — `no_update`, `incremental` (cheap warm-started
updates), and `full_retrain` (periodic full retraining via RecBole) — tracking
both recommendation quality (Recall/NDCG/HR/MRR) and energy cost per update.

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

Per-user, time-ordered 80/20 split (first 80% of each user's interactions →
historical/training, last 20% → realtime/streaming). This is also what
triggers ml-1m's auto-download if it hasn't happened yet.

```bash
cd ~/recommenderSystemClean
source .venv/bin/activate
python3 tools/split_dataset.py --dataset ml-1m
python3 tools/split_dataset.py --dataset yelp
```

Verify:
```bash
ls dataset/ml-1m-historical/ dataset/ml-1m-realtime/ dataset/yelp-historical/ dataset/yelp-realtime/
```
Each should contain one `.inter` file.

## Running an experiment

Long-running — use `screen` (or `tmux`) so it survives disconnecting:

```bash
screen -S recsys
cd ~/recommenderSystemClean
source .venv/bin/activate
python3 experiments/run_incremental_lightgcn.py --dataset yelp
```

Add `--full-retrain` to also run the (expensive) full-retrain baseline
strategy. Detach with `Ctrl+A` then `D`; reattach later with `screen -r recsys`.

Results land in `results/{dataset}_hybrid_results_{timestamp}.csv` (per-batch
Recall/NDCG/HR/MRR + energy) and a matching `_incremental_epochs.csv`
(per-update convergence info: how many epochs each incremental update
actually ran before early stopping).
