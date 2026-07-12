"""
Combine separately-run strategy CSVs (e.g. no_update run alone, incremental
run alone, full_retrain run alone) into one dataframe and re-plot, so you get
the same side-by-side comparison chart you'd get from running all strategies
in a single invocation.

Usage:
    python3 tools/combine_results.py --dataset ml-1m \
        --results-csv results/ml1m_hybrid_results_20260712_190533.csv \
        --results-csv results/ml1m_hybrid_results_20260713_083012.csv \
        --results-csv results/ml1m_hybrid_results_20260714_101500.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from experiments.run_incremental_lightgcn import DATASET_CONFIGS, UPDATE_EVERY, RESULTS_DIR
from tools.plot_utils import plot_streaming_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, action="append", required=True,
                        help="Path to a saved results CSV; repeat this flag once per file to combine")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS), required=True,
                        help="Which dataset config to pull the plot title from")
    args = parser.parse_args()

    dfs = [pd.read_csv(path) for path in args.results_csv]
    combined = pd.concat(dfs, ignore_index=True)

    strategies_in_each_file = [set(df["strategy"].unique()) for df in dfs]
    overlap = set()
    for i, s1 in enumerate(strategies_in_each_file):
        for s2 in strategies_in_each_file[i + 1:]:
            overlap |= (s1 & s2)
    if overlap:
        print(f"  WARNING: strategy/strategies {sorted(overlap)} appear in more than one "
              f"input file — combined CSV will contain duplicate rows for them.")

    training_energy_uwh = None
    for path in args.results_csv:
        energy_path = Path(str(path).replace(".csv", "_training_energy.json"))
        if energy_path.exists():
            training_energy_uwh = json.loads(energy_path.read_text())["training_energy_uwh"]
            break
    if training_energy_uwh is None:
        print("  (no training-energy sidecar found in any input file — "
              "energy plot will have no historical-training reference line)")

    dataset_prefix = args.dataset.replace("-", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = RESULTS_DIR / f"{dataset_prefix}_hybrid_results_combined_{ts}.csv"
    out_png = RESULTS_DIR / f"{dataset_prefix}_hybrid_curves_combined_{ts}.png"

    combined.to_csv(out_csv, index=False)
    print(f"Combined CSV saved → {out_csv} "
          f"(strategies: {sorted(combined['strategy'].unique())})")

    plot_streaming_results(combined, out_png, DATASET_CONFIGS[args.dataset]["plot_title"], UPDATE_EVERY,
                           training_energy_uwh=training_energy_uwh)


if __name__ == "__main__":
    main()
