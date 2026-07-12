"""
Regenerate plots from an already-saved results CSV, without rerunning
training or streaming.

Usage:
    python3 tools/regenerate_plots.py --dataset ml-1m \
        --results-csv results/ml1m_hybrid_results_20260711_162722.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from experiments.run_incremental_lightgcn import DATASET_CONFIGS, UPDATE_EVERY
from tools.plot_utils import plot_streaming_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True,
                        help="Path to a saved results CSV, e.g. "
                             "results/ml1m_hybrid_results_20260711_162722.csv")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS), required=True,
                        help="Which dataset config to pull the plot title from")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)

    energy_path = Path(str(args.results_csv).replace(".csv", "_training_energy.json"))
    training_energy_uwh = None
    if energy_path.exists():
        training_energy_uwh = json.loads(energy_path.read_text())["training_energy_uwh"]
    else:
        print(f"  (no training-energy sidecar at {energy_path} — "
              f"energy plot will have no historical-training reference line)")

    out_png = Path(str(args.results_csv).replace("_results_", "_curves_").replace(".csv", ".png"))
    plot_streaming_results(df, out_png, DATASET_CONFIGS[args.dataset]["plot_title"], UPDATE_EVERY,
                           training_energy_uwh=training_energy_uwh)


if __name__ == "__main__":
    main()
