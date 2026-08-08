"""
Statistically compare two already-saved results CSVs (e.g. no_update vs
incremental), batch-for-batch, on every metric they have in common.

For each shared metric, reports:
  - mean gap             : mean(comparison) - mean(baseline)
  - % improvement        : mean gap as a percentage of the baseline's mean
  - win rate              : fraction of batches where comparison > baseline
  - Wilcoxon signed-rank test (paired, non-parametric) p-value

The two CSVs must share a 'batch' column (they do, if both came from
run_incremental_lightgcn.py streaming the same realtime file) — rows are
aligned on 'batch' before comparing, so this remains correct even if one
file is missing a few batches the other has.

Usage:
    python3 tools/compare_results.py \
        --baseline-csv results/old/yelp_hybrid_results_no_update_20260716_184135.csv \
        --comparison-csv results/yelp_hybrid_results_incremental_....csv

    # restrict to one metric instead of auto-detecting all shared ones
    python3 tools/compare_results.py --baseline-csv ... --comparison-csv ... \
        --metric recall_at_10

    # also save the summary table as a CSV
    python3 tools/compare_results.py --baseline-csv ... --comparison-csv ... \
        --out results/no_update_vs_incremental_compare.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from tools.plot_utils import METRIC_LABELS


def compare_metric(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str) -> dict:
    """
    df_a: baseline, df_b: comparison. Rows aligned on 'batch' before
    comparing, so mismatched batch counts between the two files can't
    silently misalign the pairing.
    """
    merged = pd.merge(df_a[["batch", metric]], df_b[["batch", metric]],
                      on="batch", suffixes=("_baseline", "_comparison"))
    a = merged[f"{metric}_baseline"].to_numpy()
    b = merged[f"{metric}_comparison"].to_numpy()

    mean_gap = b.mean() - a.mean()
    pct_improvement = (mean_gap / a.mean() * 100) if a.mean() != 0 else float("nan")
    win_rate = (b > a).mean()

    # Wilcoxon signed-rank test on the paired differences; raises if every
    # difference is exactly zero (won't happen with real noisy metrics).
    try:
        _, p_value = wilcoxon(b, a)
    except ValueError:
        p_value = float("nan")

    return {
        "metric":          METRIC_LABELS.get(metric, metric),
        "n_batches":       len(merged),
        "mean_baseline":   a.mean(),
        "mean_comparison": b.mean(),
        "mean_gap":        mean_gap,
        "pct_improvement": pct_improvement,
        "win_rate":        win_rate,
        "wilcoxon_p":      p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, required=True,
                        help="Results CSV for the baseline strategy (e.g. no_update)")
    parser.add_argument("--comparison-csv", type=Path, required=True,
                        help="Results CSV for the strategy being compared against the baseline")
    parser.add_argument("--metric", type=str, default=None,
                        help="Restrict to one metric column; default is every metric "
                             "column present in both CSVs")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save the summary table as a CSV to this path")
    args = parser.parse_args()

    df_a = pd.read_csv(args.baseline_csv)
    df_b = pd.read_csv(args.comparison_csv)

    if args.metric:
        metrics = [args.metric]
    else:
        metrics = [m for m in METRIC_LABELS if m in df_a.columns and m in df_b.columns]
        if not metrics:
            parser.error("No shared metric columns found between the two CSVs — "
                         "pass --metric explicitly with a column name present in both.")

    print(f"Baseline:   {args.baseline_csv}")
    print(f"Comparison: {args.comparison_csv}\n")

    rows = [compare_metric(df_a, df_b, m) for m in metrics]
    summary = pd.DataFrame(rows).set_index("metric")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary.to_string())

    if args.out:
        summary.to_csv(args.out)
        print(f"\nSummary saved → {args.out}")


if __name__ == "__main__":
    main()
