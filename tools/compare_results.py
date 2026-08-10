"""
Statistically compare two paired sequences of a metric, batch-for-batch.

One purpose-built function per comparison — import and call directly, or
use the matching CLI subcommand. Add a new function + subcommand here as
new comparisons come up; each is self-contained.

  compare_no_update_vs_incremental(baseline_csv, comparison_csv, ...)
      Two results CSVs from run_incremental_lightgcn.py — e.g. no_update vs
      incremental, same metric column name in both files. Columns are named
      after the two strategies (mean_no_update, mean_incremental, ...) by
      default, since the two experiments are genuinely different things,
      not a generic "baseline"/"comparison" pair.

  compare_new_user_analysis(csv, ...)
      One results CSV from run_new_user_analysis.py, which has
      existing/new_user/overall all as separate columns
      (recall_existing, recall_new_user, recall_overall, ...) in one file.
      'overall' is the actual reference point here — it's the real blended
      metric across every user, not an arbitrary third leg — so both
      'existing' and 'new_user' are reported as gap/% vs 'overall', with
      all three raw means shown side by side in the same row.

For each metric compared, reports:
  - mean of each group involved
  - gap             : mean(group) - mean(reference)
  - % improvement   : gap as a percentage of the reference's mean
  - win rate        : fraction of batches where group > reference
  - Wilcoxon signed-rank test (paired, non-parametric) p-value

Rows are always aligned on 'batch' before comparing (an inner merge), so
mismatched batch counts between two files can't silently misalign the
pairing.

CLI usage:
    python3 tools/compare_results.py no-update-vs-incremental \
        --baseline-csv results/old/yelp_hybrid_results_no_update_20260716_184135.csv \
        --comparison-csv results/yelp_hybrid_results_incremental_....csv \
        [--metric recall_at_10] [--out out.csv] [--latex out.tex]

    python3 tools/compare_results.py new-user-analysis \
        --csv results/yelp_new_user_analysis_20260808_200908.csv \
        [--out out.csv] [--latex out.tex]

Python usage:
    from tools.compare_results import compare_no_update_vs_incremental, compare_new_user_analysis
    compare_no_update_vs_incremental("no_update.csv", "incremental.csv")
    compare_new_user_analysis("new_user_analysis.csv")
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from tools.plot_utils import METRIC_LABELS, _CONTENT_COLDSTART_METRICS


def compare_columns(df_a: pd.DataFrame, col_a: str, label_a: str,
                    df_b: pd.DataFrame, col_b: str, label_b: str) -> dict:
    """
    Paired comparison between col_a (the reference/baseline) and col_b (the
    group being compared against it), aligned on 'batch'. df_a and df_b may
    be the same DataFrame (comparing two columns within one CSV — col_a !=
    col_b) or two different DataFrames (comparing the same column name
    across two CSVs — col_a == col_b). Columns are renamed to fixed
    internal names before merging so both cases work without a name
    collision either way.

    Returns a dict keyed "mean_{label_a}", "mean_{label_b}", "mean_gap"
    (b - a), "pct_improvement" (gap as % of a's mean), "win_rate" (fraction
    of batches where b > a), "wilcoxon_p", "n_batches".
    """
    left  = df_a[["batch", col_a]].rename(columns={col_a: "_a"})
    right = df_b[["batch", col_b]].rename(columns={col_b: "_b"})
    merged = pd.merge(left, right, on="batch")
    a = merged["_a"].to_numpy()
    b = merged["_b"].to_numpy()

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
        f"mean_{label_a}": a.mean(),
        f"mean_{label_b}": b.mean(),
        "mean_gap":        mean_gap,
        "pct_improvement": pct_improvement,
        "win_rate":        win_rate,
        "wilcoxon_p":      p_value,
        "n_batches":       len(merged),
    }


def format_for_latex(summary: pd.DataFrame) -> pd.DataFrame:
    """
    summary: a metric-indexed DataFrame as built by the compare_* functions
    below. Raw floats format badly for a thesis table as-is — mean/gap
    columns want a fixed decimal count, pct columns read better with a
    literal '%', and wilcoxon_p rounds to '0.0000' at 4 decimals (these
    p-values commonly run ~1e-30 to 1e-90), so it gets scientific notation
    instead. Matches on column-name prefix rather than a fixed column list,
    since the two compare_* functions produce different column sets.
    Returns a string-valued DataFrame ready for to_latex(), with the index
    restored as plain columns — except a nameless default RangeIndex (e.g.
    summarize_new_user_windows's single-row summary), which carries no
    information and would otherwise show up as a meaningless "index" column.
    """
    has_real_index = summary.index.name is not None or isinstance(summary.index, pd.MultiIndex)
    df = summary.reset_index() if has_real_index else summary.copy()
    for col in df.columns:
        if col == "n_batches" or col == "metric":
            continue
        if col.startswith("mean_") or col.startswith("gap_"):
            df[col] = df[col].map(lambda x: f"{x:.4f}")
        elif col.startswith("pct_"):
            df[col] = df[col].map(lambda x: f"{x:.2f}\\%")
        elif col.startswith("win_rate"):
            df[col] = df[col].map(lambda x: f"{x:.3f}")
        elif col.startswith("wilcoxon_p"):
            df[col] = df[col].map(lambda x: f"{x:.2e}")
    return df


def _report(summary: pd.DataFrame, out: Path = None, latex: Path = None):
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary.to_string())
    if out:
        has_real_index = summary.index.name is not None or isinstance(summary.index, pd.MultiIndex)
        summary.to_csv(out, index=has_real_index)
        print(f"\nSummary saved → {out}")
    if latex:
        Path(latex).write_text(format_for_latex(summary).to_latex(index=False))
        print(f"LaTeX table saved → {latex}")


def compare_no_update_vs_incremental(baseline_csv, comparison_csv,
                                     baseline_label: str = "no_update",
                                     comparison_label: str = "incremental",
                                     metric: str = None,
                                     out: Path = None, latex: Path = None) -> pd.DataFrame:
    """
    Compare two results CSVs from run_incremental_lightgcn.py, batch-for-
    batch, on every metric column they have in common (or just `metric`,
    if given). baseline_label/comparison_label name the mean columns
    (mean_{baseline_label}, mean_{comparison_label}) — override them if
    you're comparing something other than the default no_update/incremental
    pair (e.g. full_retrain vs incremental).
    """
    df_a = pd.read_csv(baseline_csv)
    df_b = pd.read_csv(comparison_csv)

    metrics = [metric] if metric else [m for m in METRIC_LABELS if m in df_a.columns and m in df_b.columns]
    if not metrics:
        raise ValueError("No shared metric columns found between the two CSVs — "
                         "pass metric= explicitly with a column name present in both.")

    print(f"{baseline_label} (baseline):   {baseline_csv}")
    print(f"{comparison_label} (comparison): {comparison_csv}\n")

    rows = []
    for m in metrics:
        row = compare_columns(df_a, m, baseline_label, df_b, m, comparison_label)
        row["metric"] = METRIC_LABELS.get(m, m)
        rows.append(row)

    col_order = [f"mean_{baseline_label}", f"mean_{comparison_label}",
                "mean_gap", "pct_improvement", "win_rate", "wilcoxon_p", "n_batches"]
    summary = pd.DataFrame(rows).set_index("metric")[col_order]
    _report(summary, out, latex)
    return summary


def compare_no_update_vs_content_init(no_update_csv, content_csv,
                                      metric: str = None,
                                      out: Path = None, latex: Path = None) -> pd.DataFrame:
    """
    Compare a run_incremental_lightgcn.py no_update CSV's overall
    recall/precision/ndcg ("{metric}_at_10") against a
    run_content_coldstart.py CSV's overall, content-init performance
    ("{metric}_overall_content"), batch-for-batch. Column names differ
    between the two sources (different scripts, different schemas), unlike
    compare_no_update_vs_incremental where both sides share one CSV schema.
    """
    df_a = pd.read_csv(no_update_csv)
    df_b = pd.read_csv(content_csv)

    metrics = [metric] if metric else ["recall", "precision", "ndcg"]

    print(f"no_update (baseline):      {no_update_csv}")
    print(f"content_init (comparison): {content_csv}\n")

    rows = []
    for m in metrics:
        row = compare_columns(df_a, f"{m}_at_10", "no_update",
                              df_b, f"{m}_overall_content", "content_init")
        row["metric"] = METRIC_LABELS.get(f"{m}_at_10", m)
        rows.append(row)

    col_order = ["mean_no_update", "mean_content_init",
                "mean_gap", "pct_improvement", "win_rate", "wilcoxon_p", "n_batches"]
    summary = pd.DataFrame(rows).set_index("metric")[col_order]
    _report(summary, out, latex)
    return summary


def compare_new_user_analysis(csv, out: Path = None, latex: Path = None) -> pd.DataFrame:
    """
    Compare existing/new_user against overall within a single
    run_new_user_analysis.py results CSV, for every metric that has all
    three group columns ("{metric}_existing"/"{metric}_new_user"/
    "{metric}_overall"). 'overall' is the reference point both other groups
    are measured against — it's the actual blended metric across every
    user in the batch, so "how far below overall does new_user sit" is the
    meaningful question, not an arbitrary pairwise combination.

    One row per metric, with all three raw means side by side plus
    pct_improvement for existing-vs-overall and new_user-vs-overall.
    """
    df = pd.read_csv(csv)
    suffix = "_existing"
    prefixes = [
        col[: -len(suffix)] for col in df.columns
        if col.endswith(suffix)
        and f"{col[: -len(suffix)]}_new_user" in df.columns
        and f"{col[: -len(suffix)]}_overall" in df.columns
    ]
    if not prefixes:
        raise ValueError(f"No columns found with existing/new_user/overall suffixes in {csv}.")

    print(f"CSV: {csv}\n")

    rows = []
    for p in prefixes:
        existing_cmp = compare_columns(df, f"{p}_overall", "overall", df, f"{p}_existing", "existing")
        new_user_cmp = compare_columns(df, f"{p}_overall", "overall", df, f"{p}_new_user", "new_user")

        rows.append({
            "metric":         METRIC_LABELS.get(f"{p}_at_10", p),
            "mean_existing":  existing_cmp["mean_existing"],
            "mean_new_user":  new_user_cmp["mean_new_user"],
            "mean_overall":   existing_cmp["mean_overall"],
            "pct_existing":   existing_cmp["pct_improvement"],
            "pct_new_user":   new_user_cmp["pct_improvement"],
        })

    summary = pd.DataFrame(rows).set_index("metric")
    _report(summary, out, latex)
    return summary


def compare_content_coldstart(csv, out: Path = None, latex: Path = None) -> pd.DataFrame:
    """
    Compare existing/new against overall within a single
    run_content_coldstart.py results CSV ("{metric}_existing"/
    "{metric}_new_content"/"{metric}_overall_content") — mirrors
    compare_new_user_analysis exactly, just against content-init's column
    names instead of mean-init's. 'overall_content' is the reference point
    both other groups are measured against.

    One row per metric, with all three raw means side by side plus
    pct_improvement for existing-vs-overall and new-vs-overall. Restricted
    to _CONTENT_COLDSTART_METRICS (recall/precision/ndcg — same metrics
    plot_content_vs_no_update charts), not every "_existing"-suffixed
    column, since this CSV also has hr/mrr, which aren't wanted here.
    """
    df = pd.read_csv(csv)
    prefixes = [
        m for m, _, _ in _CONTENT_COLDSTART_METRICS
        if f"{m}_existing" in df.columns
        and f"{m}_new_content" in df.columns
        and f"{m}_overall_content" in df.columns
    ]
    if not prefixes:
        raise ValueError(f"No columns found with existing/new_content/overall_content suffixes in {csv}.")

    print(f"CSV: {csv}\n")

    rows = []
    for p in prefixes:
        existing_cmp = compare_columns(df, f"{p}_overall_content", "overall", df, f"{p}_existing", "existing")
        new_cmp = compare_columns(df, f"{p}_overall_content", "overall", df, f"{p}_new_content", "new")

        rows.append({
            "metric":        METRIC_LABELS.get(f"{p}_at_10", p),
            "mean_existing": existing_cmp["mean_existing"],
            "mean_new":      new_cmp["mean_new"],
            "mean_overall":  existing_cmp["mean_overall"],
            "pct_existing":  existing_cmp["pct_improvement"],
            "pct_new":       new_cmp["pct_improvement"],
        })

    summary = pd.DataFrame(rows).set_index("metric")
    _report(summary, out, latex)
    return summary


def summarize_new_user_windows(csv, update_every: int = 20, batch_size: int = 1000,
                               out: Path = None, latex: Path = None) -> pd.DataFrame:
    """
    Summarize a run_new_user_analysis.py results CSV at the 20-batch-window
    level (the same window an incremental update trains on — see
    tools/plot_utils._plot_new_user_arrivals / _plot_interaction_volume,
    which this mirrors), averaged across all windows in the run. Two rows,
    same three columns, one per counting unit:

      "users"        — mean_new/mean_total are people-counts per window.
                        Requires the "window_unique_users" column (a
                        running, cross-batch-deduplicated active-user count
                        reset every update_every batches — see
                        run_new_user_analysis.py) rather than summing the
                        per-batch n_unique_users column, which would
                        double-count users active in more than one batch
                        within the same window.
      "interactions" — mean_new/mean_total are raw event counts per
                        window. Always safe to sum across batches (an
                        interaction happens once, no dedup needed) — see
                        n_new_user_interactions in run_new_user_analysis.py.

    pct_new (both rows) is new/total per window, averaged — not computed
    from the mean_new/mean_total columns — matching how the CSV's own
    pct_new_user column is already a per-batch mean-of-ratios, not a
    ratio-of-means.
    """
    df = pd.read_csv(csv)
    required = ["n_new_users", "window_unique_users", "n_new_user_interactions"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv} is missing column(s) {missing} — re-run "
                         "run_new_user_analysis.py to generate them.")

    grouped = (
        df.assign(chunk=(df["batch"] - 1) // update_every)
          .groupby("chunk")
          .agg(n_new_users=("n_new_users", "sum"),
               unique_users=("window_unique_users", "max"),
               n_new_user_interactions=("n_new_user_interactions", "sum"),
               n_batches=("batch", "count"))
    )
    grouped["total_interactions"] = grouped["n_batches"] * batch_size
    grouped["pct_new_user"] = grouped["n_new_users"] / grouped["unique_users"] * 100
    grouped["pct_new_user_interactions"] = (
        grouped["n_new_user_interactions"] / grouped["total_interactions"] * 100
    )

    print(f"CSV:    {csv}")
    print(f"Window: {update_every} batches ({grouped['n_batches'].sum()} batches / "
         f"{len(grouped)} windows)\n")

    summary = pd.DataFrame([
        {
            "type":      "users",
            "mean_new":  grouped["n_new_users"].mean(),
            "mean_total": grouped["unique_users"].mean(),
            "pct_new":   grouped["pct_new_user"].mean(),
        },
        {
            "type":      "interactions",
            "mean_new":  grouped["n_new_user_interactions"].mean(),
            "mean_total": grouped["total_interactions"].mean(),
            "pct_new":   grouped["pct_new_user_interactions"].mean(),
        },
    ]).set_index("type")
    _report(summary, out, latex)
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("no-update-vs-incremental",
                        help="Compare two results CSVs (e.g. no_update vs incremental)")
    p1.add_argument("--baseline-csv", type=Path, required=True)
    p1.add_argument("--comparison-csv", type=Path, required=True)
    p1.add_argument("--baseline-label", type=str, default="no_update")
    p1.add_argument("--comparison-label", type=str, default="incremental")
    p1.add_argument("--metric", type=str, default=None,
                    help="Restrict to one metric column; default is every shared metric")
    p1.add_argument("--out", type=Path, default=None)
    p1.add_argument("--latex", type=Path, default=None)

    p2 = sub.add_parser("new-user-analysis",
                        help="Compare existing/new_user against overall within one "
                             "run_new_user_analysis.py CSV")
    p2.add_argument("--csv", type=Path, required=True)
    p2.add_argument("--out", type=Path, default=None)
    p2.add_argument("--latex", type=Path, default=None)

    p3 = sub.add_parser("new-user-windows",
                        help="Summarize a run_new_user_analysis.py CSV at the "
                             "20-batch-window level (people + interaction counts)")
    p3.add_argument("--csv", type=Path, required=True)
    p3.add_argument("--update-every", type=int, default=20)
    p3.add_argument("--out", type=Path, default=None)
    p3.add_argument("--latex", type=Path, default=None)

    p4 = sub.add_parser("no-update-vs-content-init",
                        help="Compare no_update's overall performance against "
                             "content-init's overall performance")
    p4.add_argument("--no-update-csv", type=Path, required=True)
    p4.add_argument("--content-csv", type=Path, required=True)
    p4.add_argument("--metric", type=str, default=None,
                    help="Restrict to one of recall/precision/ndcg; default is all three")
    p4.add_argument("--out", type=Path, default=None)
    p4.add_argument("--latex", type=Path, default=None)

    p5 = sub.add_parser("content-coldstart",
                        help="Compare existing/new against overall within one "
                             "run_content_coldstart.py CSV")
    p5.add_argument("--csv", type=Path, required=True)
    p5.add_argument("--out", type=Path, default=None)
    p5.add_argument("--latex", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "no-update-vs-incremental":
        compare_no_update_vs_incremental(args.baseline_csv, args.comparison_csv,
                                         baseline_label=args.baseline_label,
                                         comparison_label=args.comparison_label,
                                         metric=args.metric, out=args.out, latex=args.latex)
    elif args.command == "new-user-analysis":
        compare_new_user_analysis(args.csv, out=args.out, latex=args.latex)
    elif args.command == "new-user-windows":
        summarize_new_user_windows(args.csv, update_every=args.update_every,
                                   out=args.out, latex=args.latex)
    elif args.command == "no-update-vs-content-init":
        compare_no_update_vs_content_init(args.no_update_csv, args.content_csv,
                                          metric=args.metric, out=args.out, latex=args.latex)
    elif args.command == "content-coldstart":
        compare_content_coldstart(args.csv, out=args.out, latex=args.latex)


if __name__ == "__main__":
    main()
