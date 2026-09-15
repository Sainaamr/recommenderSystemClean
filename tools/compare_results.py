"""
usage:
python tools/compare_results.py <subcommand> <inputs> [--out X.csv] [--latex X.tex]
| Subcommand                  | Inputs                                                                                                                                   
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------  
| no-update-vs-incremental    | --baseline-csv --comparison-csv · optional --baseline-label --comparison-label --metric                                                  
| new-user-analysis           | --csv                                                                                                                                    
| new-user-windows            | --csv · optional --update-every (default 20)                                                                                             
| no-update-vs-content-init   | --no-update-csv --content-csv · optional --metric                                                                                        
| content-coldstart           | --csv                                                                                                                                    
| content-incremental-vs-all  | --content-incremental-csv --no-update-csv --incremental-csv --content-csv · optional --metric                                            
| content-init-vs-incremental | --content-csv --incremental-csv · optional --metric                                                                                      
| energy-summary              | four pairs: --{no-update,incremental,content-coldstart,content-incremental}-csv plus a matching -results-csv for each                     
| energy-summary-compact      | same eight as above                                                                                                                      
| energy-summary-hybrid       | --no-update-csv --no-update-results-csv --incremental-csv --incremental-results-csv --full-retrain-training-json --full-retrain-results-csv 
| full-retrain-vs-all         | --full-retrain-csv --no-update-csv --incremental-csv · optional --metric                                                                 
| frequency-vs-baseline       | --csv-dir · optional --baseline-update-every (default 270)                                                                               
| update-cost                 | --csv label=path, repeatable · optional --training-emissions-mg                                                                          
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from tools.plot_utils import (METRIC_LABELS, _CONTENT_COLDSTART_METRICS, plot_energy_summary_stacked,
                              _discover_frequency_runs, _frequency_run_energies)


def compare_columns(df_a: pd.DataFrame, col_a: str, label_a: str,
                    df_b: pd.DataFrame, col_b: str, label_b: str) -> dict:

    left  = df_a[["batch", col_a]].rename(columns={col_a: "_a"})
    right = df_b[["batch", col_b]].rename(columns={col_b: "_b"})
    merged = pd.merge(left, right, on="batch")
    a = merged["_a"].to_numpy()
    b = merged["_b"].to_numpy()

    mean_gap = b.mean() - a.mean()
    pct_improvement = (mean_gap / a.mean() * 100) if a.mean() != 0 else float("nan")
    win_rate = (b > a).mean()

    # Wilcoxon on the paired differences; raises only if every diff is
    # exactly zero (won't happen with real noisy metrics).
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
        elif col.endswith("_mg"):
            df[col] = df[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "--")
        elif col == "n_updates":
            df[col] = df[col].map(lambda x: f"{int(x)}")
    return df


def _report(summary: pd.DataFrame, out: Path = None, latex: Path = None):
    """Prints summary; saves to `out` (CSV) and/or `latex` (.tex) if given."""
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


def compare_content_init_vs_incremental(content_csv, incremental_csv,
                                        metric: str = None,
                                        out: Path = None, latex: Path = None) -> pd.DataFrame:
  
    df_a = pd.read_csv(incremental_csv)
    df_b = pd.read_csv(content_csv)

    metrics = [metric] if metric else ["recall", "precision", "ndcg"]

    print(f"incremental (baseline):    {incremental_csv}")
    print(f"content_init (comparison): {content_csv}\n")

    rows = []
    for m in metrics:
        row = compare_columns(df_a, f"{m}_at_10", "incremental",
                              df_b, f"{m}_overall_content", "content_init")
        row["metric"] = METRIC_LABELS.get(f"{m}_at_10", m)
        rows.append(row)

    col_order = ["mean_incremental", "mean_content_init", "pct_improvement", "win_rate", "wilcoxon_p"]
    summary = pd.DataFrame(rows).set_index("metric")[col_order]
    _report(summary, out, latex)
    return summary


def compare_content_incremental_vs_all(content_incremental_csv, no_update_csv,
                                       incremental_csv, content_csv,
                                       metric: str = None,
                                       out: Path = None, latex: Path = None) -> pd.DataFrame:
  
    df_ci = pd.read_csv(content_incremental_csv)
    df_no_update = pd.read_csv(no_update_csv)
    df_incremental = pd.read_csv(incremental_csv)
    df_content = pd.read_csv(content_csv)

    metrics = [metric] if metric else ["recall", "precision", "ndcg"]

    print(f"content_incremental: {content_incremental_csv}")
    print(f"no_update:           {no_update_csv}")
    print(f"incremental:         {incremental_csv}")
    print(f"content:             {content_csv}\n")

    rows = []
    for m in metrics:
        vs_no_update = compare_columns(df_no_update, f"{m}_at_10", "no_update",
                                       df_ci, f"{m}_overall_content", "content_incremental")
        vs_incremental = compare_columns(df_incremental, f"{m}_at_10", "incremental",
                                         df_ci, f"{m}_overall_content", "content_incremental")
        vs_content = compare_columns(df_content, f"{m}_overall_content", "content",
                                     df_ci, f"{m}_overall_content", "content_incremental")

        rows.append({
            "metric":                   METRIC_LABELS.get(f"{m}_at_10", m),
            "mean_content_incremental": vs_no_update["mean_content_incremental"],
            "pct_vs_no_update":         vs_no_update["pct_improvement"],
            "pct_vs_incremental":       vs_incremental["pct_improvement"],
            "pct_vs_content":           vs_content["pct_improvement"],
        })

    summary = pd.DataFrame(rows).set_index("metric")
    _report(summary, out, latex)
    return summary


def compare_full_retrain_vs_all(full_retrain_csv, no_update_csv, incremental_csv,
                                metric: str = None,
                                out: Path = None, latex: Path = None) -> pd.DataFrame:

    df_fr = pd.read_csv(full_retrain_csv)
    df_no_update = pd.read_csv(no_update_csv)
    df_incremental = pd.read_csv(incremental_csv)

    metrics = [metric] if metric else [
        m for m in METRIC_LABELS
        if m in df_fr.columns and m in df_no_update.columns and m in df_incremental.columns
    ]

    print(f"full_retrain: {full_retrain_csv}")
    print(f"no_update:    {no_update_csv}")
    print(f"incremental:  {incremental_csv}\n")

    rows = []
    for m in metrics:
        vs_no_update = compare_columns(df_no_update, m, "no_update", df_fr, m, "full_retrain")
        vs_incremental = compare_columns(df_incremental, m, "incremental", df_fr, m, "full_retrain")

        rows.append({
            "metric":             METRIC_LABELS.get(m, m),
            "mean_full_retrain":  vs_no_update["mean_full_retrain"],
            "pct_vs_no_update":   vs_no_update["pct_improvement"],
            "pct_vs_incremental": vs_incremental["pct_improvement"],
        })

    summary = pd.DataFrame(rows).set_index("metric")
    _report(summary, out, latex)
    return summary


def compare_frequency_sweep_vs_baseline(csv_dir, baseline_update_every: int = 270,
                                        out: Path = None, latex: Path = None) -> pd.DataFrame:
 
    entries = _discover_frequency_runs(csv_dir)
    energies = _frequency_run_energies(entries)
    valid = {ue: (df, e) for (ue, df, _), e in zip(entries, energies) if e is not None}

    if baseline_update_every not in valid:
        raise ValueError(f"baseline_update_every={baseline_update_every} not found among "
                         f"discovered runs: {sorted(valid, reverse=True)}")

    df_base, e_base = valid[baseline_update_every]
    metrics = ["recall", "precision", "ndcg"]

    print(f"csv_dir:  {csv_dir}")
    print(f"baseline: update every {baseline_update_every} batches\n")

    rows = []
    for ue in sorted(valid, reverse=True):
        df_ue, e_ue = valid[ue]
        row = {"update_every": ue}
        for m in metrics:
            cmp = compare_columns(df_base, f"{m}_overall_content", "baseline",
                                  df_ue, f"{m}_overall_content", "run")
            row[f"pct_{m}"] = cmp["pct_improvement"]
        row["pct_energy"] = (e_ue - e_base) / e_base * 100 if e_base else float("nan")
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("update_every")
    _report(summary, out, latex)
    return summary


def compare_update_cost(strategy_csvs: dict, training_emissions_mg: float = None,
                        out: Path = None, latex: Path = None) -> pd.DataFrame:
  
    rows = []
    for label, csv_path in strategy_csvs.items():
        df = pd.read_csv(csv_path)
        n_updates = int(df["updated"].sum())
        total = df["update_emissions_mg"].sum()
        row = {
            "strategy":           label,
            "n_updates":          n_updates,
            "mean_per_update_mg": total / max(n_updates, 1),
            "total_mg":           total,
        }
        if training_emissions_mg is not None:
            row["total_with_training_mg"] = training_emissions_mg + total
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("strategy")
    _report(summary, out, latex)
    return summary


def compare_new_user_analysis(csv, out: Path = None, latex: Path = None) -> pd.DataFrame:
    
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
  
    df = pd.read_csv(csv)
    required = ["n_new_users", "window_unique_users",
                "n_first_time_new_user_interactions"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv} is missing column(s) {missing} — re-run "
                         "run_new_user_analysis.py to generate them.")

    grouped = (
        df.assign(chunk=(df["batch"] - 1) // update_every)
          .groupby("chunk")
          .agg(n_new_users=("n_new_users", "sum"),
               unique_users=("window_unique_users", "max"),
               n_new_user_interactions=("n_first_time_new_user_interactions", "sum"),
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



_ENERGY_SUMMARY_SECTIONS = [
    "training_emissions_mg",
    "id_mapping_emissions_mg",
    "checkpoint_load_emissions_mg",
    "build_user_history_emissions_mg",
    "embedding_snapshot_emissions_mg",
    "content_build_emissions_mg",
    "id_resolution_emissions_mg",
    "recovered_history_seed_emissions_mg",
    "expand_embeddings_emissions_mg",
    "gt_split_emissions_mg",
    "scoring_emissions_mg",
    "update_prep_emissions_mg",
    "history_update_emissions_mg",
    "apply_content_seeds_emissions_mg",
    "batch_total_emissions_mg",
    "update_total_emissions_mg",
    "n_updates",
    "avg_emissions_per_update_mg",
    "streaming_emissions_mg",
    "grand_total_mg",
]

_HYBRID_BATCH_SUBTASKS = [
    "id_resolution_emissions_mg", "expand_embeddings_emissions_mg",
    "scoring_emissions_mg", "update_prep_emissions_mg", "history_update_emissions_mg",
]
_CONTENT_BATCH_SUBTASKS = [
    "id_resolution_emissions_mg", "recovered_history_seed_emissions_mg",
    "expand_embeddings_emissions_mg", "gt_split_emissions_mg", "scoring_emissions_mg",
    "history_update_emissions_mg", "apply_content_seeds_emissions_mg",
]


def _energy_from_hybrid(csv_path, results_csv_path, strategy):
    row = pd.read_csv(csv_path).iloc[0]
    results = pd.read_csv(results_csv_path)
    section_row = {
        "training_emissions_mg":           row.get("training_emissions_mg", 0.0),
        "id_mapping_emissions_mg":          row.get(f"{strategy}_id_mapping_emissions_mg", float("nan")),
        "checkpoint_load_emissions_mg":     row.get(f"{strategy}_checkpoint_load_emissions_mg", float("nan")),
        "build_user_history_emissions_mg":  row.get(f"{strategy}_build_user_history_emissions_mg", float("nan")),
        "embedding_snapshot_emissions_mg":  float("nan"),
        "content_build_emissions_mg":       float("nan"),
        "recovered_history_seed_emissions_mg": float("nan"),
        "gt_split_emissions_mg":            float("nan"),
        "apply_content_seeds_emissions_mg": float("nan"),
        "batch_total_emissions_mg":         row.get(f"{strategy}_total_batch_emissions_mg", float("nan")),
        "update_total_emissions_mg":        row.get(f"{strategy}_total_update_emissions_mg", 0.0),
        "n_updates":                        int(row.get(f"{strategy}_n_updates", 0)),
        "avg_emissions_per_update_mg":      row.get(f"{strategy}_avg_emissions_per_update_mg", 0.0),
        "streaming_emissions_mg":           row.get(f"{strategy}_streaming_emissions_mg", float("nan")),
    }
    for col in _HYBRID_BATCH_SUBTASKS:
        section_row[col] = results[col].sum() if col in results.columns else float("nan")
    return section_row


def _energy_from_content(csv_path, results_csv_path):
    row = pd.read_csv(csv_path).iloc[0]
    results = pd.read_csv(results_csv_path)
    section_row = {
        "training_emissions_mg":           row.get("training_emissions_mg", 0.0),
        "id_mapping_emissions_mg":          row.get("id_mapping_emissions_mg", float("nan")),
        "checkpoint_load_emissions_mg":     row.get("checkpoint_load_emissions_mg", float("nan")),
        "build_user_history_emissions_mg":  row.get("build_user_history_emissions_mg", float("nan")),
        "embedding_snapshot_emissions_mg":  row.get("embedding_snapshot_emissions_mg", float("nan")),
        "content_build_emissions_mg":       row.get("content_build_emissions_mg", float("nan")),
        "update_prep_emissions_mg":         float("nan"),
        "batch_total_emissions_mg":         row.get("total_batch_emissions_mg", float("nan")),
        "update_total_emissions_mg":        row.get("total_update_emissions_mg", 0.0),
        "n_updates":                        int(row.get("n_updates", 0)),
        "avg_emissions_per_update_mg":      row.get("avg_emissions_per_update_mg", 0.0),
        "streaming_emissions_mg":           row.get("streaming_emissions_mg", float("nan")),
    }
    for col in _CONTENT_BATCH_SUBTASKS:
        section_row[col] = results[col].sum() if col in results.columns else float("nan")
    return section_row


def _energy_from_legacy_full_retrain(training_json_path, results_csv_path) -> dict:

    training_emissions_mg = json.loads(Path(training_json_path).read_text())["training_emissions_mg"]
    results = pd.read_csv(results_csv_path)
    n_updates = int(results["updated"].sum())
    update_total = results["update_emissions_mg"].sum()
    return {
        "training_emissions_mg":               training_emissions_mg,
        "id_mapping_emissions_mg":              float("nan"),
        "checkpoint_load_emissions_mg":         float("nan"),
        "build_user_history_emissions_mg":      float("nan"),
        "embedding_snapshot_emissions_mg":      float("nan"),
        "content_build_emissions_mg":           float("nan"),
        "id_resolution_emissions_mg":           float("nan"),
        "recovered_history_seed_emissions_mg":  float("nan"),
        "expand_embeddings_emissions_mg":       float("nan"),
        "gt_split_emissions_mg":                float("nan"),
        "scoring_emissions_mg":                 float("nan"),
        "update_prep_emissions_mg":             float("nan"),
        "history_update_emissions_mg":          float("nan"),
        "apply_content_seeds_emissions_mg":     float("nan"),
        "batch_total_emissions_mg":             float("nan"),
        "update_total_emissions_mg":            update_total,
        "n_updates":                            n_updates,
        "avg_emissions_per_update_mg":          update_total / max(n_updates, 1),
        "streaming_emissions_mg":               float("nan"),
    }


def compare_energy_summary_hybrid(no_update_csv, no_update_results_csv,
                                  incremental_csv, incremental_results_csv,
                                  full_retrain_training_json, full_retrain_results_csv,
                                  out: Path = None, latex: Path = None, plot: Path = None) -> pd.DataFrame:

    print(f"no_update:    {no_update_csv} / {no_update_results_csv}")
    print(f"incremental:  {incremental_csv} / {incremental_results_csv}")
    print(f"full_retrain: {full_retrain_training_json} / {full_retrain_results_csv}\n")

    columns = {
        "no_update":    _energy_from_hybrid(no_update_csv, no_update_results_csv, "no_update"),
        "incremental":  _energy_from_hybrid(incremental_csv, incremental_results_csv, "incremental"),
        "full_retrain": _energy_from_legacy_full_retrain(full_retrain_training_json, full_retrain_results_csv),
    }
    summary = pd.DataFrame(columns).reindex(_ENERGY_SUMMARY_SECTIONS[:-1])
    summary.loc["grand_total_mg"] = summary.loc["training_emissions_mg"] + summary.loc["streaming_emissions_mg"]
    summary.index.name = "section"

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(summary.to_string())
    if out:
        summary.to_csv(out)
        print(f"\nSummary saved → {out}")
    if latex:
        formatted = summary.astype(object)
        for section in formatted.index:
            if section == "n_updates":
                formatted.loc[section] = formatted.loc[section].map(lambda x: f"{int(x)}")
            else:
                formatted.loc[section] = formatted.loc[section].map(
                    lambda x: f"{x:,.0f}" if pd.notna(x) else "--")
        Path(latex).write_text(formatted.reset_index().to_latex(index=False))
        print(f"LaTeX table saved → {latex}")
    if plot:
        plot_energy_summary_stacked(summary, plot, "Energy Breakdown by Run")
    return summary


def _energy_summary_columns(no_update_csv, no_update_results_csv,
                            incremental_csv, incremental_results_csv,
                            content_coldstart_csv, content_coldstart_results_csv,
                            content_incremental_csv, content_incremental_results_csv) -> dict:

    print(f"no_update:           {no_update_csv} / {no_update_results_csv}")
    print(f"incremental:         {incremental_csv} / {incremental_results_csv}")
    print(f"content_coldstart:   {content_coldstart_csv} / {content_coldstart_results_csv}")
    print(f"content_incremental: {content_incremental_csv} / {content_incremental_results_csv}\n")

    return {
        "no_update":           _energy_from_hybrid(no_update_csv, no_update_results_csv, "no_update"),
        "incremental":         _energy_from_hybrid(incremental_csv, incremental_results_csv, "incremental"),
        "content_coldstart":   _energy_from_content(content_coldstart_csv, content_coldstart_results_csv),
        "content_incremental": _energy_from_content(content_incremental_csv, content_incremental_results_csv),
    }


def compare_energy_summary(no_update_csv, no_update_results_csv,
                           incremental_csv, incremental_results_csv,
                           content_coldstart_csv, content_coldstart_results_csv,
                           content_incremental_csv, content_incremental_results_csv,
                           out: Path = None, latex: Path = None, plot: Path = None) -> pd.DataFrame:

    columns = _energy_summary_columns(no_update_csv, no_update_results_csv,
                                      incremental_csv, incremental_results_csv,
                                      content_coldstart_csv, content_coldstart_results_csv,
                                      content_incremental_csv, content_incremental_results_csv)
    summary = pd.DataFrame(columns).reindex(_ENERGY_SUMMARY_SECTIONS[:-1])
    summary.loc["grand_total_mg"] = summary.loc["training_emissions_mg"] + summary.loc["streaming_emissions_mg"]
    summary.index.name = "section"

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(summary.to_string())
    if out:
        summary.to_csv(out)
        print(f"\nSummary saved → {out}")
    if latex:
        formatted = summary.astype(object)
        for section in formatted.index:
            if section == "n_updates":
                formatted.loc[section] = formatted.loc[section].map(lambda x: f"{int(x)}")
            else:
                formatted.loc[section] = formatted.loc[section].map(
                    lambda x: f"{x:,.0f}" if pd.notna(x) else "--")
        Path(latex).write_text(formatted.reset_index().to_latex(index=False))
        print(f"LaTeX table saved → {latex}")
    if plot:
        plot_energy_summary_stacked(summary, plot, "Energy Breakdown by Run")
    return summary


def compare_energy_summary_compact(no_update_csv, no_update_results_csv,
                                   incremental_csv, incremental_results_csv,
                                   content_coldstart_csv, content_coldstart_results_csv,
                                   content_incremental_csv, content_incremental_results_csv,
                                   out: Path = None, latex: Path = None) -> pd.DataFrame:
 
    columns = _energy_summary_columns(no_update_csv, no_update_results_csv,
                                      incremental_csv, incremental_results_csv,
                                      content_coldstart_csv, content_coldstart_results_csv,
                                      content_incremental_csv, content_incremental_results_csv)
    runs = pd.DataFrame(columns).T
    summary = pd.DataFrame({
        "streaming_emissions_mg":         runs["streaming_emissions_mg"],
        "content_init_emissions_mg":      runs["content_build_emissions_mg"]
                                           + runs["recovered_history_seed_emissions_mg"]
                                           + runs["apply_content_seeds_emissions_mg"],
        "update_total_emissions_mg":      runs["update_total_emissions_mg"],
        "expand_embeddings_emissions_mg": runs["expand_embeddings_emissions_mg"],
    })
    summary.index.name = "run"

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(summary.to_string())
    if out:
        summary.to_csv(out)
        print(f"\nSummary saved → {out}")
    if latex:
        formatted = summary.astype(object)
        for col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "--")
        Path(latex).write_text(formatted.reset_index().to_latex(index=False))
        print(f"LaTeX table saved → {latex}")
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

    p6 = sub.add_parser("content-incremental-vs-all",
                        help="Compare content_incremental against no_update, "
                             "incremental, and content-init")
    p6.add_argument("--content-incremental-csv", type=Path, required=True)
    p6.add_argument("--no-update-csv", type=Path, required=True)
    p6.add_argument("--incremental-csv", type=Path, required=True)
    p6.add_argument("--content-csv", type=Path, required=True)
    p6.add_argument("--metric", type=str, default=None,
                    help="Restrict to one of recall/precision/ndcg; default is all three")
    p6.add_argument("--out", type=Path, default=None)
    p6.add_argument("--latex", type=Path, default=None)

    p7 = sub.add_parser("content-init-vs-incremental",
                        help="Compare content-init's overall performance against "
                             "incremental update's overall performance")
    p7.add_argument("--content-csv", type=Path, required=True)
    p7.add_argument("--incremental-csv", type=Path, required=True)
    p7.add_argument("--metric", type=str, default=None,
                    help="Restrict to one of recall/precision/ndcg; default is all three")
    p7.add_argument("--out", type=Path, default=None)
    p7.add_argument("--latex", type=Path, default=None)

    p8 = sub.add_parser("energy-summary",
                        help="Compare energy/emissions across no_update, incremental, "
                             "content_coldstart, and content_incremental, including the "
                             "full per-batch sub-task breakdown (scoring, id_resolution, etc.)")
    p8.add_argument("--no-update-csv", type=Path, required=True,
                    help="A run_incremental_lightgcn.py *_hybrid_emissions_summary CSV "
                         "(no_update run)")
    p8.add_argument("--no-update-results-csv", type=Path, required=True,
                    help="The matching *_hybrid_results_no_update_*.csv (per-batch)")
    p8.add_argument("--incremental-csv", type=Path, required=True,
                    help="A run_incremental_lightgcn.py *_hybrid_emissions_summary CSV "
                         "(incremental run)")
    p8.add_argument("--incremental-results-csv", type=Path, required=True,
                    help="The matching *_hybrid_results_incremental_*.csv (per-batch)")
    p8.add_argument("--content-coldstart-csv", type=Path, required=True,
                    help="A run_content_coldstart.py *_energy.csv sidecar")
    p8.add_argument("--content-coldstart-results-csv", type=Path, required=True,
                    help="The matching *_content_coldstart_*.csv (per-batch, without _energy suffix)")
    p8.add_argument("--content-incremental-csv", type=Path, required=True,
                    help="A run_content_incremental.py *_energy.csv sidecar")
    p8.add_argument("--content-incremental-results-csv", type=Path, required=True,
                    help="The matching *_content_incremental_*.csv (per-batch, without _energy suffix)")
    p8.add_argument("--out", type=Path, default=None)
    p8.add_argument("--latex", type=Path, default=None)
    p8.add_argument("--plot", type=Path, default=None,
                    help="Save a stacked-bar-chart PNG of the full breakdown "
                         "(one bar per run, streaming only, training excluded)")

    p8b = sub.add_parser("energy-summary-compact",
                         help="Compact, thesis-table-sized energy summary — one row per "
                              "run: streaming, content_build, seeding, update_total, "
                              "expand_embeddings. Same inputs as energy-summary.")
    p8b.add_argument("--no-update-csv", type=Path, required=True)
    p8b.add_argument("--no-update-results-csv", type=Path, required=True)
    p8b.add_argument("--incremental-csv", type=Path, required=True)
    p8b.add_argument("--incremental-results-csv", type=Path, required=True)
    p8b.add_argument("--content-coldstart-csv", type=Path, required=True)
    p8b.add_argument("--content-coldstart-results-csv", type=Path, required=True)
    p8b.add_argument("--content-incremental-csv", type=Path, required=True)
    p8b.add_argument("--content-incremental-results-csv", type=Path, required=True)
    p8b.add_argument("--out", type=Path, default=None)
    p8b.add_argument("--latex", type=Path, default=None)

    p8c = sub.add_parser("energy-summary-hybrid",
                         help="3-way hybrid-only energy comparison — no_update/incremental/"
                              "full_retrain, no content mechanism required (e.g. MovieLens)")
    p8c.add_argument("--no-update-csv", type=Path, required=True)
    p8c.add_argument("--no-update-results-csv", type=Path, required=True)
    p8c.add_argument("--incremental-csv", type=Path, required=True)
    p8c.add_argument("--incremental-results-csv", type=Path, required=True)
    p8c.add_argument("--full-retrain-training-json", type=Path, required=True,
                     help="Legacy training-only JSON sidecar for full_retrain "
                          "(no section breakdown available)")
    p8c.add_argument("--full-retrain-results-csv", type=Path, required=True)
    p8c.add_argument("--out", type=Path, default=None)
    p8c.add_argument("--latex", type=Path, default=None)
    p8c.add_argument("--plot", type=Path, default=None)

    p9 = sub.add_parser("full-retrain-vs-all",
                        help="Compare full_retrain against no_update and incremental "
                             "(e.g. the MovieLens 3-strategy comparison)")
    p9.add_argument("--full-retrain-csv", type=Path, required=True)
    p9.add_argument("--no-update-csv", type=Path, required=True)
    p9.add_argument("--incremental-csv", type=Path, required=True)
    p9.add_argument("--metric", type=str, default=None,
                    help="Restrict to one metric column; default is every shared metric")
    p9.add_argument("--out", type=Path, default=None)
    p9.add_argument("--latex", type=Path, default=None)

    p9b = sub.add_parser("frequency-vs-baseline",
                         help="Compare every content_incremental update-frequency run in a "
                              "directory against one baseline frequency (default: 270)")
    p9b.add_argument("--csv-dir", type=Path, required=True,
                     help="Directory of content_incremental run CSVs + *_energy.csv sidecars "
                          "(e.g. results/frequency_change/)")
    p9b.add_argument("--baseline-update-every", type=int, default=270)
    p9b.add_argument("--out", type=Path, default=None)
    p9b.add_argument("--latex", type=Path, default=None)

    p10 = sub.add_parser("update-cost",
                         help="Summarize update_emissions_mg per strategy — n_updates, "
                              "mean per update, total (e.g. full_retrain vs incremental)")
    p10.add_argument("--csv", type=str, action="append", required=True,
                     help="label=path, repeatable, e.g. --csv full_retrain=results/x.csv "
                          "--csv incremental=results/y.csv")
    p10.add_argument("--training-emissions-mg", type=float, default=None,
                     help="Shared one-time training cost — adds a total_with_training_mg column")
    p10.add_argument("--out", type=Path, default=None)
    p10.add_argument("--latex", type=Path, default=None)

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
    elif args.command == "content-incremental-vs-all":
        compare_content_incremental_vs_all(args.content_incremental_csv, args.no_update_csv,
                                           args.incremental_csv, args.content_csv,
                                           metric=args.metric, out=args.out, latex=args.latex)
    elif args.command == "content-init-vs-incremental":
        compare_content_init_vs_incremental(args.content_csv, args.incremental_csv,
                                            metric=args.metric, out=args.out, latex=args.latex)
    elif args.command == "energy-summary":
        compare_energy_summary(args.no_update_csv, args.no_update_results_csv,
                               args.incremental_csv, args.incremental_results_csv,
                               args.content_coldstart_csv, args.content_coldstart_results_csv,
                               args.content_incremental_csv, args.content_incremental_results_csv,
                               out=args.out, latex=args.latex, plot=args.plot)
    elif args.command == "energy-summary-compact":
        compare_energy_summary_compact(args.no_update_csv, args.no_update_results_csv,
                               args.incremental_csv, args.incremental_results_csv,
                               args.content_coldstart_csv, args.content_coldstart_results_csv,
                               args.content_incremental_csv, args.content_incremental_results_csv,
                               out=args.out, latex=args.latex)
    elif args.command == "energy-summary-hybrid":
        compare_energy_summary_hybrid(args.no_update_csv, args.no_update_results_csv,
                               args.incremental_csv, args.incremental_results_csv,
                               args.full_retrain_training_json, args.full_retrain_results_csv,
                               out=args.out, latex=args.latex, plot=args.plot)
    elif args.command == "full-retrain-vs-all":
        compare_full_retrain_vs_all(args.full_retrain_csv, args.no_update_csv, args.incremental_csv,
                                    metric=args.metric, out=args.out, latex=args.latex)
    elif args.command == "frequency-vs-baseline":
        compare_frequency_sweep_vs_baseline(args.csv_dir,
                                            baseline_update_every=args.baseline_update_every,
                                            out=args.out, latex=args.latex)
    elif args.command == "update-cost":
        strategy_csvs = dict(item.split("=", 1) for item in args.csv)
        compare_update_cost(strategy_csvs, training_emissions_mg=args.training_emissions_mg,
                            out=args.out, latex=args.latex)


if __name__ == "__main__":
    main()
