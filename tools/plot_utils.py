"""
Shared plotting utilities for experiment scripts.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ── Shared style constants ────────────────────────────────────────────────────
COLORS = {
    "no_update":            "#e63946",
    "incremental":          "#2a9d8f",
    "lightgcn_incremental": "#2a9d8f",
    "fixed_interval":       "#2a9d8f",
    "adaptive_drift":       "#6a4c93",
    "itemknn_only":         "#e63946",
    "hybrid":               "#457b9d",
    "itemknn_static":       "#e63946",
    "lgcn_itemknn":         "#f4a261",
    "energy":               "#e63946",
    "energy_per":           "#e9c46a",
}
SMOOTH = 30   # rolling average window for recall lines
DPI    = 150
# ─────────────────────────────────────────────────────────────────────────────


def style_ax(ax, xlabel=None, ylabel=None, title=None):
    """Apply common axis styling."""
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)


METRIC_LABELS = {
    "recall_at_10":    "Recall@10",
    "precision_at_10": "Precision@10",
    "ndcg_at_10":      "NDCG@10",
    "hr_at_10":        "HR@10",
    "recall_at_20":    "Recall@20",
    "precision_at_20": "Precision@20",
    "ndcg_at_20":      "NDCG@20",
    "hr_at_20":        "HR@20",
    "mrr":             "MRR",
}

STRATEGY_COLORS = COLORS  # alias for clarity outside this module


def plot_metric_over_time(ax, df: pd.DataFrame, update_every: int,
                          metric: str = "recall_at_10"):
    """
    Plot any metric over time for no_update and incremental strategies.
    Faint raw lines + bold smoothed trend. Dashed lines mark update moments.
    """
    ylabel = METRIC_LABELS.get(metric, metric)
    strategies = df["strategy"].unique()

    # Support both 2-strategy (no_update/incremental) and multi-strategy (hybrid serving)
    label_map = {
        "no_update":            "No update (model goes stale)",
        "incremental":          f"Incremental update (every {update_every} batches)",
        "lightgcn_incremental": "LightGCN incremental",
        "fixed_interval":       f"Fixed interval (every {update_every} batches)",
        "adaptive_drift":       "Adaptive drift trigger (cosine similarity)",
        "itemknn_only":         "ItemKNN only (no update)",
        "itemknn_static":       "ItemKNN static (co-occurrence)",
        "hybrid":               "Hybrid (ItemKNN + LightGCN blend)",
        "lgcn_itemknn":         "LightGCN-ItemKNN (embedding similarity)",
    }

    for strategy, grp in df.groupby("strategy"):
        color = COLORS.get(strategy, None)
        label = label_map.get(strategy, strategy)
        smoothed = grp[metric].rolling(SMOOTH, min_periods=1, center=True).mean()
        ax.plot(grp["interactions"], grp[metric],
                color=color, alpha=0.15, linewidth=0.8)
        ax.plot(grp["interactions"], smoothed,
                color=color, label=label, linewidth=2.0)

    updates = df[df.get("updated", pd.Series(False, index=df.index)) == True]
    inc_updates = updates[updates["strategy"].isin(["incremental", "lightgcn_incremental", "hybrid"])]
    for x in inc_updates["interactions"]:
        ax.axvline(x, color=COLORS["incremental"], alpha=0.25, linewidth=0.8, linestyle="--")
    if len(inc_updates):
        ax.axvline(inc_updates["interactions"].iloc[0], color=COLORS["incremental"],
                   alpha=0.25, linewidth=0.8, linestyle="--", label="Update triggered")

    max_x = int(df["interactions"].max())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.005))
    ax.grid(which="minor", alpha=0.15)
    style_ax(ax, ylabel=ylabel, title=f"{ylabel} Over Time")
    ax.set_xlim(left=0, right=max_x)
    filtered = [t for t in ax.get_xticks() if 0 <= t <= max_x]
    ax.set_xticks(sorted(set(filtered + [max_x])))
    ax.legend(loc="upper left", frameon=True)


def plot_recall_over_time(ax, df: pd.DataFrame, update_every: int):
    """Backwards-compatible wrapper — plots recall_at_10."""
    plot_metric_over_time(ax, df, update_every, metric="recall_at_10")


def plot_energy_bars(ax, df: pd.DataFrame, bar_width: int = 800,
                     training_energy_uwh: float = None):
    """Plot energy cost per update as bars (incremental + full_retrain,
    side by side where both occur), plus a reference line for the
    one-time historical training energy (if provided)."""
    inc_updates = df[(df["strategy"] == "incremental") & (df["updated"])]
    retrain_updates = df[(df["strategy"] == "full_retrain") & (df["updated"])]

    has_both = len(inc_updates) and len(retrain_updates)
    offset = bar_width * 0.55 if has_both else 0
    if len(inc_updates):
        ax.bar(inc_updates["interactions"] - offset, inc_updates["update_energy_uwh"],
               width=bar_width, color=COLORS["incremental"], alpha=0.7,
               label="Incremental update energy")
    if len(retrain_updates):
        ax.bar(retrain_updates["interactions"] + offset, retrain_updates["update_energy_uwh"],
               width=bar_width, color=COLORS["hybrid"], alpha=0.7,
               label="Full retrain energy")
    if training_energy_uwh is not None:
        ax.axhline(training_energy_uwh, color=COLORS["no_update"],
                   linewidth=1.5, linestyle="--",
                   label=f"Historical training energy (one-time, {training_energy_uwh:.2f} µWh)")
    style_ax(ax,
             xlabel="Interactions seen (real-time stream)",
             ylabel="Update energy (µWh)",
             title="Energy Cost per Update")


def plot_streaming_results(df: pd.DataFrame, out_path: Path,
                           title: str, update_every: int,
                           training_energy_uwh: float = None):
    """
    One PNG per available metric + one energy PNG.
    out_path is used as base — metric name suffix added per file.
    """
    base = Path(str(out_path).replace(".png", ""))

    # One plot per metric column present in df
    available = [m for m in METRIC_LABELS if m in df.columns]
    for metric in available:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(title, fontsize=13)
        plot_metric_over_time(ax, df, update_every, metric=metric)
        plt.tight_layout()
        path = Path(f"{base}_{metric}.png")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()

    # Energy plot
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(title, fontsize=13)
    plot_energy_bars(ax, df, training_energy_uwh=training_energy_uwh)
    plt.tight_layout()
    energy_path = Path(f"{base}_energy.png")
    plt.savefig(energy_path, dpi=DPI)
    print(f"Plot saved → {energy_path}")
    plt.close()


def plot_update_freq_recall_curves(df: pd.DataFrame, out_path: Path, title: str):
    """
    Smoothed Recall@10 over time for each UPDATE_EVERY value.
    update_every=0 is the no_update baseline, shown as a dashed red line.
    Fixed BATCH_SIZE means all curves share the same x-axis density.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.viridis

    # No-update baseline (update_every=0)
    if 0 in df["update_every"].values:
        grp = df[df["update_every"] == 0]
        smoothed = grp["recall_at_10"].rolling(SMOOTH, min_periods=1, center=True).mean()
        ax.plot(grp["interactions"], grp["recall_at_10"],
                color=COLORS["no_update"], alpha=0.15, linewidth=0.8)
        ax.plot(grp["interactions"], smoothed,
                color=COLORS["no_update"], linewidth=2.0, linestyle="--",
                label="No update (baseline)")

    incremental_values = sorted(v for v in df["update_every"].unique() if v != 0)
    colors = [cmap(i / max(len(incremental_values) - 1, 1))
              for i in range(len(incremental_values))]

    for ue, color in zip(incremental_values, colors):
        grp = df[df["update_every"] == ue]
        raw = grp["recall_at_10"]
        smoothed = raw.rolling(SMOOTH, min_periods=1, center=True).mean()
        ax.plot(grp["interactions"], raw, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(grp["interactions"], smoothed, color=color, linewidth=2.0,
                label=f"update_every={ue} ({ue*1000:,} inter/upd)")

    style_ax(ax, xlabel="Interactions seen (real-time stream)",
             ylabel="Recall@10", title=title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    print(f"Plot saved → {out_path}")
    plt.close()


def plot_update_freq_ablation(summary: pd.DataFrame, out_path: Path, title: str):
    """
    Three-panel plot: quality / total energy / energy-per-update vs interactions_per_update.
    Used by find_optimal_batch_size.py.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13)

    x_labels = summary["interactions_per_update"].astype(int).astype(str)

    axes[0].plot(summary["interactions_per_update"], summary["avg_recall"],
                 marker="o", color=COLORS["incremental"], linewidth=2)
    style_ax(axes[0], xlabel="Interactions per update", ylabel="Avg Recall@10",
             title="Quality vs Update Frequency")

    axes[1].bar(x_labels, summary["total_energy"], color=COLORS["energy"], alpha=0.7)
    style_ax(axes[1], xlabel="Interactions per update",
             ylabel="Total update energy (µWh)", title="Total Energy vs Update Frequency")

    axes[2].bar(x_labels, summary["energy_per_update"],
                color=COLORS["energy_per"], alpha=0.7)
    style_ax(axes[2], xlabel="Interactions per update",
             ylabel="Avg energy per update (µWh)", title="Energy per Update vs Update Frequency")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    print(f"Plot saved → {out_path}")
    plt.close()
