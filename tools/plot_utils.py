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
    "full_retrain":         "#e7ab51",
    "lightgcn_incremental": "#2a9d8f",
    "fixed_interval":       "#2a9d8f",
    "adaptive_drift":       "#6a4c93",
    "itemknn_only":         "#e63946",
    "hybrid":               "#457b9d",
    "itemknn_static":       "#e63946",
    "lgcn_itemknn":         "#f4a261",
}
SMOOTH = 30   # rolling average window for recall lines
DPI    = 150
# ─────────────────────────────────────────────────────────────────────────────


def style_ax(ax, xlabel=None, ylabel=None, title=None, log_y=False):
    """Apply common axis styling."""
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    ax.set_xlim(left=0)
    # log_y: for plots mixing a one-time cost with many much-smaller
    # per-update costs (e.g. energy bars) — a linear axis sized to fit the
    # larger one makes the smaller bars round to invisible pixels. Log
    # scale can't include 0, so skip the bottom=0 clamp in that case.
    if log_y:
        ax.set_yscale("log")
    else:
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

    # Support both 2-strategy (no_update/incremental) and multi-strategy (hybrid serving)
    label_map = {
        "no_update":            "No update (model goes stale)",
        "incremental":          f"Incremental update (every {update_every} batches)",
        "full_retrain":         f"Full retrain (every {update_every} batches)",
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

    # Dashed vertical lines mark update moments, one color per updating
    # strategy (no_update never has updated=True rows, so it's naturally
    # excluded).
    updates = df[df.get("updated", pd.Series(False, index=df.index)) == True]
    for strategy, grp in updates.groupby("strategy"):
        color = COLORS.get(strategy, COLORS["incremental"])
        for j, x in enumerate(grp["interactions"]):
            ax.axvline(x, color=color, alpha=0.25, linewidth=0.8, linestyle="--",
                       label=f"{label_map.get(strategy, strategy)} triggered" if j == 0 else None)

    max_x = int(df["interactions"].max())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.005))
    ax.grid(which="minor", alpha=0.15)
    style_ax(ax, ylabel=ylabel, title=f"{ylabel} Over Time")
    ax.set_xlim(left=0, right=max_x)
    filtered = [t for t in ax.get_xticks() if 0 <= t <= max_x]
    ax.set_xticks(sorted(set(filtered + [max_x])))
    ax.legend(loc="upper left", frameon=True)


def plot_energy_bars(ax, df: pd.DataFrame, bar_width: int = 800,
                     training_emissions_mg: float = None):
    """Plot CO2 emissions cost per update as bars (incremental + full_retrain,
    side by side where both occur), plus a reference line for the
    one-time historical training emissions (if provided)."""
    inc_updates = df[(df["strategy"] == "incremental") & (df["updated"])]
    retrain_updates = df[(df["strategy"] == "full_retrain") & (df["updated"])]

    has_both = len(inc_updates) and len(retrain_updates)
    offset = bar_width * 0.55 if has_both else 0
    if len(inc_updates):
        ax.bar(inc_updates["interactions"] - offset, inc_updates["update_emissions_mg"],
               width=bar_width, color=COLORS["incremental"], alpha=0.7,
               label="Incremental update emissions")
    if len(retrain_updates):
        ax.bar(retrain_updates["interactions"] + offset, retrain_updates["update_emissions_mg"],
               width=bar_width, color=COLORS["hybrid"], alpha=0.7,
               label="Full retrain emissions")
    if training_emissions_mg is not None:
        ax.axhline(training_emissions_mg, color=COLORS["no_update"],
                   linewidth=1.5, linestyle="--",
                   label=f"Historical training emissions (one-time, {training_emissions_mg:.2f} mg CO2eq)")
    style_ax(ax,
             xlabel="Interactions seen (real-time stream)",
             ylabel="Update emissions (mg CO2eq, log scale)",
             title="Emissions Cost per Update",
             log_y=True)


def plot_streaming_results(df: pd.DataFrame, out_path: Path,
                           title: str, update_every: int,
                           training_emissions_mg: float = None):
    """
    One PNG per available metric + one emissions PNG.
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

    # Emissions plot
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(title, fontsize=13)
    plot_energy_bars(ax, df, training_emissions_mg=training_emissions_mg)
    plt.tight_layout()
    energy_path = Path(f"{base}_energy.png")
    plt.savefig(energy_path, dpi=DPI)
    print(f"Plot saved → {energy_path}")
    plt.close()


def combine_results(csv_paths: list, out_path: Path, title: str,
                    update_every: int, training_emissions_mg: float = None) -> pd.DataFrame:
    """
    Load two or more separately-saved results CSVs (each with a 'strategy'
    column, matching run_incremental_lightgcn.py's per-strategy output
    schema), concatenate them in memory — never written to disk as a new
    CSV — and plot the combined comparison via plot_streaming_results.
    Returns the combined DataFrame in case the caller wants it for anything
    else.
    """
    dfs = [pd.read_csv(p) for p in csv_paths]
    combined = pd.concat(dfs, ignore_index=True)
    plot_streaming_results(combined, out_path, title, update_every,
                           training_emissions_mg=training_emissions_mg)
    return combined


def _plot_new_user_arrivals(df: pd.DataFrame, base: Path, title: str,
                            batch_size: int = 1000):
    """
    New user arrivals per batch — shared by plot_new_user_analysis and
    plot_content_coldstart, both of which track the same n_new_users column.
    """
    x = df["interactions"]
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(title, fontsize=13)
    ax.bar(x, df["n_new_users"], width=batch_size * 0.8,
           color="#e63946", alpha=0.4, label="New users per batch")
    ax.set_ylabel("Unique new users")
    ax.set_xlabel("Interactions seen")
    ax.set_title("New User Arrivals per Batch")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    plt.tight_layout()
    arrivals_path = Path(f"{base}_new_user_arrivals.png")
    plt.savefig(arrivals_path, dpi=DPI)
    print(f"Plot saved → {arrivals_path}")
    plt.close()


def _plot_population_growth(df: pd.DataFrame, base: Path, title: str):
    """
    Cumulative total population (historical baseline + new users seen so
    far) — shared by plot_new_user_analysis and plot_content_coldstart, both
    of which track n_users_trained/n_new_users.
    """
    x = df["interactions"]
    total_population = df["n_users_trained"].iloc[0] + df["n_new_users"].cumsum()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(title, fontsize=13)
    ax.plot(x, total_population, color="#457b9d", linewidth=2,
            label="Total distinct users seen (historical + new)")
    ax.axhline(df["n_users_trained"].iloc[0], color="#2a9d8f", linewidth=1.5,
               linestyle="--", label="Historical baseline")
    ax.set_ylabel("Total distinct users")
    ax.set_xlabel("Interactions seen")
    ax.set_title("Population Growth Over Time")
    ax.set_xlim(left=0)
    # No ylim(bottom=0) here deliberately — the population only ever grows by
    # a small fraction of its starting size (e.g. ml-1m: 6022 -> ~6034), so
    # forcing the axis down to 0 would squeeze all the actual variation into
    # a sliver at the top of the chart. Let matplotlib auto-scale to the
    # data's real range instead.
    ax.legend()

    plt.tight_layout()
    growth_path = Path(f"{base}_population_growth.png")
    plt.savefig(growth_path, dpi=DPI)
    print(f"Plot saved → {growth_path}")
    plt.close()


def plot_new_user_analysis(df: pd.DataFrame, out_path: Path, title: str,
                           batch_size: int = 1000, smooth: int = 20):
    """
    One PNG per metric (recall/precision/ndcg @10 — each showing existing vs
    new vs overall users), plus one PNG for new user arrivals per batch, plus
    one PNG for cumulative total population growth over time.
    """
    base = Path(str(out_path).replace(".png", ""))
    x = df["interactions"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    group_colors = {
        "existing": "#2a9d8f",
        "new_user": "#e63946",
        "overall":  "#457b9d",
    }
    group_labels = {
        "existing": "Existing users",
        "new_user": "New users — mean emb",
        "overall":  "Overall",
    }
    metrics = [("recall", "Recall@10"), ("precision", "Precision@10"), ("ndcg", "NDCG@10")]

    for metric, label in metrics:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(title, fontsize=13)

        for group, color in group_colors.items():
            col = f"{metric}_{group}"
            ax.plot(x, df[col], color=color, alpha=0.2, linewidth=0.8)
            ax.plot(x, smoothed(col), color=color, linewidth=2,
                    label=f"{group_labels[group]} (smoothed)")

        ax.set_ylabel(label)
        ax.set_title(f"{label} by User Group Over Time")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")

        plt.tight_layout()
        path = Path(f"{base}_{metric}.png")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()

    _plot_new_user_arrivals(df, base, title, batch_size=batch_size)
    _plot_population_growth(df, base, title)



# metric key -> (display label, has a mean-init baseline column to compare against)
# run_new_user_analysis.py only computes recall/precision/ndcg, not hr/mrr, so
# those two have no mean-init counterpart to merge in or plot.
_CONTENT_COLDSTART_METRICS = [
    ("recall",    "Recall@10",    True),
    ("precision", "Precision@10", True),
    ("ndcg",      "NDCG@10",      True),
    ("hr",        "HR@10",        False),
    ("mrr",       "MRR",          False),
]


def plot_content_coldstart(df: pd.DataFrame, out_path: Path, title: str,
                           batch_size: int = 1000, smooth: int = 20):
    """
    For every metric in _CONTENT_COLDSTART_METRICS, one PNG comparing new-user
    performance under content-init (and mean-init too, if the corresponding
    *_new_mean column is present — merge it in from a run_new_user_analysis.py
    CSV via --new-user-csv, since that script already computes the identical
    mean-init numbers and recomputing them here would be redundant), and one
    PNG comparing overall performance, plus the same shared new-user-arrivals
    and population-growth PNGs used by plot_new_user_analysis.
    """
    base = Path(str(out_path).replace(".png", ""))
    x = df["interactions"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    for metric, ylabel, has_baseline_col in _CONTENT_COLDSTART_METRICS:
        has_mean_baseline = has_baseline_col and f"{metric}_new_mean" in df.columns

        # ── New user: content init (+ mean init, if merged in) ──────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(title, fontsize=13)
        series = [(f"{metric}_new_content", "#2a9d8f", "New users — content init")]
        if has_mean_baseline:
            series.append((f"{metric}_new_mean", "#e63946", "New users — mean init"))
        for col, color, label in series:
            ax.plot(x, df[col], color=color, alpha=0.2, linewidth=0.8)
            ax.plot(x, smoothed(col), color=color, linewidth=2, label=f"{label} (smoothed)")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Interactions seen")
        ax.set_title(f"New User {ylabel}: Content-Aware Init"
                     + (" vs Mean Init" if has_mean_baseline else ""))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        plt.tight_layout()
        path = Path(f"{base}_new_user_{metric}.png")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()

        # ── Overall comparison ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(title, fontsize=13)
        series = [
            (f"{metric}_existing",        "#457b9d", "Existing users"),
            (f"{metric}_overall_content", "#2a9d8f", "Overall — content init"),
        ]
        if has_mean_baseline:
            series.append((f"{metric}_overall_mean", "#e63946", "Overall — mean init"))
        for col, color, label in series:
            ax.plot(x, df[col], color=color, alpha=0.2, linewidth=0.8)
            ax.plot(x, smoothed(col), color=color, linewidth=2, label=f"{label} (smoothed)")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Interactions seen")
        ax.set_title(f"Overall {ylabel} Comparison")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        plt.tight_layout()
        path = Path(f"{base}_overall_{metric}.png")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()

    _plot_new_user_arrivals(df, base, title, batch_size=batch_size)
    _plot_population_growth(df, base, title)
