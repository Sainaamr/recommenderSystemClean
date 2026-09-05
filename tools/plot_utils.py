"""
Shared plotting utilities for experiment scripts.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path

# ── Shared style constants ────────────────────────────────────────────────────
COLORS = {
    "no_update":    "#e63946",
    "incremental":  "#2a9d8f",
    "full_retrain": "#2a9d8f",
}
SMOOTH = 30   # rolling average window for recall lines
DPI    = 150

# Output format for every figure. PDF keeps the plots as vectors, so they stay
# sharp at any zoom, embed into LaTeX without resampling, and are far smaller
# than the 150-DPI rasters they replace. DPI still applies to any rasterized
# element inside a vector figure.
FIG_EXT = "pdf"


def _base(out_path) -> str:
    """
    Caller-supplied output path minus its figure extension.

    Call sites pass names ending in ".png" (historically the output format).
    Stripping either extension here means those callers keep working unchanged
    while the files written come out as FIG_EXT.
    """
    s = str(out_path)
    for ext in (".png", ".pdf"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def _out(out_path) -> Path:
    """Caller-supplied output path with its extension normalized to FIG_EXT."""
    return Path(f"{_base(out_path)}.{FIG_EXT}")
# ─────────────────────────────────────────────────────────────────────────────

# Keyed by cost source (not strategy) — one bar can stack multiple sources.
# Unlisted sources cycle through _EMISSIONS_FALLBACK_CYCLE.
# Colorblind-safe: Okabe-Ito for the base set, Paul Tol's "muted" scheme for
# the extra hues plot_energy_summary_stacked's 14 sections need; setup
# phases use grey tints instead of scarce distinct hues (negligible + not
# core content).
EMISSIONS_COMPONENT_COLORS = {
    "training":      "#2a78d6",  # blue
    "update":        "#eb6834",  # orange
    "content_seed":  "#e7ab51",
    "streaming":     "#f4a261",
    "id_mapping":              "#E8E8E8",
    "checkpoint_load":         "#C4C4C4",
    "build_user_history":      "#A0A0A0",
    "embedding_snapshot":      "#7C7C7C",
    "content_build":           "#332288",  # Tol indigo
    "id_resolution":           "#DDCC77",  # Tol sand
    "recovered_history_seed":  "#CC6677",  # Tol rose
    "expand_embeddings":       "#117733",  # Tol green
    "gt_split":                "#88CCEE",  # Tol cyan
    "scoring":                 "#44AA99",  # Tol teal
    "update_prep":             "#999933",  # Tol olive
    "history_update":          "#882255",  # Tol wine
    "apply_content_seeds":     "#AA4499",  # Tol purple
    "update_total":            "#D55E00",  # Okabe-Ito vermilion
}
_EMISSIONS_FALLBACK_CYCLE = ["#e63946", "#8ab17d", "#264653", "#f28482"]


def _center_suptitle_over_axes(fig, ax, title: str, fontsize: int = 13):
    """Centers suptitle over the axes' plotted area instead of the full canvas."""
    fig.suptitle(title, fontsize=fontsize)
    plt.tight_layout()
    pos = ax.get_position()
    fig.suptitle(title, fontsize=fontsize, x=(pos.x0 + pos.x1) / 2)


def _add_end_xtick(ax, max_x: int, min_gap_frac: float = 0.04):
    """
    Forces max_x to be a labeled tick so the last data point is readable.

    Any automatic tick closer than min_gap_frac of the axis range is dropped
    first: without this, a run ending at e.g. 504,000 prints "500000" and
    "504000" on top of each other.
    """
    filtered = [t for t in ax.get_xticks() if 0 <= t <= max_x]
    min_gap = max_x * min_gap_frac
    filtered = [t for t in filtered if max_x - t > min_gap]
    ax.set_xticks(sorted(set(filtered + [max_x])))


def style_ax(ax, xlabel=None, ylabel=None, title=None, zero_bottom=True):
    """Apply common axis styling."""
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    ax.set_xlim(left=0)
    if zero_bottom:
        ax.set_ylim(bottom=0)
    ax.legend()


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


def plot_metric_over_time(ax, df: pd.DataFrame, metric: str = "recall_at_10", subtitle: str = None):
    """
    Plots one metric over time for no_update/incremental/full_retrain, with
    update-trigger markers. Draws onto an existing ax — no file output itself
    (see plot_streaming_results, which wraps this per metric and saves PNGs).
    """
    ylabel = METRIC_LABELS.get(metric, metric)

    label_map = {
        "no_update":    "No-Update",
        "incremental":  "Incremental Update",
        "full_retrain": "Full Retrain",
    }

    for strategy, grp in df.groupby("strategy"):
        color = COLORS.get(strategy, None)
        label = label_map.get(strategy, strategy)
        smoothed = grp[metric].rolling(SMOOTH, min_periods=1, center=True).mean()
        ax.plot(grp["interactions"], grp[metric],
                color=color, alpha=0.15, linewidth=0.8)
        ax.plot(grp["interactions"], smoothed,
                color=color, label=label, linewidth=2.0)

    # no_update never has updated=True rows, so it's naturally excluded here.
    updates = df[df.get("updated", pd.Series(False, index=df.index)) == True]
    for strategy, grp in updates.groupby("strategy"):
        for j, x in enumerate(grp["interactions"]):
            ax.axvline(x, color="black", alpha=0.55, linewidth=1.2, linestyle="--",
                       label=f"{label_map.get(strategy, strategy)} triggered" if j == 0 else None)

    max_x = int(df["interactions"].max())
    style_ax(ax, xlabel="Interactions seen", ylabel=ylabel,
             title=subtitle or f"{ylabel} Over Time", zero_bottom=False)
    ax.set_xlim(left=0, right=max_x)

    # 1st-99th percentile zoom, not strict min/max — keeps rare noise spikes
    # from flattening the real variation.
    y_min, y_max = df[metric].quantile(0.01), df[metric].quantile(0.99)
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(max(0, y_min - pad), y_max + pad)  # metric can't go negative
    _add_end_xtick(ax, max_x)
    ax.legend(loc="upper left", frameon=True)


def _pretty_component_label(col: str) -> str:
    """'update_emissions_mg' -> 'Update', 'content_seed_emissions_mg' -> 'Content seed'."""
    name = col.replace("_emissions_mg", "").replace("_", " ")
    return name[:1].upper() + name[1:]


def _draw_emissions_bars(ax, labels: list, strategy_components: dict, component_order: list,
                         training_emissions_mg: float = None, log_y: bool = False):

    x = list(range(len(labels)))
    bar_width = 0.5

    segments_per_bar = []
    for label in labels:
        segs = []
        if training_emissions_mg is not None:
            segs.append(("training", training_emissions_mg))
        for col in component_order:
            segs.append((col, strategy_components[label].get(col, 0.0)))
        segments_per_bar.append(segs)
    totals = [sum(h for _, h in segs) for segs in segments_per_bar]
    scale = max(totals) if totals and max(totals) > 0 else 1.0

    if log_y:
        nonzero = [h for segs in segments_per_bar for _, h in segs if h > 0]
        axis_floor = (min(nonzero) / 3) if nonzero else 1.0
        ax.set_yscale("log")
        ax.set_ylim(bottom=axis_floor, top=scale * 3)
    else:
        ax.set_ylim(bottom=0, top=scale * 1.2)

    bottoms = [0.0] * len(labels)

    def draw_component(heights, color, legend_label):
        nonlocal bottoms
        ax.bar(x, heights, width=bar_width, bottom=bottoms, color=color, label=legend_label)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    if training_emissions_mg is not None:
        heights = [training_emissions_mg] * len(labels)
        draw_component(heights, EMISSIONS_COMPONENT_COLORS["training"], "Training (one-time)")

    for i, col in enumerate(component_order):
        heights = [strategy_components[label].get(col, 0.0) for label in labels]
        key = col.replace("_emissions_mg", "")
        color = EMISSIONS_COMPONENT_COLORS.get(
            key, _EMISSIONS_FALLBACK_CYCLE[i % len(_EMISSIONS_FALLBACK_CYCLE)])
        draw_component(heights, color, _pretty_component_label(col))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5 - (1 - bar_width), len(labels) - 0.5 + (1 - bar_width))
    ax.set_ylabel("Emissions (mg CO2eq, log scale)" if log_y else "Emissions (mg CO2eq)")
    # Legend outside the axes — with many components it'd cover bars otherwise.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    if not log_y:
        ax.yaxis.set_major_formatter(lambda val, _: f"{val:,.0f}")


def plot_emissions_stacked(strategy_csvs: dict, out_path: Path, title: str,
                           training_emissions_mg: float = None):
    """
    Stacked bar chart of total emissions per strategy/run — one bar per entry
    in strategy_csvs ({label: results_csv_path}). Every column ending in
    "_emissions_mg" is summed per CSV and added as its own stacked segment.
    training_emissions_mg, if given, adds a shared bottom segment on every bar.

    Outputs (results/recent/): yelp_emissions_stacked_no_update_vs_incremental.png,
    yelp_emissions_stacked_no_update_vs_incremental_ongoing_only.png
    """
    labels = list(strategy_csvs.keys())

    # Track column names in first-seen order so segment colors/legend order
    # stay stable regardless of which strategy is listed first.
    component_order = []
    strategy_components = {}
    for label, csv_path in strategy_csvs.items():
        df = pd.read_csv(csv_path)
        sums = {}
        for col in df.columns:
            if col.endswith("_emissions_mg"):
                sums[col] = df[col].sum()
                if col not in component_order:
                    component_order.append(col)
        strategy_components[label] = sums

    fig, ax = plt.subplots(figsize=(max(7, 2.8 * len(labels)), 6.5))
    fig.suptitle(title, fontsize=13, wrap=True)
    _draw_emissions_bars(ax, labels, strategy_components, component_order,
                         training_emissions_mg=training_emissions_mg)
    plt.tight_layout()
    out_path = _out(out_path)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")  # fits the outside legend
    print(f"Plot saved → {out_path}")
    plt.close()

    if component_order:
        ongoing_path = Path(f"{_base(out_path)}_ongoing_only.{FIG_EXT}")
        fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(labels)), 6))
        fig.suptitle(f"{title} (ongoing costs only, training excluded)", fontsize=13, wrap=True)
        _draw_emissions_bars(ax, labels, strategy_components, component_order,
                             training_emissions_mg=None)
        plt.tight_layout()
        plt.savefig(ongoing_path, dpi=DPI, bbox_inches="tight")
        print(f"Plot saved → {ongoing_path}")
        plt.close()


# Segments drawn by plot_energy_summary_stacked — compare_energy_summary's
# full section set, minus training (dwarfs everything else) and the
# derived total/count rows (would double-count).
_ENERGY_STACK_SECTIONS = [
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
    "update_total_emissions_mg",
]


def plot_energy_summary_stacked(summary: pd.DataFrame, out_path: Path, title: str,
                                log_y: bool = False):
    """
    Stacked bar chart of compare_energy_summary()'s full section breakdown
    — one bar per run, one segment per section. Streaming only (training
    excluded, see _ENERGY_STACK_SECTIONS). NaN sections draw as zero-height.

    Outputs: <out_path>.
    """
    labels = list(summary.columns)
    component_order = [s for s in _ENERGY_STACK_SECTIONS if s in summary.index]
    strategy_components = {
        label: {
            section: (float(summary.loc[section, label])
                      if pd.notna(summary.loc[section, label]) else 0.0)
            for section in component_order
        }
        for label in labels
    }

    fig, ax = plt.subplots(figsize=(max(7, 2.8 * len(labels)), 6.5))
    fig.suptitle(title, fontsize=13, wrap=True)
    _draw_emissions_bars(ax, labels, strategy_components, component_order,
                         training_emissions_mg=None, log_y=log_y)
    plt.tight_layout()
    out_path = _out(out_path)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")  # fits the outside legend
    print(f"Plot saved → {out_path}")
    plt.close()


def plot_streaming_results(df: pd.DataFrame, out_path: Path,
                           title: str,
                           training_emissions_mg: float = None, subtitle: str = None,
                           metrics: list = None):
    """
    One PNG per metric column present in df (optionally restricted by
    metrics). No emissions plot here — use plot_emissions_stacked directly.

    Outputs (results/recent/): yelp_combined_no_update_vs_incremental_recall_at_10.png,
    _precision_at_10.png, _ndcg_at_10.png, _hr_at_10.png, _recall_at_20.png,
    _precision_at_20.png, _ndcg_at_20.png, _hr_at_20.png, _mrr.png
    """
    base = Path(_base(out_path))

    available = [m for m in METRIC_LABELS if m in df.columns and (metrics is None or m in metrics)]
    for metric in available:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_metric_over_time(ax, df, metric=metric, subtitle=subtitle)
        _center_suptitle_over_axes(fig, ax, title)
        path = Path(f"{base}_{metric}.{FIG_EXT}")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()


def _plot_new_user_arrivals(df: pd.DataFrame, base: Path, title: str,
                            batch_size: int = 1000, update_every: int = 20,
                            subtitle: str = None):
    """
    New user arrivals per update_every-batch window (the same window an
    incremental update trains on), one bar per window centered on the
    interaction range it covers. Called by plot_new_user_analysis.

    If df has "window_unique_users" (run_new_user_analysis.py only), the bar
    is stacked: users arriving for the first time at the bottom, every other
    active user on top. Without it, only the arrivals bar is drawn.

    The top segment is "returning", not "existing": it is everyone active who
    did not arrive in THIS window, which includes untrained users who arrived
    in an earlier one (the model never retrains here, so they still hold a
    mean-initialised embedding). On yelp-timecut that is ~2,987 of the ~8,336
    users in the top segment, so calling it "existing" would imply the model
    knows three thousand users it has never trained on.

    subtitle, if given, swaps title/subtitle roles: the descriptive text
    becomes the per-axes title instead of the suptitle.

    Outputs (results/recent/): yelp_new_user_analysis_20260809_190658_replot_new_user_arrivals.png
    """
    # Per-chunk width, not a fixed batch_size * update_every — the last
    # chunk is partial whenever batch count isn't a multiple of update_every.
    agg = {"n_new_users": ("n_new_users", "sum"),
          "interactions": ("interactions", "max"),
          "n_batches": ("batch", "count")}
    has_existing = "window_unique_users" in df.columns
    if has_existing:
        # max, not sum: window_unique_users is already a running total, so
        # its value at the window's last batch is the window's true count.
        agg["window_unique_users"] = ("window_unique_users", "max")

    grouped = df.assign(chunk=(df["batch"] - 1) // update_every).groupby("chunk").agg(**agg)
    widths = grouped["n_batches"] * batch_size
    x = grouped["interactions"] - widths / 2  # center each bar on its window

    fig, ax = plt.subplots(figsize=(12, 5))
    # width=widths: windows are contiguous, so bars must touch with no gaps.
    ax.bar(x, grouped["n_new_users"], width=widths,
           color="#E69F00", alpha=0.4, edgecolor="#E69F00", linewidth=1.2,
           label=f"New unique users per {update_every} batches")
    if has_existing:
        n_existing_active = grouped["window_unique_users"] - grouped["n_new_users"]
        ax.bar(x, n_existing_active, width=widths, bottom=grouped["n_new_users"],
              color="#0072B2", alpha=0.4, edgecolor="#0072B2", linewidth=1.2,
              label="Returning active users")
        ax.set_ylabel("Unique active users")
        top_values = grouped["window_unique_users"]
    else:
        ax.set_ylabel("Unique new users")
        top_values = grouped["n_new_users"]
    ax.set_xlabel("Interactions seen")
    descriptive_title = ("Active Users per Update Window" if has_existing
                         else "New Unique User Arrivals per Update Window")
    max_x = int(df["interactions"].max())
    ax.set_xlim(left=0, right=max_x)
    y_min, y_max = top_values.min(), top_values.max()
    pad = (y_max - y_min) * 0.1
    bottom = 0 if has_existing else max(0, y_min - pad)
    ax.set_ylim(bottom, y_max + pad)
    _add_end_xtick(ax, max_x)
    ax.legend(loc="upper left")
    if subtitle:
        ax.set_title(subtitle)
        _center_suptitle_over_axes(fig, ax, descriptive_title)
    else:
        ax.set_title(descriptive_title)
        _center_suptitle_over_axes(fig, ax, title)

    arrivals_path = Path(f"{base}_new_user_arrivals.{FIG_EXT}")
    plt.savefig(arrivals_path, dpi=DPI)
    print(f"Plot saved → {arrivals_path}")
    plt.close()


def _plot_interaction_volume(df: pd.DataFrame, base: Path, title: str,
                             batch_size: int = 1000, update_every: int = 20,
                             subtitle: str = None):
    """
    Interaction-volume counterpart to _plot_new_user_arrivals — same stacked
    layout, and on the same ARRIVALS basis: interactions belonging to users
    who appeared for the first time in this window, via
    "n_first_time_new_user_interactions" (run_new_user_analysis.py only).
    Unlike a person-count, an event count is always safe to sum.

    It must not use "n_new_user_interactions": that counts events from every
    untrained user however long ago they arrived, so pairing it with the
    arrivals bar chart beside it compares two different populations. On
    yelp-timecut the two differ by 5.6x (8,744 vs 1,560 per window) and even
    trend in opposite directions — untrained users accumulate over the
    stream, while arrivals decline.

    subtitle, if given, swaps title/subtitle roles, same as in
    _plot_new_user_arrivals.

    Outputs (results/recent/): yelp_new_user_analysis_20260809_190658_replot_interaction_volume.png
    """
    grouped = (
        df.assign(chunk=(df["batch"] - 1) // update_every)
          .groupby("chunk")
          .agg(n_new_user_interactions=("n_first_time_new_user_interactions", "sum"),
               interactions=("interactions", "max"),
               n_batches=("batch", "count"))
    )
    widths = grouped["n_batches"] * batch_size
    x = grouped["interactions"] - widths / 2
    n_existing_interactions = widths - grouped["n_new_user_interactions"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, grouped["n_new_user_interactions"], width=widths,
          color="#E69F00", alpha=0.4, edgecolor="#E69F00", linewidth=1.2,
          label=f"New unique user interactions per {update_every} batches")
    ax.bar(x, n_existing_interactions, width=widths, bottom=grouped["n_new_user_interactions"],
          color="#0072B2", alpha=0.4, edgecolor="#0072B2", linewidth=1.2,
          label="Returning-user interactions")
    ax.set_ylabel("Interactions")
    ax.set_xlabel("Interactions seen")
    max_x = int(df["interactions"].max())
    ax.set_xlim(left=0, right=max_x)
    ax.set_ylim(0, widths.max() * 1.1)  # stacked total — must start at 0
    _add_end_xtick(ax, max_x)
    ax.legend(loc="upper left")
    if subtitle:
        ax.set_title(subtitle)
        _center_suptitle_over_axes(fig, ax, "Interaction Volume per Update Window")
    else:
        ax.set_title("Interaction Volume per Update Window")
        _center_suptitle_over_axes(fig, ax, title)

    path = Path(f"{base}_interaction_volume.{FIG_EXT}")
    plt.savefig(path, dpi=DPI)
    print(f"Plot saved → {path}")
    plt.close()


def plot_new_user_analysis(df: pd.DataFrame, out_path: Path, title: str,
                           batch_size: int = 1000, smooth: int = 20,
                           subtitle: str = None, update_every: int = 20):
    """
    One PNG per metric (recall/precision/ndcg@10, existing vs new vs
    overall), plus new-user-arrivals and (if available) interaction-volume
    charts.

    Outputs (results/recent/): yelp_new_user_analysis_20260809_190658_replot_recall.png,
    _precision.png, _ndcg.png, _new_user_arrivals.png, _interaction_volume.png
    """
    base = Path(_base(out_path))
    x = df["interactions"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    group_colors = {
        "existing": "#0072B2",  # blue (Wong/Okabe-Ito)
        "new_user": "#E69F00",  # orange
        "overall":  "#56B4E9",  # sky blue
    }
    group_labels = {
        "existing": "Existing users",
        "new_user": "New users",
        "overall":  "Overall",
    }
    metrics = [("recall", "Recall@10"), ("precision", "Precision@10"), ("ndcg", "NDCG@10")]
    max_x = int(x.max())

    for metric, label in metrics:
        fig, ax = plt.subplots(figsize=(12, 5))

        for group, color in group_colors.items():
            col = f"{metric}_{group}"
            ax.plot(x, df[col], color=color, alpha=0.2, linewidth=0.8)
            ax.plot(x, smoothed(col), color=color, linewidth=2,
                    label=group_labels[group])

        # 1st-99th percentile zoom — see plot_metric_over_time for why.
        cols = [f"{metric}_{group}" for group in group_colors]
        y_min, y_max = df[cols].stack().quantile(0.01), df[cols].stack().quantile(0.99)
        pad = (y_max - y_min) * 0.08

        ax.set_xlabel("Interactions seen")
        ax.set_ylabel(label)
        ax.set_title(subtitle or f"{label} by User Group Over Time")
        ax.set_xlim(left=0, right=max_x)
        ax.set_ylim(max(0, y_min - pad), y_max + pad)  # metric can't go negative
        _add_end_xtick(ax, max_x)
        ax.legend(loc="upper left")
        _center_suptitle_over_axes(fig, ax, title)

        path = Path(f"{base}_{metric}.{FIG_EXT}")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()

    _plot_new_user_arrivals(df, base, title, batch_size=batch_size, update_every=update_every,
                            subtitle=subtitle)
    if "n_first_time_new_user_interactions" in df.columns:
        _plot_interaction_volume(df, base, title, batch_size=batch_size, update_every=update_every,
                                 subtitle=subtitle)


# metric key -> (display label, has a mean-init baseline column to compare against)
_CONTENT_COLDSTART_METRICS = [
    ("recall",    "Recall@10",    True),
    ("precision", "Precision@10", True),
    ("ndcg",      "NDCG@10",      True),
]


def plot_content_incremental_groups(df: pd.DataFrame, out_path: Path, title: str,
                                    subtitle: str = None, smooth: int = 20):
    """
    content_incremental equivalent of plot_new_user_analysis's per-metric
    chart — existing/new_content/overall_content on one axes per metric,
    using content_incremental's own column names.

    Outputs (results/): yelp_content_incremental_20260810_211929_replot_recall.png,
    _precision.png, _ndcg.png
    """
    base = Path(_base(out_path))
    x = df["interactions"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    group_colors = {
        "existing":       "#0072B2",  # blue — matches plot_new_user_analysis
        "new_content":    "#E69F00",  # orange
        "overall_content":"#56B4E9",  # sky blue
    }
    group_labels = {
        "existing":        "Existing users",
        "new_content":     "New users",
        "overall_content": "Overall",
    }
    max_x = int(x.max())

    for metric, label, _ in _CONTENT_COLDSTART_METRICS:
        fig, ax = plt.subplots(figsize=(12, 5))

        for group, color in group_colors.items():
            col = f"{metric}_{group}"
            ax.plot(x, df[col], color=color, alpha=0.2, linewidth=0.8)
            ax.plot(x, smoothed(col), color=color, linewidth=2, label=group_labels[group])

        cols = [f"{metric}_{group}" for group in group_colors]
        y_min, y_max = df[cols].stack().quantile(0.01), df[cols].stack().quantile(0.99)
        pad = (y_max - y_min) * 0.08

        ax.set_xlabel("Interactions seen")
        ax.set_ylabel(label)
        ax.set_title(subtitle or f"{label} by User Group Over Time")
        ax.set_xlim(left=0, right=max_x)
        ax.set_ylim(max(0, y_min - pad), y_max + pad)
        _add_end_xtick(ax, max_x)
        ax.legend(loc="upper left")
        _center_suptitle_over_axes(fig, ax, title)

        path = Path(f"{base}_{metric}.{FIG_EXT}")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()


def plot_content_vs_no_update(df: pd.DataFrame, out_path: Path, title: str,
                              subtitle: str = None, smooth: int = 20):
    """
    Two comparison chart families against a frozen no_update baseline, each
    drawn only if its required merged-in columns are present (see
    run_content_coldstart.py's merge_no_update_overall / merge_new_user_baseline):

      "vs_no_update_overall" (needs "{metric}_no_update"): no_update's
      overall metric vs content-init's/content-incremental's overall.

      "vs_no_update_groups" (needs "{metric}_new_mean"/"{metric}_overall_mean"):
      existing (one shared line — verified identical under mean-init and
      content-init) + new/overall under mean-init vs content-init.

    Outputs (results/recent/, from run_content_coldstart.py):
    yelp_content_coldstart_20260810_122201_replot_vs_no_update_overall_recall.png,
    _precision.png, _ndcg.png, _vs_no_update_groups_recall.png, _precision.png, _ndcg.png.
    Also called from run_content_incremental.py (results/): yelp_no_update_vs_content_incremental_recall.png,
    _precision.png, _ndcg.png (only the vs_no_update_overall family, since that CSV has no *_new_mean column).
    """
    base = Path(_base(out_path))
    x = df["interactions"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    max_x = int(x.max())

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        # ── vs_no_update_overall: no_update vs content-init, overall only ──
        if f"{metric}_no_update" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            # content_incremental has an "updated" column (it retrains);
            # plain content_coldstart doesn't — label/title accordingly.
            is_content_incremental = "updated" in df.columns
            overall_label = ("Overall — content incremental" if is_content_incremental
                             else "Overall — content init")
            series = [
                (f"{metric}_no_update",       COLORS["no_update"],   "No update"),
                (f"{metric}_overall_content", COLORS["incremental"], overall_label),
            ]
            for col, color, label in series:
                ax.plot(x, df[col], color=color, alpha=0.2, linewidth=0.8)
                ax.plot(x, smoothed(col), color=color, linewidth=2, label=label)
            if "updated" in df.columns:
                update_x = df.loc[df["updated"] == True, "interactions"]
                for j, xv in enumerate(update_x):
                    ax.axvline(xv, color="black", alpha=0.55, linewidth=1.2, linestyle="--",
                              label="Update triggered" if j == 0 else None)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Interactions seen")
            ax.set_xlim(left=0, right=max_x)
            cols = [c for c, _, _ in series]
            y_min, y_max = df[cols].stack().quantile(0.01), df[cols].stack().quantile(0.99)
            pad = (y_max - y_min) * 0.08
            ax.set_ylim(max(0, y_min - pad), y_max + pad)
            _add_end_xtick(ax, max_x)
            ax.legend(loc="upper left")
            chart_title = ("No Update vs Content Incremental" if is_content_incremental
                          else "No Update vs Content-Init")
            if subtitle:
                ax.set_title(subtitle)
                _center_suptitle_over_axes(fig, ax, chart_title)
            else:
                ax.set_title(chart_title)
                _center_suptitle_over_axes(fig, ax, title)
            path = Path(f"{base}_vs_no_update_overall_{metric}.{FIG_EXT}")
            plt.savefig(path, dpi=DPI)
            print(f"Plot saved → {path}")
            plt.close()

        # ── vs_no_update_groups: existing/new/overall, no_update vs content ─
        if f"{metric}_new_mean" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            # New and Overall each share one hue across their no-update /
            # content-init pair (faded vs full) — "same group, two strategies".
            series = [
                (f"{metric}_existing",        "#2E7D32", "Existing",               1.0),
                (f"{metric}_new_mean",        "#E69F00", "New — no update",        0.5),
                (f"{metric}_overall_mean",    "#1F77B4", "Overall — no update",    0.5),
                (f"{metric}_new_content",     "#E69F00", "New — content init",     1.0),
                (f"{metric}_overall_content", "#1F77B4", "Overall — content init", 1.0),
            ]
            for col, color, label, line_alpha in series:
                ax.plot(x, df[col], color=color, alpha=0.15, linewidth=0.8)
                ax.plot(x, smoothed(col), color=color, linewidth=2, label=label, alpha=line_alpha)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Interactions seen")
            ax.set_xlim(left=0, right=max_x)
            ax.set_ylim(bottom=0)
            _add_end_xtick(ax, max_x)
            ax.legend(loc="upper left", fontsize=9)
            chart_title = "No Update vs Content-Init by User Group"
            if subtitle:
                ax.set_title(subtitle)
                _center_suptitle_over_axes(fig, ax, chart_title)
            else:
                ax.set_title(chart_title)
                _center_suptitle_over_axes(fig, ax, title)
            path = Path(f"{base}_vs_no_update_groups_{metric}.{FIG_EXT}")
            plt.savefig(path, dpi=DPI)
            print(f"Plot saved → {path}")
            plt.close()


def plot_content_init_vs_content_incremental(df_content_init: pd.DataFrame, df_content_incremental: pd.DataFrame,
                                             out_path: Path, title: str, subtitle: str = None, smooth: int = 20):
    """
    Existing/new/overall content-based performance, content-init vs
    content-incremental, on one axes per metric — the two-strategy
    extension of plot_content_vs_no_update's vs_no_update_groups chart.
    Unlike that chart, "Existing" is NOT shared: content-incremental
    retrains, so its existing users' embeddings genuinely diverge from
    content-init's frozen ones (verified, not assumed — max per-batch
    recall diff 0.024), so each group gets its own content-init /
    content-incremental pair, six lines total.

    Outputs (results/): yelp_content_init_vs_content_incremental_{metric}.png per metric.
    """
    df = pd.merge(df_content_init, df_content_incremental, on="batch", suffixes=("_ci", "_cinc"))
    base = Path(_base(out_path))
    x = df["interactions_ci"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    max_x = int(x.max())

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        fig, ax = plt.subplots(figsize=(12, 5))
        # Each group (existing/new/overall) shares one hue across its
        # content-init / content-incremental pair — faded for content-init,
        # full for content-incremental — same light/dark pairing technique
        # used in plot_content_vs_no_update's groups chart.
        series = [
            (f"{metric}_existing_ci",          "#2E7D32", "Existing — content init",        0.5),
            (f"{metric}_existing_cinc",        "#2E7D32", "Existing — content incremental", 1.0),
            (f"{metric}_new_content_ci",       "#E69F00", "New — content init",             0.5),
            (f"{metric}_new_content_cinc",     "#E69F00", "New — content incremental",      1.0),
            (f"{metric}_overall_content_ci",   "#1F77B4", "Overall — content init",         0.5),
            (f"{metric}_overall_content_cinc", "#1F77B4", "Overall — content incremental",  1.0),
        ]
        for col, color, label, line_alpha in series:
            # Raw line is invisible (alpha=0), not omitted — it still
            # counts toward y-axis autoscaling, so hiding the noise doesn't
            # zoom the axes in tighter and push the legend over the lines.
            ax.plot(x, df[col], color=color, alpha=0.0, linewidth=0.8)
            ax.plot(x, smoothed(col), color=color, linewidth=2, label=label, alpha=line_alpha)
        # content-incremental retrains; content-init never does, so update
        # markers only ever come from the _cinc side.
        update_x = df.loc[df["updated"] == True, "interactions_ci"]
        for j, xv in enumerate(update_x):
            ax.axvline(xv, color="black", alpha=0.55, linewidth=1.2, linestyle="--",
                      label="Update triggered" if j == 0 else None)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Interactions seen")
        ax.set_xlim(left=0, right=max_x)
        ax.set_ylim(bottom=0)
        _add_end_xtick(ax, max_x)
        ax.legend(loc="upper left", fontsize=9)
        chart_title = "Content-Init vs Content Incremental by User Group"
        if subtitle:
            ax.set_title(subtitle)
            _center_suptitle_over_axes(fig, ax, chart_title)
        else:
            ax.set_title(chart_title)
            _center_suptitle_over_axes(fig, ax, title)
        path = Path(f"{base}_{metric}.{FIG_EXT}")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()


def plot_all_strategies_comparison(df_no_update: pd.DataFrame, df_incremental: pd.DataFrame,
                                   df_content_coldstart: pd.DataFrame, df_content_incremental: pd.DataFrame,
                                   out_path: Path, title: str, subtitle: str = None, smooth: int = 20,
                                   df_full_retrain: pd.DataFrame = None):
    """
    Overall recall/precision/ndcg@10 for no_update/incremental/
    content_coldstart/content_incremental on one set of axes, smoothed lines
    only. Update-trigger batches — verified identical between incremental
    and content_incremental — are marked with one shared set of dashed lines.

    df_full_retrain, if given, adds a fifth line from its "{metric}_at_10"
    columns (run_incremental_lightgcn.py full_retrain output).

    Outputs (results/): yelp_strategy_comparison_recall.png, _precision.png, _ndcg.png
    (ad hoc — not wired into any script, run manually when needed).
    """
    base = Path(_base(out_path))
    max_x = int(df_no_update["interactions"].max())

    def smoothed(df, col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    update_x = df_incremental.loc[df_incremental["updated"] == True, "interactions"]

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        fig, ax = plt.subplots(figsize=(12, 5))
        # Incremental update and Content cold-start share one hue (orange) at
        # two alphas — same light/dark pairing as the existing/new/overall chart.
        series = [
            (df_no_update,           f"{metric}_at_10",          COLORS["no_update"],   "No update",           1.0),
            (df_incremental,         f"{metric}_at_10",          "#E69F00",             "Incremental update",  1.0),
            (df_content_coldstart,   f"{metric}_overall_content","#56B4E9",             "Content-init",  1.0),
            (df_content_incremental, f"{metric}_overall_content", COLORS["incremental"], "Content incremental", 1.0),
        ]
        if df_full_retrain is not None:
            series.append(
                (df_full_retrain, f"{metric}_at_10", "#7B3294", "Full retrain", 1.0))
        for df_s, col, color, label, line_alpha in series:
            ax.plot(df_s["interactions"], smoothed(df_s, col), color=color, linewidth=2,
                   label=label, alpha=line_alpha)

        for j, xv in enumerate(update_x):
            ax.axvline(xv, color="black", alpha=0.55, linewidth=1.2, linestyle="--",
                      label="Update triggered" if j == 0 else None)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Interactions seen")
        ax.set_xlim(left=0, right=max_x)

        all_vals = pd.concat([df_s[col] for df_s, col, _, _, _ in series])
        y_min, y_max = all_vals.quantile(0.01), all_vals.quantile(0.99)
        span = y_max - y_min
        # Extra top headroom for the top-left legend box, not just clipping room.
        ax.set_ylim(max(0, y_min - span * 0.08), y_max + span * 0.30)
        _add_end_xtick(ax, max_x)
        ax.legend(loc="upper left", fontsize=9)
        chart_title = "Strategy Comparison"
        if subtitle:
            ax.set_title(subtitle)
            _center_suptitle_over_axes(fig, ax, chart_title)
        else:
            ax.set_title(chart_title)
            _center_suptitle_over_axes(fig, ax, title)
        path = Path(f"{base}_{metric}.{FIG_EXT}")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()


def _frequency_comparison_colors(n: int) -> list:
    """
    n evenly-spaced colors from a perceptually uniform sequential colormap
    (plasma, range restricted to avoid the too-light yellow end) — update
    frequency is ordinal, so a light-to-dark ramp reads the low-to-high
    relationship directly, unlike arbitrary categorical hues.
    """
    if n == 1:
        return [mcolors.rgb2hex(cm.plasma(0.15))]
    return [mcolors.rgb2hex(cm.plasma(t)) for t in np.linspace(0.05, 0.85, n)]


def plot_content_incremental_frequency_comparison(runs: dict, out_path: Path, title: str,
                                                   subtitle: str = None, smooth: int = 20):
    """
    Overall recall/precision/ndcg@10 for several content_incremental runs
    at different update_every settings, one line per run — the frequency-
    sweep counterpart to plot_all_strategies_comparison. No update-trigger
    markers: each run has its own schedule, so a shared dashed-line set
    doesn't apply here.

    runs: {label: df}, e.g. {"Update every 270 batches": df1, ...} — label
    order sets both legend order and color assignment (see
    plot_content_incremental_frequency_sweep for auto-discovery + labeling).

    Outputs: <out_path>_{metric}.png per metric.
    """
    base = Path(_base(out_path))
    labels = list(runs.keys())
    colors = _frequency_comparison_colors(len(labels))
    max_x = int(max(df["interactions"].max() for df in runs.values()))

    def smoothed(df, col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        col = f"{metric}_overall_content"
        fig, ax = plt.subplots(figsize=(12, 5))

        for i, label in enumerate(labels):
            df = runs[label]
            ax.plot(df["interactions"], smoothed(df, col), color=colors[i], linewidth=2, label=label)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Interactions seen")
        ax.set_xlim(left=0, right=max_x)

        all_vals = pd.concat([runs[label][col] for label in labels])
        y_min, y_max = all_vals.quantile(0.01), all_vals.quantile(0.99)
        span = y_max - y_min
        ax.set_ylim(max(0, y_min - span * 0.08), y_max + span * 0.30)
        _add_end_xtick(ax, max_x)
        ax.legend(loc="upper left", fontsize=9)
        chart_title = "Update Frequency Comparison"
        if subtitle:
            ax.set_title(subtitle)
            _center_suptitle_over_axes(fig, ax, chart_title)
        else:
            ax.set_title(chart_title)
            _center_suptitle_over_axes(fig, ax, title)
        path = Path(f"{base}_{metric}.{FIG_EXT}")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()


def _discover_frequency_runs(csv_dir: Path) -> list:
    """
    Every content_incremental results CSV in csv_dir (*content_incremental*.csv,
    excluding *_energy.csv sidecars), paired with its actual update_every —
    recovered straight from the data (the batch number of its first
    updated==True row), not guessed from n_updates or hardcoded. Returns
    [(update_every, df, path), ...] sorted least to most frequent
    (descending update_every); a run with no updates at all is skipped.
    """
    csv_paths = [p for p in sorted(Path(csv_dir).glob("*content_incremental*.csv"))
                if "_energy" not in p.name]
    entries = []
    for p in csv_paths:
        df = pd.read_csv(p)
        update_batches = df.loc[df["updated"] == True, "batch"]
        if update_batches.empty:
            continue
        entries.append((int(update_batches.iloc[0]), df, p))
    entries.sort(key=lambda e: e[0], reverse=True)
    return entries


def plot_content_incremental_frequency_sweep(csv_dir: Path, out_path: Path, title: str,
                                             subtitle: str = None, smooth: int = 20):
    """
    Auto-discovers every content_incremental run in csv_dir (see
    _discover_frequency_runs) and plots them via
    plot_content_incremental_frequency_comparison, labeled "Update every N
    batches".

    Outputs: <out_path>_{metric}.png per metric.
    """
    runs = {f"Update every {update_every} batch{'es' if update_every != 1 else ''}": df
            for update_every, df, _ in _discover_frequency_runs(csv_dir)}
    plot_content_incremental_frequency_comparison(runs, out_path, title, subtitle=subtitle, smooth=smooth)


def plot_content_incremental_frequency_summary(csv_dir: Path, out_path: Path, title: str,
                                               subtitle: str = None):
    """
    One point per content_incremental run in csv_dir (see
    _discover_frequency_runs): x = update_every (log scale, ascending —
    more frequent updates to the left), y = that run's mean overall
    recall/precision/ndcg@10. The summary-statistic counterpart to
    plot_content_incremental_frequency_sweep's full time series — shows
    the frequency/quality trend (and where it plateaus) at a glance.

    Outputs: <out_path>_{metric}.png per metric.
    """
    entries = sorted(_discover_frequency_runs(csv_dir), key=lambda e: e[0])
    update_everys = [ue for ue, _, _ in entries]

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        col = f"{metric}_overall_content"
        means = [df[col].mean() for _, df, _ in entries]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(update_everys, means, color=COLORS["incremental"], linewidth=2, marker="o", markersize=6)
        ax.set_xscale("log")
        ax.set_xticks(update_everys)
        ax.set_xticklabels([str(ue) for ue in update_everys])
        ax.set_xlabel("Update every (batches)")
        ax.set_ylabel(f"Mean {ylabel}")
        y_min, y_max = min(means), max(means)
        pad = (y_max - y_min) * 0.15
        ax.set_ylim(max(0, y_min - pad), y_max + pad)
        chart_title = "Update Frequency vs Quality"
        if subtitle:
            ax.set_title(subtitle)
            _center_suptitle_over_axes(fig, ax, chart_title)
        else:
            ax.set_title(chart_title)
            _center_suptitle_over_axes(fig, ax, title)
        path = Path(f"{_base(out_path)}_{metric}.{FIG_EXT}")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()


def _frequency_run_energies(entries: list) -> list:
    """
    streaming_emissions_mg per (update_every, df, path) entry from
    _discover_frequency_runs, read from each run's *_energy.csv sidecar
    (same basename as its results CSV) — None where the sidecar is missing.
    """
    energies = []
    for _, _, path in entries:
        energy_path = Path(str(path).replace(".csv", "_energy.csv"))
        if energy_path.exists():
            energies.append(pd.read_csv(energy_path).iloc[0]["streaming_emissions_mg"])
        else:
            energies.append(None)
    return energies


def _knee_point_index(xs: list, ys: list) -> int:
    """
    Index of the knee/elbow point on (xs, ys), via the kneed package's
    KneeLocator (the standard reference implementation of the Kneedle
    algorithm, Satopaa et al. 2011) — concave, increasing curve (quality
    rises with cost, climbs steeply then flattens). xs/ys are used in the
    order given, so callers should already be sorted along x ascending.
    """
    from kneed import KneeLocator
    knee_x = KneeLocator(xs, ys, curve="concave", direction="increasing").knee
    return xs.index(knee_x)


def _sagitta_feet(xs: list, ys: list) -> list:
    """
    For every point, the foot of its perpendicular onto the chord from the
    first to the last point — i.e. where each point's "sagitta" segment
    (see https://github.com/vlavorini/kneefinder) lands on the chord.
    xs/ys are assumed already min-max normalized to [0, 1] (equal footing
    for both axes — required for the perpendicular to be geometrically
    meaningful, and to render at a true 90° once plotted on equal-aspect
    axes; see plot_content_incremental_quality_vs_energy).
    Returns [(foot_x, foot_y), ...], one per input point.
    """
    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[-1], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = dx * dx + dy * dy

    feet = []
    for px, py in zip(xs, ys):
        t = ((px - x0) * dx + (py - y0) * dy) / denom if denom else 0.0
        feet.append((x0 + t * dx, y0 + t * dy))
    return feet


def plot_content_incremental_quality_vs_energy(csv_dir: Path, out_path: Path, title: str,
                                               subtitle: str = None):
    """
    Quality plotted directly against energy cost — x = streaming energy
    (mg CO2eq), y = mean recall/precision/ndcg@10, one point per
    content_incremental run in csv_dir, connected in update-frequency
    order (least to most frequent). The classic accuracy-vs-cost curve: it
    climbs steeply then bends over while x keeps climbing — diminishing
    returns made visible directly.

    Sagitta-style construction (https://github.com/vlavorini/kneefinder):
    blue curve, orange reference chord from the cheapest to the priciest
    run, and a thin red perpendicular segment from every point to that
    chord — the knee is whichever segment is longest, drawn thicker so
    it's visually obvious which one "wins," not just asserted. Plotted in
    min-max-normalized [0, 1] space with equal-aspect axes (mg CO2eq and a
    0-1 metric have no shared unit, so raw-unit axes would visually skew
    the "perpendicular" segments away from true 90°) — tick labels are
    then relabeled back to the real energy/quality values.

    Each point is labeled with its update_every (energy alone doesn't say
    how frequently that run updated). A run without an *_energy.csv
    sidecar is dropped (no x position).

    Outputs: <out_path>_{metric}.png per metric.
    """
    entries = sorted(_discover_frequency_runs(csv_dir), key=lambda e: e[0], reverse=True)
    energies = _frequency_run_energies(entries)
    valid = [(ue, df, e) for (ue, df, _), e in zip(entries, energies) if e is not None]

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        col = f"{metric}_overall_content"
        xs = [e for _, _, e in valid]
        ys = [df[col].mean() for _, df, _ in valid]
        point_labels = [ue for ue, _, _ in valid]
        knee = _knee_point_index(xs, ys)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        xn = [(x - x_min) / (x_max - x_min) for x in xs]
        yn = [(y - y_min) / (y_max - y_min) for y in ys]
        feet = _sagitta_feet(xn, yn)

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        pad = 0.08
        ax.set_xlim(-pad, 1 + pad)
        ax.set_ylim(-pad, 1 + pad)
        ax.set_aspect("equal")

        ax.plot([xn[0], xn[-1]], [yn[0], yn[-1]], color="#555555", linewidth=1.5,
               zorder=1, label="Reference chord")
        for i, (x, y) in enumerate(zip(xn, yn)):
            fx, fy = feet[i]
            is_knee = i == knee
            ax.plot([x, fx], [y, fy], color=COLORS["no_update"] if is_knee else "#bbbbbb",
                   linewidth=2.5 if is_knee else 1, zorder=2)
        ax.plot(xn, yn, color="#0072B2", linewidth=2, marker="o", markersize=7, zorder=3,
               label="Quality vs. energy")
        ax.scatter([xn[knee]], [yn[knee]], color=COLORS["no_update"], marker="o",
                  s=90, zorder=4, label=f"Knee: update every {point_labels[knee]} batches")
        n = len(xn)
        for i, (x, y, ue) in enumerate(zip(xn, yn, point_labels)):
            # Offset each label perpendicular to the curve's local direction
            # (not a fixed straight-up offset) so it clears the line even on
            # the steep early segments, where "up" runs almost parallel to
            # the curve itself and the label ends up sitting on top of it.
            j0 = i if i == 0 else i - 1
            j1 = i if i == n - 1 else i + 1
            dx, dy = xn[j1] - xn[j0], yn[j1] - yn[j0]
            nx, ny = -dy, dx
            norm = (nx ** 2 + ny ** 2) ** 0.5
            nx, ny = (nx / norm, ny / norm) if norm else (0.0, 1.0)
            mag = 14
            ax.annotate(str(ue), (x, y), textcoords="offset points",
                       xytext=(nx * mag, ny * mag),
                       ha="center", va="center", fontsize=8, color="#555555", zorder=5,
                       fontweight="bold",
                       bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=2))

        tick_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        ax.set_xticks(tick_fracs)
        ax.set_xticklabels([f"{x_min + t * (x_max - x_min):,.0f}" for t in tick_fracs])
        ax.set_yticks(tick_fracs)
        ax.set_yticklabels([f"{y_min + t * (y_max - y_min):.4f}" for t in tick_fracs])
        ax.set_xlabel("Streaming energy (mg CO2eq)")
        ax.set_ylabel(f"Mean {ylabel}")
        ax.legend(loc="lower right", fontsize=8)
        chart_title = "Quality vs Energy Cost"
        if subtitle:
            ax.set_title(subtitle)
            _center_suptitle_over_axes(fig, ax, chart_title)
        else:
            ax.set_title(chart_title)
            _center_suptitle_over_axes(fig, ax, title)
        path = Path(f"{_base(out_path)}_{metric}.{FIG_EXT}")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()


def plot_no_update_incremental_full_retrain(df_no_update: pd.DataFrame, df_incremental: pd.DataFrame,
                                             df_full_retrain: pd.DataFrame, out_path: Path, title: str,
                                             subtitle: str = None, smooth: int = 30):
    """
    Overall recall/precision/ndcg@10 for no_update, incremental, and
    full_retrain on one set of axes per metric — built for the MovieLens
    3-strategy comparison (run_incremental_lightgcn.py's "{metric}_at_10"
    columns, same schema all three strategies share). Fixed colors of its
    own (no_update=red, incremental=teal, full_retrain=dark blue),
    independent of the shared COLORS dict, so this chart's color choices
    don't leak into other charts that reuse COLORS.

    Outputs (results/): ml1m_no_update_vs_incremental_vs_full_retrain_{metric}.png per metric.
    """
    base = Path(_base(out_path))
    max_x = int(df_no_update["interactions"].max())

    def smoothed(df, col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    # incremental and full_retrain share the same update_every schedule in
    # this dataset (verified) — one shared marker set is enough.
    update_x = df_incremental.loc[df_incremental["updated"] == True, "interactions"]

    series_colors = {
        "no_update":    "#e63946",  # red
        "incremental":  "#2a9d8f",  # teal
        "full_retrain": "#023e8a",  # dark blue
    }
    series_labels = {
        "no_update":    "No-Update",
        "incremental":  "Incremental Update",
        "full_retrain": "Full Retrain",
    }

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        col = f"{metric}_at_10"
        fig, ax = plt.subplots(figsize=(12, 5))
        series = [
            (df_no_update,    "no_update"),
            (df_incremental,  "incremental"),
            (df_full_retrain, "full_retrain"),
        ]
        for df_s, strategy in series:
            color = series_colors[strategy]
            ax.plot(df_s["interactions"], df_s[col], color=color, alpha=0.15, linewidth=0.8)
            ax.plot(df_s["interactions"], smoothed(df_s, col), color=color, linewidth=2,
                    label=series_labels[strategy])

        for j, xv in enumerate(update_x):
            ax.axvline(xv, color="black", alpha=0.55, linewidth=1.2, linestyle="--",
                      label="Update triggered" if j == 0 else None)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Interactions seen")
        ax.set_xlim(left=0, right=max_x)

        # 1st-99th percentile zoom, not strict min/max — see plot_metric_over_time.
        all_vals = pd.concat([df_s[col] for df_s, _ in series])
        y_min, y_max = all_vals.quantile(0.01), all_vals.quantile(0.99)
        pad = (y_max - y_min) * 0.08
        ax.set_ylim(max(0, y_min - pad), y_max + pad)

        _add_end_xtick(ax, max_x)
        ax.legend(loc="upper left", fontsize=9)
        chart_title = "No-Update vs Incremental vs Full Retrain"
        if subtitle:
            ax.set_title(subtitle)
            _center_suptitle_over_axes(fig, ax, chart_title)
        else:
            ax.set_title(chart_title)
            _center_suptitle_over_axes(fig, ax, title)
        path = Path(f"{base}_{metric}.{FIG_EXT}")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()
