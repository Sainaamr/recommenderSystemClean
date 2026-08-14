"""
Shared plotting utilities for experiment scripts.
"""

import matplotlib.pyplot as plt
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
# ─────────────────────────────────────────────────────────────────────────────

# Keyed by cost source (not strategy) — one bar can stack multiple sources.
# Unlisted sources cycle through _EMISSIONS_FALLBACK_CYCLE.
EMISSIONS_COMPONENT_COLORS = {
    "training":      "#2a78d6",  # blue
    "update":        "#eb6834",  # orange
    "content_seed":  "#e7ab51",
    "content_build": "#6a4c93",
    "streaming":     "#f4a261",
}
_EMISSIONS_FALLBACK_CYCLE = ["#e63946", "#8ab17d", "#264653", "#f28482"]


def _center_suptitle_over_axes(fig, ax, title: str, fontsize: int = 13):
    """Centers suptitle over the axes' plotted area instead of the full canvas."""
    fig.suptitle(title, fontsize=fontsize)
    plt.tight_layout()
    pos = ax.get_position()
    fig.suptitle(title, fontsize=fontsize, x=(pos.x0 + pos.x1) / 2)


def _add_end_xtick(ax, max_x: int):
    """Forces max_x to be a labeled tick so the last data point is readable."""
    filtered = [t for t in ax.get_xticks() if 0 <= t <= max_x]
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
    ax.legend(loc="upper left")
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
    plt.savefig(out_path, dpi=DPI)
    print(f"Plot saved → {out_path}")
    plt.close()

    if component_order:
        ongoing_path = Path(str(out_path).replace(".png", "_ongoing_only.png"))
        fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(labels)), 6))
        fig.suptitle(f"{title} (ongoing costs only, training excluded)", fontsize=13, wrap=True)
        _draw_emissions_bars(ax, labels, strategy_components, component_order,
                             training_emissions_mg=None)
        plt.tight_layout()
        plt.savefig(ongoing_path, dpi=DPI)
        print(f"Plot saved → {ongoing_path}")
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
    base = Path(str(out_path).replace(".png", ""))

    available = [m for m in METRIC_LABELS if m in df.columns and (metrics is None or m in metrics)]
    for metric in available:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_metric_over_time(ax, df, metric=metric, subtitle=subtitle)
        _center_suptitle_over_axes(fig, ax, title)
        path = Path(f"{base}_{metric}.png")
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
    is stacked: new users at the bottom, existing active users on top.
    Without it, only the new-user bar is drawn.

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
           label=f"New users per {update_every} batches")
    if has_existing:
        n_existing_active = grouped["window_unique_users"] - grouped["n_new_users"]
        ax.bar(x, n_existing_active, width=widths, bottom=grouped["n_new_users"],
              color="#0072B2", alpha=0.4, edgecolor="#0072B2", linewidth=1.2,
              label="Existing active users")
        ax.set_ylabel("Unique active users")
        top_values = grouped["window_unique_users"]
    else:
        ax.set_ylabel("Unique new users")
        top_values = grouped["n_new_users"]
    ax.set_xlabel("Interactions seen")
    descriptive_title = ("Active Users per Update Window" if has_existing
                         else "New User Arrivals per Update Window")
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

    arrivals_path = Path(f"{base}_new_user_arrivals.png")
    plt.savefig(arrivals_path, dpi=DPI)
    print(f"Plot saved → {arrivals_path}")
    plt.close()


def _plot_interaction_volume(df: pd.DataFrame, base: Path, title: str,
                             batch_size: int = 1000, update_every: int = 20,
                             subtitle: str = None):
    """
    Interaction-volume counterpart to _plot_new_user_arrivals — same stacked
    layout but counts raw interactions via "n_new_user_interactions"
    (run_new_user_analysis.py only), which unlike a person-count is always
    safe to sum across a window.

    subtitle, if given, swaps title/subtitle roles, same as in
    _plot_new_user_arrivals.

    Outputs (results/recent/): yelp_new_user_analysis_20260809_190658_replot_interaction_volume.png
    """
    grouped = (
        df.assign(chunk=(df["batch"] - 1) // update_every)
          .groupby("chunk")
          .agg(n_new_user_interactions=("n_new_user_interactions", "sum"),
               interactions=("interactions", "max"),
               n_batches=("batch", "count"))
    )
    widths = grouped["n_batches"] * batch_size
    x = grouped["interactions"] - widths / 2
    n_existing_interactions = widths - grouped["n_new_user_interactions"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, grouped["n_new_user_interactions"], width=widths,
          color="#E69F00", alpha=0.4, edgecolor="#E69F00", linewidth=1.2,
          label=f"New-user interactions per {update_every} batches")
    ax.bar(x, n_existing_interactions, width=widths, bottom=grouped["n_new_user_interactions"],
          color="#0072B2", alpha=0.4, edgecolor="#0072B2", linewidth=1.2,
          label="Existing-user interactions")
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

    path = Path(f"{base}_interaction_volume.png")
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
    base = Path(str(out_path).replace(".png", ""))
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

        path = Path(f"{base}_{metric}.png")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()

    _plot_new_user_arrivals(df, base, title, batch_size=batch_size, update_every=update_every,
                            subtitle=subtitle)
    if "n_new_user_interactions" in df.columns:
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
    base = Path(str(out_path).replace(".png", ""))
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

        path = Path(f"{base}_{metric}.png")
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
    base = Path(str(out_path).replace(".png", ""))
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
            path = Path(f"{base}_vs_no_update_overall_{metric}.png")
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
            path = Path(f"{base}_vs_no_update_groups_{metric}.png")
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
    base = Path(str(out_path).replace(".png", ""))
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
        path = Path(f"{base}_{metric}.png")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()


def plot_all_strategies_comparison(df_no_update: pd.DataFrame, df_incremental: pd.DataFrame,
                                   df_content_coldstart: pd.DataFrame, df_content_incremental: pd.DataFrame,
                                   out_path: Path, title: str, subtitle: str = None, smooth: int = 20):
    """
    Overall recall/precision/ndcg@10 for no_update/incremental/
    content_coldstart/content_incremental on one set of axes, smoothed lines
    only. Update-trigger batches — verified identical between incremental
    and content_incremental — are marked with one shared set of dashed lines.

    Outputs (results/): yelp_strategy_comparison_recall.png, _precision.png, _ndcg.png
    (ad hoc — not wired into any script, run manually when needed).
    """
    base = Path(str(out_path).replace(".png", ""))
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
        path = Path(f"{base}_{metric}.png")
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
    base = Path(str(out_path).replace(".png", ""))
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
        path = Path(f"{base}_{metric}.png")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()
