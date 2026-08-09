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

# Colors for stacked emissions charts — keyed by cost SOURCE (training,
# update, content_seed, ...), not by strategy name, since one bar stacks
# multiple cost sources for the same strategy. Extra sources not listed
# here fall back to _EMISSIONS_FALLBACK_CYCLE, cycled in first-seen order.
EMISSIONS_COMPONENT_COLORS = {
    "training":      "#2a78d6",  # blue
    "update":        "#eb6834",  # orange
    "content_seed":  "#e7ab51",
    "content_build": "#6a4c93",
    "streaming":     "#f4a261",
}
_EMISSIONS_FALLBACK_CYCLE = ["#e63946", "#8ab17d", "#264653", "#f28482"]


def _center_suptitle_over_axes(fig, ax, title: str, fontsize: int = 13):
    """fig.suptitle() centers on the whole canvas by default, which reads as
    off-center whenever the axes don't span the full figure width (e.g. the
    y-axis label eats into the left margin) — recenter it over the axes'
    actual plotted area instead, once tight_layout has settled where that is."""
    fig.suptitle(title, fontsize=fontsize)
    plt.tight_layout()
    pos = ax.get_position()
    fig.suptitle(title, fontsize=fontsize, x=(pos.x0 + pos.x1) / 2)


def _add_end_xtick(ax, max_x: int):
    """Force the axis's actual right edge (max_x) to be a labeled tick, so the
    last data point's x-value is always readable even when it doesn't land on
    one of matplotlib's auto-chosen round-number ticks."""
    filtered = [t for t in ax.get_xticks() if 0 <= t <= max_x]
    ax.set_xticks(sorted(set(filtered + [max_x])))


def style_ax(ax, xlabel=None, ylabel=None, title=None, log_y=False, zero_bottom=True):
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
    elif zero_bottom:
        ax.set_ylim(bottom=0)
    # else: let matplotlib auto-scale to the data's actual range, so small
    # variation isn't squeezed into a sliver at the top of the chart.
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

STRATEGY_COLORS = COLORS  # alias for clarity outside this module


def plot_metric_over_time(ax, df: pd.DataFrame, update_every: int,
                          metric: str = "recall_at_10", subtitle: str = None):
    """
    Plot any metric over time for no_update and incremental strategies.
    Faint raw lines + bold smoothed trend. Dashed lines mark update moments.

    subtitle overrides the per-axes title (defaults to "{ylabel} Over Time")
    — e.g. pass the dataset name instead, when the figure's own suptitle
    already states what's being measured.
    """
    ylabel = METRIC_LABELS.get(metric, metric)

    # Support both 2-strategy (no_update/incremental) and multi-strategy (hybrid serving)
    label_map = {
        "no_update":            "No-Update",
        "incremental":          "Incremental Update",
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
        for j, x in enumerate(grp["interactions"]):
            ax.axvline(x, color="black", alpha=0.55, linewidth=1.2, linestyle="--",
                       label=f"{label_map.get(strategy, strategy)} triggered" if j == 0 else None)

    max_x = int(df["interactions"].max())
    style_ax(ax, xlabel="Interactions seen", ylabel=ylabel,
             title=subtitle or f"{ylabel} Over Time", zero_bottom=False)
    ax.set_xlim(left=0, right=max_x)

    # Zoom to the metric's 1st-99th percentile range rather than its strict
    # min/max — the raw per-batch noise has occasional one-batch spikes far
    # outside where the signal actually lives, and including those in the
    # frame is what was making real variation look flat. Percentile-based
    # bounds let a couple of extreme noise points clip at the very edge
    # (they're the faint background line, not the signal) while still
    # containing essentially all of the data and all of the smoothed trend.
    y_min, y_max = df[metric].quantile(0.01), df[metric].quantile(0.99)
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(max(0, y_min - pad), y_max + pad)  # metric can't go negative
    _add_end_xtick(ax, max_x)
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


def _pretty_component_label(col: str) -> str:
    """'update_emissions_mg' -> 'Update', 'content_seed_emissions_mg' -> 'Content seed'."""
    name = col.replace("_emissions_mg", "").replace("_", " ")
    return name[:1].upper() + name[1:]


def _draw_emissions_bars(ax, labels: list, strategy_components: dict, component_order: list,
                         training_emissions_mg: float = None, log_y: bool = False):
    """
    Draws one stacked bar per label onto ax, narrower than matplotlib's
    default so there's visible space around and between them. Shared by
    plot_emissions_stacked's two output charts (with and without training)
    so both stay visually consistent.

    log_y=True switches the y-axis to log scale — needed when training
    (~100x bigger than a single update's cost) is stacked in the same bar,
    so the two bars (no_update vs incremental) end up at comparable,
    readable heights instead of one dwarfing the other on a linear axis.
    Stacking is additive though, so log scale can't make a tiny segment's
    *own* slice of the bar visually bigger — a segment that's <1% of the
    total is still <1% of the bar's height regardless of axis scale, so
    per-segment value labels aren't drawn in log mode (they'd just collide
    at the top). Instead each bar gets a single total label above it; the
    ongoing-only chart (no training, linear scale) is where the small
    per-segment values are actually legible.
    """
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
    Stacked bar charts comparing TOTAL emissions across two or more
    strategies/runs — one bar per entry in strategy_csvs, not a
    time series (see plot_energy_bars for that).

    strategy_csvs: {label: results_csv_path}, e.g.
        {"no_update": "results/.../yelp_hybrid_results_no_update_....csv",
         "incremental": "results/.../yelp_hybrid_results_incremental_....csv"}
    Any number of entries is supported, not just two.

    For each CSV, every column whose name ends in '_emissions_mg' is summed
    across all its rows and added as its own stacked segment — so a
    strategy with more cost sources (e.g. a content+incremental run with
    both content_seed_emissions_mg and update_emissions_mg) automatically
    gets more segments than a simpler one (e.g. no_update, which sums to
    zero on every such column since it never updates), with no need to
    hardcode which columns to expect for which strategy. Each segment's
    value is printed at its center.

    training_emissions_mg, if given, is added as a shared bottom segment on
    every bar — appropriate when comparing strategies that all started from
    the same historical checkpoint, so training cost isn't duplicated as a
    separate figure to reconcile by hand.

    Produces two PNGs:
      out_path                    - full comparison, including training
      out_path with a '_ongoing_only' suffix - the same bars WITHOUT
                                     training, so the much smaller ongoing
                                     costs (update, content_seed, ...) can be
                                     compared directly without training
                                     dwarfing them on a linear axis
    """
    labels = list(strategy_csvs.keys())

    # Sum every *_emissions_mg column per strategy, tracking column names in
    # first-seen order across all input CSVs so segment colors/legend order
    # stay stable regardless of which strategy happens to be listed first.
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
                           title: str, update_every: int,
                           training_emissions_mg: float = None, subtitle: str = None):
    """
    One PNG per available metric.
    out_path is used as base — metric name suffix added per file.

    subtitle overrides each metric plot's per-axes title (see
    plot_metric_over_time) — e.g. the dataset name, when title already
    states the comparison being made.

    No emissions plot is produced here — use plot_energy_bars or
    plot_emissions_stacked directly when an emissions chart is wanted.
    """
    base = Path(str(out_path).replace(".png", ""))

    # One plot per metric column present in df
    available = [m for m in METRIC_LABELS if m in df.columns]
    for metric in available:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_metric_over_time(ax, df, update_every, metric=metric, subtitle=subtitle)
        _center_suptitle_over_axes(fig, ax, title)
        path = Path(f"{base}_{metric}.png")
        plt.savefig(path, dpi=DPI)
        print(f"Plot saved → {path}")
        plt.close()


def combine_results(csv_paths: list, out_path: Path, title: str,
                    update_every: int, training_emissions_mg: float = None,
                    subtitle: str = None) -> pd.DataFrame:
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
                           training_emissions_mg=training_emissions_mg, subtitle=subtitle)
    return combined


def _plot_new_user_arrivals(df: pd.DataFrame, base: Path, title: str,
                            batch_size: int = 1000, update_every: int = 20):
    """
    New user arrivals per update_every-batch window (i.e. the same window
    an incremental update trains on) rather than per single batch — one bar
    per training point instead of 539 barely-visible slivers, each bar
    centered on the interaction range it covers. Shared by
    plot_new_user_analysis and plot_content_coldstart.

    If df has a "window_unique_users" column (run_new_user_analysis.py
    only — a running, cross-batch-deduplicated count of active users since
    the last window boundary; its value at a window's last batch is that
    window's true unique-user total), the bar is stacked: new users at the
    bottom, existing active users (window_unique_users - n_new_users) on
    top, so the full bar height reads as "everyone active this window".
    Without that column (e.g. plot_content_coldstart, which doesn't track
    it), only the new-user bar is drawn, same as before.
    """
    # Per-chunk width, not a fixed batch_size * update_every: the last chunk
    # is partial whenever the batch count isn't a multiple of update_every
    # (e.g. 539 batches / 20 = 26 full chunks + 1 chunk of only 19), so
    # assuming every chunk is full-size would make the last bar overhang
    # into its neighbor.
    agg = {"n_new_users": ("n_new_users", "sum"),
          "interactions": ("interactions", "max"),
          "n_batches": ("batch", "count")}
    has_existing = "window_unique_users" in df.columns
    if has_existing:
        # max, not sum: window_unique_users is already a running total
        # within the window, so its value at the window's last batch (which
        # groupby-max picks out) IS the window's true unique-user count —
        # summing it across batches would double-count repeat visitors.
        agg["window_unique_users"] = ("window_unique_users", "max")

    grouped = df.assign(chunk=(df["batch"] - 1) // update_every).groupby("chunk").agg(**agg)
    widths = grouped["n_batches"] * batch_size
    x = grouped["interactions"] - widths / 2  # center each bar on its window

    fig, ax = plt.subplots(figsize=(12, 5))
    # width=widths (not narrower): these windows are contiguous and
    # exhaustive — chunk N covers exactly where chunk N-1 leaves off, so a
    # gap between bars would visually (and wrongly) suggest uncovered gaps
    # in the stream.
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
    ax.set_title("Active Users per Update Window" if has_existing
                else "New User Arrivals per Update Window")
    max_x = int(df["interactions"].max())
    ax.set_xlim(left=0, right=max_x)
    y_min, y_max = top_values.min(), top_values.max()
    pad = (y_max - y_min) * 0.1
    bottom = 0 if has_existing else max(0, y_min - pad)
    ax.set_ylim(bottom, y_max + pad)
    _add_end_xtick(ax, max_x)
    ax.legend(loc="upper left")
    _center_suptitle_over_axes(fig, ax, title)

    arrivals_path = Path(f"{base}_new_user_arrivals.png")
    plt.savefig(arrivals_path, dpi=DPI)
    print(f"Plot saved → {arrivals_path}")
    plt.close()


def plot_new_user_analysis(df: pd.DataFrame, out_path: Path, title: str,
                           batch_size: int = 1000, smooth: int = 20,
                           subtitle: str = None, update_every: int = 20):
    """
    One PNG per metric (recall/precision/ndcg @10 — each showing existing vs
    new vs overall users), plus one PNG for new user arrivals.

    subtitle overrides each metric plot's per-axes title (see
    plot_metric_over_time) — e.g. the dataset name, when title already
    states what's being measured.
    """
    base = Path(str(out_path).replace(".png", ""))
    x = df["interactions"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    group_colors = {
        "existing": "#0072B2",  # blue (Wong/Okabe-Ito, Nature Methods 2011)
        "new_user": "#E69F00",  # orange (Wong/Okabe-Ito)
        "overall":  "#56B4E9",  # sky blue (Wong/Okabe-Ito)
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

        # Zoom to the 1st-99th percentile range across all three groups
        # instead of forcing the axis down to 0 — see plot_metric_over_time
        # for why (occasional one-batch noise spikes otherwise dictate the
        # whole frame and flatten the real variation).
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

    _plot_new_user_arrivals(df, base, title, batch_size=batch_size, update_every=update_every)



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
