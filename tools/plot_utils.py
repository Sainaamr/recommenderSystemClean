"""
Shared plotting utilities for experiment scripts.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path

COLORS = {
    "no_update":    "#e63946",
    "incremental":  "#2a9d8f",
    "full_retrain": "#2a9d8f",
}
SMOOTH = 30
DPI    = 150

FIG_EXT = "pdf"


def _base(out_path) -> str:

    s = str(out_path)
    for ext in (".png", ".pdf"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def _out(out_path) -> Path:
    """Caller-supplied output path with its extension normalized to FIG_EXT."""
    return Path(f"{_base(out_path)}.{FIG_EXT}")

EMISSIONS_COMPONENT_COLORS = {
    "training":      "#2a78d6",
    "update":        "#eb6834",
    "content_seed":  "#e7ab51",
    "streaming":     "#f4a261",
    "id_mapping":              "#E8E8E8",
    "checkpoint_load":         "#C4C4C4",
    "build_user_history":      "#A0A0A0",
    "embedding_snapshot":      "#7C7C7C",
    "content_build":           "#332288",
    "id_resolution":           "#DDCC77",
    "recovered_history_seed":  "#CC6677",
    "expand_embeddings":       "#117733",
    "gt_split":                "#88CCEE",
    "scoring":                 "#44AA99",
    "update_prep":             "#999933",
    "history_update":          "#882255",
    "apply_content_seeds":     "#AA4499",
    "update_total":            "#D55E00",
}
_EMISSIONS_FALLBACK_CYCLE = ["#e63946", "#8ab17d", "#264653", "#f28482"]


def _center_suptitle_over_axes(fig, ax, title: str, fontsize: int = 13):
   
    fig.suptitle(title, fontsize=fontsize)
    plt.tight_layout()
    pos = ax.get_position()
    fig.suptitle(title, fontsize=fontsize, x=(pos.x0 + pos.x1) / 2)


def _add_end_xtick(ax, max_x: int, min_gap_frac: float = 0.04):
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

    updates = df[df.get("updated", pd.Series(False, index=df.index)) == True]
    for strategy, grp in updates.groupby("strategy"):
        for j, x in enumerate(grp["interactions"]):
            ax.axvline(x, color="black", alpha=0.55, linewidth=1.2, linestyle="--",
                       label=f"{label_map.get(strategy, strategy)} triggered" if j == 0 else None)

    max_x = int(df["interactions"].max())
    style_ax(ax, xlabel="Interactions seen", ylabel=ylabel,
             title=subtitle or f"{ylabel} Over Time", zero_bottom=False)
    ax.set_xlim(left=0, right=max_x)

    y_min, y_max = df[metric].quantile(0.01), df[metric].quantile(0.99)
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(max(0, y_min - pad), y_max + pad)
    _add_end_xtick(ax, max_x)
    ax.legend(loc="upper left", frameon=True)


def _pretty_component_label(col: str) -> str:
 
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
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    if not log_y:
        ax.yaxis.set_major_formatter(lambda val, _: f"{val:,.0f}")


def plot_emissions_stacked(strategy_csvs: dict, out_path: Path, title: str,
                           training_emissions_mg: float = None):

    labels = list(strategy_csvs.keys())

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
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
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
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.close()


def plot_streaming_results(df: pd.DataFrame, out_path: Path,
                           title: str,
                           training_emissions_mg: float = None, subtitle: str = None,
                           metrics: list = None):
 
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

    agg = {"n_new_users": ("n_new_users", "sum"),
          "interactions": ("interactions", "max"),
          "n_batches": ("batch", "count")}
    has_existing = "window_unique_users" in df.columns
    if has_existing:
        agg["window_unique_users"] = ("window_unique_users", "max")

    grouped = df.assign(chunk=(df["batch"] - 1) // update_every).groupby("chunk").agg(**agg)
    widths = grouped["n_batches"] * batch_size
    x = grouped["interactions"] - widths / 2

    fig, ax = plt.subplots(figsize=(12, 5))
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
    ax.set_ylim(0, widths.max() * 1.1)
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
  
    base = Path(_base(out_path))
    x = df["interactions"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    group_colors = {
        "existing": "#0072B2",
        "new_user": "#E69F00",
        "overall":  "#56B4E9",
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

    _plot_new_user_arrivals(df, base, title, batch_size=batch_size, update_every=update_every,
                            subtitle=subtitle)
    if "n_first_time_new_user_interactions" in df.columns:
        _plot_interaction_volume(df, base, title, batch_size=batch_size, update_every=update_every,
                                 subtitle=subtitle)


_CONTENT_COLDSTART_METRICS = [
    ("recall",    "Recall@10",    True),
    ("precision", "Precision@10", True),
    ("ndcg",      "NDCG@10",      True),
]


def plot_content_incremental_groups(df: pd.DataFrame, out_path: Path, title: str,
                                    subtitle: str = None, smooth: int = 20):
   
    base = Path(_base(out_path))
    x = df["interactions"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    group_colors = {
        "existing":       "#0072B2",
        "new_content":    "#E69F00",
        "overall_content":"#56B4E9",
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
  
    base = Path(_base(out_path))
    x = df["interactions"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    max_x = int(x.max())

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        if f"{metric}_no_update" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
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

        if f"{metric}_new_mean" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
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
   
    df = pd.merge(df_content_init, df_content_incremental, on="batch", suffixes=("_ci", "_cinc"))
    base = Path(_base(out_path))
    x = df["interactions_ci"]

    def smoothed(col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    max_x = int(x.max())

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        fig, ax = plt.subplots(figsize=(12, 5))
        series = [
            (f"{metric}_existing_ci",          "#2E7D32", "Existing — content init",        0.5),
            (f"{metric}_existing_cinc",        "#2E7D32", "Existing — content incremental", 1.0),
            (f"{metric}_new_content_ci",       "#E69F00", "New — content init",             0.5),
            (f"{metric}_new_content_cinc",     "#E69F00", "New — content incremental",      1.0),
            (f"{metric}_overall_content_ci",   "#1F77B4", "Overall — content init",         0.5),
            (f"{metric}_overall_content_cinc", "#1F77B4", "Overall — content incremental",  1.0),
        ]
        for col, color, label, line_alpha in series:
            ax.plot(x, df[col], color=color, alpha=0.0, linewidth=0.8)
            ax.plot(x, smoothed(col), color=color, linewidth=2, label=label, alpha=line_alpha)
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

    base = Path(_base(out_path))
    max_x = int(df_no_update["interactions"].max())

    def smoothed(df, col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    update_x = df_incremental.loc[df_incremental["updated"] == True, "interactions"]

    for metric, ylabel, _ in _CONTENT_COLDSTART_METRICS:
        fig, ax = plt.subplots(figsize=(12, 5))
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

    if n == 1:
        return [mcolors.rgb2hex(cm.plasma(0.15))]
    return [mcolors.rgb2hex(cm.plasma(t)) for t in np.linspace(0.05, 0.85, n)]


def plot_content_incremental_frequency_comparison(runs: dict, out_path: Path, title: str,
                                                   subtitle: str = None, smooth: int = 20):
 
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
 
    runs = {f"Update every {update_every} batch{'es' if update_every != 1 else ''}": df
            for update_every, df, _ in _discover_frequency_runs(csv_dir)}
    plot_content_incremental_frequency_comparison(runs, out_path, title, subtitle=subtitle, smooth=smooth)


def plot_content_incremental_frequency_summary(csv_dir: Path, out_path: Path, title: str,
                                               subtitle: str = None):

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

    energies = []
    for _, _, path in entries:
        energy_path = Path(str(path).replace(".csv", "_energy.csv"))
        if energy_path.exists():
            energies.append(pd.read_csv(energy_path).iloc[0]["streaming_emissions_mg"])
        else:
            energies.append(None)
    return energies


def _knee_point_index(xs: list, ys: list) -> int:

    from kneed import KneeLocator
    knee_x = KneeLocator(xs, ys, curve="concave", direction="increasing").knee
    return xs.index(knee_x)


def _sagitta_feet(xs: list, ys: list) -> list:

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

    base = Path(_base(out_path))
    max_x = int(df_no_update["interactions"].max())

    def smoothed(df, col):
        return df[col].rolling(smooth, min_periods=1, center=True).mean()

    update_x = df_incremental.loc[df_incremental["updated"] == True, "interactions"]

    series_colors = {
        "no_update":    "#e63946",
        "incremental":  "#2a9d8f",
        "full_retrain": "#023e8a",
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
