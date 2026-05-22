"""Score plots for forecast evaluation.

score_boxplots(scores_by_group, ...)
    Cross-experiment grouped boxplots. Used by run_comparison.py.
    Compares cases side by side, grouped by sample size (S/M/L) or
    stratification (KG/KGLC).

    Input: scores_by_group : dict[str, dict[str, xr.Dataset]]
        Outer key -- group label on x-axis, e.g. "S", "M", "L".
        Inner key -- case label, e.g. "MLP-Diag", "MLP-Rollout", "ERA5".
        Value     -- xr.Dataset with dim ``valid_time``.

error_overview_plot(scores_by_case, ...)
    Benchmarking overview: side-by-side boxplots of per-timestep RMSE/bias
    for the model and each baseline. One box per source, no grouping.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from deep_lst.evaluation.plots._design import CASE_COLORS, DEFAULT_COLOR, RC

_DEFAULT_COLOR = DEFAULT_COLOR


def score_boxplots(
    scores_by_group: dict[str, dict[str, xr.Dataset]],
    metric: str = "rmse",
    xlabel: str = "Group",
    save_path: Path = None,
    title: str = None,
    ylabel: str = None,
) -> plt.Figure:
    """Grouped boxplots of a score across groups and cases.

    Parameters
    ----------
    scores_by_group : dict[str, dict[str, xr.Dataset]]
        ``{group_label: {case_label: scores_dataset(valid_time,)}}``
    metric : str
        Which variable in the score datasets to plot. Default: "rmse".
    xlabel : str
        Label for the x-axis (e.g. "Sample size" or "Stratification").
    save_path : Path, optional
        If given, save the figure there. Parent dirs are created.
    title : str, optional
        Figure suptitle.
    ylabel : str, optional
        Y-axis label. Defaults to the metric name in uppercase.

    Returns
    -------
    plt.Figure
    """
    plt.rcParams.update(RC)

    groups = list(scores_by_group.keys())
    cases = list(next(iter(scores_by_group.values())).keys())
    n_groups = len(groups)
    n_cases = len(cases)

    # Box positions: groups spread along x, cases offset within each group
    group_spacing = n_cases + 1.5
    offsets = np.linspace(-(n_cases - 1) / 2, (n_cases - 1) / 2, n_cases) * 0.6

    fig, ax = plt.subplots(figsize=(max(6, n_groups * n_cases * 0.9), 5))

    for g_idx, group in enumerate(groups):
        x_center = g_idx * group_spacing
        for c_idx, case in enumerate(cases):
            ds = scores_by_group[group].get(case)
            if ds is None or metric not in ds:
                continue
            vals = ds[metric].values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue

            x_pos = x_center + offsets[c_idx]
            color = CASE_COLORS.get(case, _DEFAULT_COLOR)

            bp = ax.boxplot(
                vals,
                positions=[x_pos],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.6),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                flierprops=dict(marker=".", color=color, alpha=0.4, markersize=4),
                manage_ticks=False,
            )
            # Label for legend -- only add once
            if g_idx == 0:
                bp["boxes"][0].set_label(case)

    ax.set_xticks([g * group_spacing for g in range(n_groups)])
    ax.set_xticklabels(groups)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric.upper())
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        fig.suptitle(title, y=1.01)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def category_score_violinplot(
    scores: xr.Dataset,
    metric: str,
    category_dim: str,
    case_label: str | None = None,
    category_labels: dict | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Violin plot of a per-timestep score across a category dim (e.g. month or class).

    Parameters
    ----------
    scores : xr.Dataset
        Output of ``compute_grouped_scores(..., aggregation="region_time")`` or
        ``compute_class_scores(..., aggregation="region_time")``. Must contain
        ``metric`` with dims ``(category_dim, valid_time)``.
    metric : str
        Variable in ``scores`` to plot.
    category_dim : str
        Name of the categorical dim (e.g. ``"month"`` or ``"kg_class"``).
    case_label : str, optional
        Used to look up a colour from CASE_COLORS.
    category_labels : dict, optional
        Map category value → display label for x-tick labels.
    """
    plt.rcParams.update(RC)

    if metric not in scores:
        raise ValueError(f"metric '{metric}' not in dataset (have: {list(scores.data_vars)})")

    categories = scores[category_dim].values
    color = CASE_COLORS.get(case_label, _DEFAULT_COLOR)

    fig, ax = plt.subplots(figsize=(max(5, len(categories) * 0.8), 5))

    for i, cat in enumerate(categories):
        vals = scores[metric].sel({category_dim: cat}).values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        vp = ax.violinplot(vals, positions=[i], widths=0.7, showmedians=True, showextrema=True)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.65)
        for part in ("cmins", "cmaxes", "cbars"):
            vp[part].set_color(color)
            vp[part].set_linewidth(1.0)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

    def _label(c):
        if category_labels is None:
            return str(c)
        return str(category_labels.get(c, category_labels.get(str(c), c)))

    tick_labels = [_label(c) for c in categories]
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(tick_labels, rotation=0 if category_dim == "month" else 30, ha="center")
    ax.set_xlabel(xlabel or category_dim)
    ax.set_ylabel(ylabel or (f"{metric.upper()} (K)" if metric in ("rmse", "mae", "bias") else metric.upper()))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def error_overview_plot(
    scores_by_case: dict[str, xr.Dataset],
    metric: str = "rmse",
    save_path: Path = None,
    title: str = None,
    ylabel: str = None,
) -> plt.Figure:
    """Benchmarking overview: per-timestep score distribution for model + baselines.

    Shows one boxplot per case (model and each baseline) so the user can
    compare total error spread at a glance without the cross-experiment
    grouping of ``score_boxplots``.

    Parameters
    ----------
    scores_by_case : dict[str, xr.Dataset]
        ``{case_label: scores_dataset}`` -- output of
        ``compute_scores(..., aggregation="region_time")`` for the model
        and each baseline. Each dataset has dim ``valid_time``.
    metric : str
        Which variable to plot. Default: "rmse".
    save_path : Path, optional
        If given, save the figure here.
    title : str, optional
        Figure title.
    ylabel : str, optional
        Y-axis label. Defaults to metric name in uppercase + " (K)".

    Returns
    -------
    plt.Figure
    """
    plt.rcParams.update(RC)

    cases = list(scores_by_case.keys())
    fig, ax = plt.subplots(figsize=(max(5, len(cases) * 1.4), 5))

    for i, (case, ds) in enumerate(scores_by_case.items()):
        if metric not in ds:
            continue
        vals = ds[metric].values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        color = CASE_COLORS.get(case, _DEFAULT_COLOR)
        ax.boxplot(
            vals,
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker=".", color=color, alpha=0.4, markersize=4),
            manage_ticks=False,
        )

    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(cases, rotation=15, ha="right")
    ax.set_ylabel(ylabel or f"{metric.upper()} (K)")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
