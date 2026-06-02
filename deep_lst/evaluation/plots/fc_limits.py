"""Forecast-limit summary plot."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from deep_lst.evaluation.plots._design import RC

# Vertical spread of a row's series markers, in y-units (rows are spaced by 1).
_SERIES_SPREAD = 0.6


def _series_offsets(n: int):
    if n <= 1:
        return [0.0]
    half = _SERIES_SPREAD / 2
    return [half - i * (_SERIES_SPREAD / (n - 1)) for i in range(n)]


def forecast_limit_plot(
    rows: list[dict],
    series_styles: dict[str, dict],
    horizon_days: float,
    save_path,
    title: str,
    legend_title: str,
    xlabel: str = "Forecast limit (days)",
) -> None:
    """Render the forecast-limit dot-plot.

    Parameters
    ----------
    rows
        Ordered top-to-bottom. Each row is ``{"label": str, "series": {key: pt}}``
        where ``pt`` is ``{"central": float|None, "lo": float|None, "hi": float|None}``
        (days). A row with an empty ``series`` reserves a blank y-slot.
    series_styles
        ``{series_key: {"color": str, "marker": str, "label": str}}``. Defines
        draw order, colour, marker and legend label for each series.
    horizon_days
        Rollout horizon; right-censored markers/whiskers are placed here.
    """
    plt.rcParams.update(RC)
    offsets = _series_offsets(len(series_styles))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(rows) + 1.5)))

    # Rows top-to-bottom: first row at the largest y.
    y_positions = list(range(len(rows) - 1, -1, -1))

    for row, y in zip(rows, y_positions):
        for (skey, style), dy in zip(series_styles.items(), offsets):
            pt = row["series"].get(skey)
            if pt is None:
                continue
            yy = y + dy
            color, marker = style["color"], style["marker"]
            central, lo, hi = pt["central"], pt["lo"], pt["hi"]

            if not pt.get("computable", True) or central == 0:
                # Not computable (missing data) or no skill from lead 0:
                # mark with a square at init time (day 0).
                ax.plot(0.0, yy, marker="s", color=color, markersize=7, linestyle="none", clip_on=False)
                continue

            if central is None:
                # Median never crosses zero within the horizon: censored.
                ax.plot(horizon_days, yy, marker=marker, color=color, markersize=7, fillstyle="none", linestyle="none")
                ax.annotate(
                    "",
                    xy=(horizon_days + 0.35, yy),
                    xytext=(horizon_days, yy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                )
                continue

            left = central - (lo if lo is not None else central)
            right_val = hi if hi is not None else horizon_days
            right = max(0.0, right_val - central)
            ax.errorbar(
                central,
                yy,
                xerr=[[max(0.0, left)], [right]],
                fmt=marker,
                color=color,
                markersize=7,
                capsize=3,
                elinewidth=1.5,
                capthick=1.5,
            )
            if hi is None:  # upper bound censored
                ax.annotate(
                    "",
                    xy=(horizon_days + 0.35, yy),
                    xytext=(horizon_days, yy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                )

    ax.axvline(horizon_days, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    series_handles = [
        Line2D([], [], color=s["color"], marker=s["marker"], linestyle="none", markersize=7, label=s["label"])
        for s in series_styles.values()
    ]
    # Marker-shape meaning, in neutral grey so it reads as a key, not a series.
    glyph = "#404040"
    glyph_handles = [
        Line2D([], [], color=glyph, marker="o", linestyle="none", markersize=7, label="limit reached"),
        Line2D(
            [],
            [],
            color=glyph,
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor="none",
            label="skill beyond horizon",
        ),
        Line2D([], [], color=glyph, marker="s", linestyle="none", markersize=7, label="no skill (day 0)"),
    ]
    handles = series_handles + glyph_handles
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(series_handles),
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.4,
    )

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
        print(f"  Saved: {save_path}")
    plt.close(fig)
