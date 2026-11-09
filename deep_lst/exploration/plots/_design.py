"""Central design tokens for all evaluation plots: colours, linestyles, RC params.

Import from here rather than defining values locally so that every plot
always uses the same hue, style, and font size for the same case / baseline.

Usage
-----
from deep_lst.exploration.plots._design import (
    CASE_COLORS, OBS_COLOR, DEFAULT_COLOR, RC,
)
plt.rcParams.update(RC)
color = CASE_COLORS.get(case_label, DEFAULT_COLOR)
"""

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

CASE_COLORS: dict[str, str] = {
    "MLP-Rollout": "#d62728",  # red
    "MLP-Diag": "#ff7f0e",  # orange
    "ERA5": "#1f77b4",  # blue
    "hourly clim": "y",  # dark gray
    "smoothed clim": "#8c564b",  # dark brown
    "persistence": "#9467bd",  # purple
}

CASE_LINESTYLES: dict[str, str] = {
    "MLP-Rollout": "-",
    "MLP-Diag": "-",
    "ERA5": "--",
    "hourly clim": "--",
    "smoothed clim": "--",
    "persistence": "--",
}

OBS_COLOR = "#333333"  # SEVIRI observations (near-black)
DEFAULT_COLOR = "#999999"  # fallback for unknown cases
OBS_LINESTYLE = "-"  # observations — always solid
BASELINE_LINESTYLE = "--"  # fallback for unknown baselines

# ---------------------------------------------------------------------------
# RC params
# ---------------------------------------------------------------------------

RC: dict = {
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 15,
}

RC_ANIMATION: dict = {
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "figure.titlesize": 13,
}
