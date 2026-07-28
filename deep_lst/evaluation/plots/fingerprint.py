"""Fingerprint plot — (DOY × hour-of-day) heatmap of spatially-averaged values.

Accepts any xr.DataArray or xr.Dataset with a ``valid_time`` dimension.
Used for raw LST/SKT, mean absolute error, or RMSE improvement.

fingerprint_plot(data, ...)
    Multi-panel heatmap, one panel per data source.
    X-axis: hour of day (0–23).
    Y-axis: DOY (1–366) with month-name tick labels.
    Color: spatially-averaged value at each (DOY, hour) bin.

Usage
-----
from deep_lst.evaluation.plots.fingerprint import fingerprint_plot

# Raw LST — baselines or model pred/obs
fingerprint_plot(
    {"ERA5": ds_era5["skt"], "SEVIRI obs": obs_da, "MLP-Diag pred": pred_da},
    cbar_label="LST (K)",
    save_path=Path("fingerprint_lst.png"),
)

# Error fingerprint — |pred - obs| and |baseline - obs|
from deep_lst.evaluation.plots.fingerprint import abs_error_da
fingerprint_plot(
    {"MLP-Diag": abs_error_da(pred, obs), "ERA5": abs_error_da(era5_da, obs)},
    cmap="Reds", cbar_label="MAE (K)",
    save_path=Path("fingerprint_error.png"),
)

"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.plots._design import RC

# Month start DOYs (non-leap year) and labels for y-axis ticks
_MONTH_DOYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Helper DataArray constructors for error / improvement inputs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Core binning helpers
# ---------------------------------------------------------------------------


def _extract(data: "xr.DataArray | xr.Dataset", var: str | None = None) -> xr.DataArray:
    """Return a DataArray from either a DataArray or a Dataset."""
    if isinstance(data, xr.DataArray):
        return data
    var = var or ("skt" if "skt" in data else "LST")
    return data[var]


def _to_fingerprint(da: xr.DataArray) -> np.ndarray:
    """Bin a DataArray(valid_time, ...) into a (367, 24) DOY × local-solar-hour matrix.

    Pixels are grouped by their whole-hour solar shift (round(lon/15), exact for hourly
    data); each group is spatially averaged, binned on the UTC clock and placed at its
    solar hours, and the groups combine weighted by group size. A single-shift region
    (every box site) reduces to the plain spatial mean, rolled. Inputs without a lon
    coordinate (already spatially averaged) keep the UTC clock — no shift is known then.
    Cells with no observations remain NaN. DOY axis is 1-based; index 0 unused.
    """
    spatial_dims = [d for d in da.dims if d != "valid_time"]
    times = pd.DatetimeIndex(da["valid_time"].values)
    doys = times.day_of_year.values
    hours = times.hour.values

    if spatial_dims and "lon" in da.coords:
        shift_per_lon = np.round(da["lon"] / 15.0).astype(int)
        groups = []
        for shift in np.unique(shift_per_lon.values):
            group_mask = shift_per_lon == shift
            group_mean = da.where(group_mask).mean(dim=spatial_dims, skipna=True).values
            groups.append((int(shift), group_mean, int(group_mask.sum())))
    elif spatial_dims:
        groups = [(0, da.mean(dim=spatial_dims, skipna=True).values, 1)]
    else:
        groups = [(0, da.values.astype(float), 1)]

    value_sum = np.zeros((367, 24))
    weight_sum = np.zeros((367, 24))
    for shift, values, weight in groups:
        solar_hours = (hours + shift) % 24
        for i, value in enumerate(values):
            if np.isfinite(value):
                value_sum[doys[i], solar_hours[i]] += weight * value
                weight_sum[doys[i], solar_hours[i]] += weight

    mat = np.full((367, 24), np.nan, dtype=np.float32)
    filled = weight_sum > 0
    mat[filled] = value_sum[filled] / weight_sum[filled]
    return mat  # shape (367, 24); row 0 unused


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------


def fingerprint_plot(
    data: dict[str, "xr.DataArray | xr.Dataset"],
    title: str = None,
    save_path: Path = None,
    cmap: str = "plasma",
    vmin: float = None,
    vmax: float = None,
    vcenter: float = None,
    cbar_label: str = "LST / SKT  (K)",
) -> plt.Figure:
    """Multi-panel fingerprint heatmap — one panel per data source.

    Parameters
    ----------
    data : dict[str, xr.DataArray | xr.Dataset]
        ``{label: da_or_ds}`` — each entry becomes one panel.
        DataArrays must have a ``valid_time`` coordinate.
        Datasets must have ``valid_time`` and a variable named ``"skt"`` or ``"LST"``
        (auto-detected; override by passing a DataArray directly).
        Panels are shown left-to-right in insertion order.
    title : str, optional
        Figure super-title.
    save_path : Path, optional
        If given, save the figure here.
    cmap : str
        Matplotlib colormap. Default "plasma" (sequential).
        Use "RdBu_r" for improvement/diverging plots.
    vmin, vmax : float, optional
        Shared color limits. Auto-computed from 2nd–98th percentile if not given.
    vcenter : float, optional
        If set, uses a diverging norm centred at this value (e.g. 0.0 for
        improvement plots). Overrides symmetric vmin/vmax auto-scaling.
    cbar_label : str
        Colorbar label. Default: "LST / SKT  (K)".

    Returns
    -------
    plt.Figure
    """
    plt.rcParams.update(RC)

    n_panels = len(data)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.5 * n_panels + 0.8, 7),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    # --- Build fingerprint matrices ---
    matrices = {label: _to_fingerprint(_extract(src)) for label, src in data.items()}

    # --- Color limits ---
    all_vals = np.concatenate([m[1:].ravel() for m in matrices.values()])
    finite = all_vals[np.isfinite(all_vals)]
    if vmin is None:
        vmin = float(np.nanpercentile(finite, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(finite, 98))

    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        # No finite data anywhere: render a (blank) panel with a trivial norm
        # instead of crashing on TwoSlopeNorm's monotonicity check.
        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    elif vcenter is not None:
        all_negative = vmax <= vcenter
        all_positive = vmin >= vcenter
        if all_negative:
            # Model uniformly better: blue (large improvement) → white (near zero).
            # cmap = "summer"
            vmax = vcenter
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        elif all_positive:
            # Baseline uniformly better: white (near zero) → red (large regression).
            # cmap = "Reds"
            vmin = vcenter
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            # Two-sided: diverging colormap, white anchored at vcenter.
            # cmap = "RdBu"
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = None
    for ax, (label, mat) in zip(axes, matrices.items()):
        display = mat[1:]  # rows 1..366, shape (366, 24)

        im = ax.imshow(
            display,
            aspect="auto",
            origin="upper",
            cmap="RdBu",
            norm=norm,
            extent=[-0.5, 23.5, 366.5, 0.5],
            interpolation="nearest",
        )
        ax.set_title(label)
        ax.set_xlabel("Local solar hour of day")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(["00", "06", "12", "18", "23"])
        ax.set_yticks(_MONTH_DOYS)
        ax.set_yticklabels(_MONTH_NAMES if ax is axes[0] else [])

    axes[0].set_ylabel("Day of year")

    if title:
        fig.suptitle(title, y=1.01)

    fig.tight_layout()

    # Colorbar flush to the right of all panels
    fig.canvas.draw()
    pos = axes[-1].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.018, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, labelpad=8)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved fingerprint plot: {save_path}")

    return fig
