"""Spatial maps of forecast evaluation metrics.

Two functions:

bias_maps(bias_by_case, ...)
    Three side-by-side maps of mean bias (obs − pred) for cases A, B, C.
    Input: region_space aggregation → (lat, lon) per case.

improvement_map(rmse_model, rmse_baseline, ...)
    Single map of RMSE(C) − RMSE(B): positive = model better than ERA5.
    Input: region_space aggregation → (lat, lon) for model and baseline.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from deep_lst.evaluation.plots._design import RC

_PROJ = ccrs.PlateCarree()


def _add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}


def bias_maps(
    bias_by_case: dict[str, xr.DataArray],
    save_path: Path,
    eval_region: str = "",
    vmax: float = None,
    title: str = None,
) -> plt.Figure:
    """Three side-by-side maps of mean bias for cases A, B, C.

    Parameters
    ----------
    bias_by_case : dict[str, xr.DataArray]
        ``{case_label: bias_da}`` with dims (lat, lon).
        Expected keys in display order, e.g.:
        ``{"MLP-Diag": da_A, "MLP-Rollout": da_B, "ERA5": da_C}``
    save_path : Path
        Output file path.
    eval_region : str
        Used in figure title.
    vmax : float, optional
        Symmetric colorbar limit. If None, derived from data.
    title : str, optional
        Override figure suptitle.

    Returns
    -------
    plt.Figure
    """
    plt.rcParams.update(RC)
    cases = list(bias_by_case.keys())
    n = len(cases)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 5),
        subplot_kw={"projection": _PROJ},
    )
    if n == 1:
        axes = [axes]

    # Shared symmetric colorbar limits
    if vmax is None:
        all_vals = np.concatenate([da.values[np.isfinite(da.values)].ravel() for da in bias_by_case.values()])
        vmax = float(np.nanpercentile(np.abs(all_vals), 95)) if len(all_vals) else 5.0
    vmin = -vmax

    ims = []
    for ax, case in zip(axes, cases):
        da = bias_by_case[case]
        im = da.plot(
            ax=ax,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            transform=_PROJ,
        )
        ims.append(im)
        ax.set_title(case)
        _add_map_features(ax)

    fig.subplots_adjust(wspace=0.35, right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.7])
    cbar = fig.colorbar(ims[-1], cax=cbar_ax)
    cbar.set_label("Bias  (pred − obs)  (K)", rotation=270, labelpad=18)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


def acc_map(
    acc: xr.DataArray,
    save_path: Path,
    eval_region: str = "",
    title: str = None,
    cbar_label: str = "ACC  (anomaly correlation coefficient)",
) -> plt.Figure:
    """Single map of anomaly correlation coefficient (ACC) per pixel.

    Parameters
    ----------
    acc : xr.DataArray
        Per-pixel ACC with dims (lat, lon).
    save_path : Path
    eval_region : str
    title : str, optional
    """
    plt.rcParams.update(RC)

    cmap = plt.cm.get_cmap("RdYlGn", 10)
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": _PROJ})
    im = acc.plot(
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        add_colorbar=False,
        transform=_PROJ,
    )
    _add_map_features(ax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=22)

    suptitle = title or (f"ACC — {eval_region}" if eval_region else "ACC")
    ax.set_title(suptitle)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


def improvement_map_multi(
    rmse_model: xr.DataArray,
    rmse_by_baseline: dict[str, xr.DataArray],
    save_path: Path,
    model_label: str = "MLP-Rollout",
    eval_region: str = "",
    vmax: float = None,
    title: str = None,
) -> plt.Figure:
    """Multi-panel improvement map: one panel per baseline.

    Parameters
    ----------
    rmse_model : xr.DataArray
        Per-pixel RMSE of the model, dims (lat, lon).
    rmse_by_baseline : dict[str, xr.DataArray]
        ``{baseline_label: rmse_da}`` — one panel per entry.
    save_path : Path
    model_label : str
        Display name of the model (used in panel titles and suptitle).
    eval_region : str
        Used in suptitle.
    vmax : float, optional
        Shared symmetric colorbar limit. Derived from data if None.
    title : str, optional
        Override suptitle.

    Returns
    -------
    plt.Figure
    """
    plt.rcParams.update(RC)
    cases = list(rmse_by_baseline.keys())
    n = len(cases)
    if n == 0:
        return None

    improvements = {lbl: rmse_bl - rmse_model for lbl, rmse_bl in rmse_by_baseline.items()}

    if vmax is None:
        all_vals = np.concatenate([imp.values[np.isfinite(imp.values)].ravel() for imp in improvements.values()])
        vmax = float(np.nanpercentile(np.abs(all_vals), 95)) if len(all_vals) else 2.0

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(6 * n, 5.5),
        subplot_kw={"projection": _PROJ},
    )
    if n == 1:
        axes = [axes]

    ims = []
    for ax, lbl in zip(axes, cases):
        imp = improvements[lbl]
        im = imp.plot(
            ax=ax,
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
            add_colorbar=False,
            transform=_PROJ,
        )
        ims.append(im)
        ax.set_title(f"vs {lbl}")
        _add_map_features(ax)

    fig.subplots_adjust(wspace=0.35, right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.7])
    cbar = fig.colorbar(ims[-1], cax=cbar_ax)
    cbar.set_label(
        "RMSE diff (K) - Positive = DeepLST better",
        rotation=270,
        labelpad=22,
    )

    suptitle = title or (f"RMSE(baseline) − RMSE({model_label})")
    fig.suptitle(suptitle, y=1.02)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


def lead_bucket_map_grid(values, bucket_labels, baseline_labels, save_path, kind, tod_label, model_label, eval_region):
    """Grid of per-pixel skill maps: rows = lead buckets, cols = baselines.

    values[bucket_label][baseline_label] = DataArray(lat, lon). Both 'improvement'
    (RMSE_baseline − RMSE_model, K) and 'skill' (1 − RMSE_model/RMSE_baseline) use a diverging
    RdBu map centred on 0 (positive = model beats baseline)."""
    plt.rcParams.update(RC)
    n_row, n_col = len(bucket_labels), len(baseline_labels)
    pool = [
        values[b][bl].values[np.isfinite(values[b][bl].values)].ravel()
        for b in bucket_labels
        for bl in baseline_labels
        if values[b].get(bl) is not None
    ]
    all_vals = np.concatenate(pool) if pool else np.array([1.0])
    vmax = float(np.nanpercentile(np.abs(all_vals), 95)) if all_vals.size else 1.0

    fig, axes = plt.subplots(
        n_row, n_col, figsize=(5.2 * n_col, 4.6 * n_row), subplot_kw={"projection": _PROJ}, squeeze=False
    )
    im = None
    for i, b in enumerate(bucket_labels):
        for j, bl in enumerate(baseline_labels):
            ax = axes[i][j]
            da = values[b].get(bl)
            if da is None:
                ax.set_visible(False)
                continue
            im = da.plot(ax=ax, cmap="RdBu", vmin=-vmax, vmax=vmax, add_colorbar=False, transform=_PROJ)
            _add_map_features(ax)
            ax.set_title(f"vs {bl}" if i == 0 else "")
            if j == 0:
                ax.text(-0.32, 0.5, b, transform=ax.transAxes, rotation=90, va="center", ha="center", fontweight="bold")

    fig.subplots_adjust(left=0.10, wspace=0.35, hspace=0.10, right=0.89)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    label = "RMSE(baseline) − RMSE(model) [K]" if kind == "improvement" else "Skill  1 − RMSE_model/RMSE_baseline"
    cbar.set_label(f"{label}\nPositive = {model_label} better", rotation=270, labelpad=30)
    fig.suptitle(f"{eval_region} — {kind} maps, {tod_label}  ({model_label})", y=1.0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


def lead_bucket_rmse_grid(values, bucket_labels, col_labels, save_path, tod_label, eval_region):
    """Grid of per-pixel RMSE maps: rows = lead buckets, cols = [model] + baselines (sequential cmap,
    shared 0→vmax). values[bucket_label][col_label] = DataArray(lat, lon)."""
    plt.rcParams.update(RC)
    n_row, n_col = len(bucket_labels), len(col_labels)
    pool = [
        values[b][c].values[np.isfinite(values[b][c].values)].ravel()
        for b in bucket_labels
        for c in col_labels
        if values[b].get(c) is not None
    ]
    all_vals = np.concatenate(pool) if pool else np.array([5.0])
    vmax = float(np.nanpercentile(all_vals, 98)) if all_vals.size else 5.0

    fig, axes = plt.subplots(
        n_row, n_col, figsize=(4.6 * n_col, 4.6 * n_row), subplot_kw={"projection": _PROJ}, squeeze=False
    )
    im = None
    for i, b in enumerate(bucket_labels):
        for j, c in enumerate(col_labels):
            ax = axes[i][j]
            da = values[b].get(c)
            if da is None:
                ax.set_visible(False)
                continue
            im = da.plot(ax=ax, cmap="viridis", vmin=0, vmax=vmax, add_colorbar=False, transform=_PROJ)
            _add_map_features(ax)
            ax.set_title(c if i == 0 else "")
            if j == 0:
                ax.text(-0.32, 0.5, b, transform=ax.transAxes, rotation=90, va="center", ha="center", fontweight="bold")

    fig.subplots_adjust(left=0.10, wspace=0.35, hspace=0.10, right=0.89)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("RMSE [K]", rotation=270, labelpad=18)
    fig.suptitle(f"{eval_region} — RMSE maps, {tod_label}", y=1.0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


def improvement_map(
    rmse_model: xr.DataArray,
    rmse_baseline: xr.DataArray,
    save_path: Path,
    model_label: str = "MLP-Rollout",
    baseline_label: str = "ERA5",
    eval_region: str = "",
    vmax: float = None,
    title: str = None,
) -> plt.Figure:
    """Single map of RMSE(baseline) − RMSE(model): positive = model better.

    Parameters
    ----------
    rmse_model : xr.DataArray
        Per-pixel RMSE of the model forecast, dims (lat, lon).
    rmse_baseline : xr.DataArray
        Per-pixel RMSE of the baseline (ERA5), dims (lat, lon).
    save_path : Path
        Output file path.
    model_label, baseline_label : str
        Used in title and colorbar.
    eval_region : str
        Used in figure title.
    vmax : float, optional
        Symmetric colorbar limit. If None, derived from data.
    title : str, optional
        Override figure suptitle.

    Returns
    -------
    plt.Figure
    """
    plt.rcParams.update(RC)

    improvement = rmse_baseline - rmse_model  # positive = model better

    if vmax is None:
        vals = improvement.values[np.isfinite(improvement.values)]
        vmax = float(np.nanpercentile(np.abs(vals), 95)) if len(vals) else 2.0
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": _PROJ})
    im = improvement.plot(
        ax=ax,
        cmap="RdBu",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        transform=_PROJ,
    )
    _add_map_features(ax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(
        f"RMSE({baseline_label}) − RMSE({model_label})  (K)\nPositive = {model_label} better",
        rotation=270,
        labelpad=22,
    )

    suptitle = title or (
        f"Improvement: {model_label} vs {baseline_label}" + (f" — {eval_region}" if eval_region else "")
    )
    ax.set_title(suptitle)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig
