from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.plots._design import RC


def regression_parity(
    ds: xr.Dataset,
    variable: str,
    save_dir: Path,
    init_time: int = 0,
    n_lead_times: int = None,
) -> None:
    """Parity scatter plot coloured by lead time.

    Args:
        ds: ForecastResult Dataset with dims (init_time, lead_time, lat, lon).
        variable: Variable name, e.g. "LST".
        save_dir: Directory in which to save the PNG.
        init_time: Integer index into init_time dimension.
        n_lead_times: Number of lead times to include (None = all).
    """
    plt.rcParams.update(RC)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lead_times = ds.lead_time if n_lead_times is None else ds.lead_time[:n_lead_times]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lead_times)))

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, lt in enumerate(lead_times):
        ax.scatter(
            ds[f"{variable}_obs"].isel(init_time=init_time, lead_time=i).values.ravel(),
            ds[f"{variable}_pred"].isel(init_time=init_time, lead_time=i).values.ravel(),
            c=[colors[i]],
            label=f"LT {i}",
            alpha=0.4,
            s=10,
        )

    lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("SEVIRI")
    ax.set_ylabel("Forecast")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"{variable}_regression_parity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plume_diagram(
    ds: xr.Dataset,
    variable: str,
    save_dir: Path,
    n_pixels: int = 10,
    n_init_times: int = 4,
    seed: int = 42,
) -> None:
    """Forecast distribution boxplots vs observation line per valid time.

    Args:
        ds: ForecastResult Dataset.
        variable: Variable name.
        save_dir: Directory in which to save the PNG.
        n_pixels: Number of random pixels to sample.
        n_init_times: Number of init times to include.
        seed: Random seed.
    """
    plt.rcParams.update(RC)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    init_times = ds.init_time.values[:n_init_times]
    lead_times = pd.TimedeltaIndex(ds.lead_time.values)

    valid_mask = np.isfinite(ds[f"{variable}_pred"].sel(init_time=init_times)).all(
        dim=["init_time", "lead_time"]
    ) & np.isfinite(ds[f"{variable}_obs"].sel(init_time=init_times)).all(dim=["init_time", "lead_time"])
    valid_lats, valid_lons = np.where(valid_mask.values)
    if len(valid_lats) == 0:
        print("No fully valid pixels found for plume_diagram.")
        return

    rng = np.random.default_rng(seed)
    n_pixels = min(n_pixels, len(valid_lats))
    chosen = rng.choice(len(valid_lats), size=n_pixels, replace=False)
    pixel_lat_idx = valid_lats[chosen]
    pixel_lon_idx = valid_lons[chosen]

    valid_time_data: dict = {}
    for li, lo in zip(pixel_lat_idx, pixel_lon_idx):
        for it in init_times:
            for j, lt in enumerate(lead_times):
                vt = pd.Timestamp(it) + lt
                pred_val = float(ds[f"{variable}_pred"].sel(init_time=it).isel(lead_time=j, lat=li, lon=lo))
                obs_val = float(ds[f"{variable}_obs"].sel(init_time=it).isel(lead_time=j, lat=li, lon=lo))
                if np.isfinite(pred_val):
                    if vt not in valid_time_data:
                        valid_time_data[vt] = {"forecasts": [], "obs": []}
                    valid_time_data[vt]["forecasts"].append(pred_val)
                    if np.isfinite(obs_val):
                        valid_time_data[vt]["obs"].append(obs_val)

    sorted_times = sorted(valid_time_data)
    forecast_distributions = [valid_time_data[vt]["forecasts"] for vt in sorted_times]
    obs_means = [np.mean(valid_time_data[vt]["obs"]) if valid_time_data[vt]["obs"] else np.nan for vt in sorted_times]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        forecast_distributions,
        positions=range(len(sorted_times)),
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="darkblue", linewidth=2),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue"),
    )
    ax.plot(
        range(len(sorted_times)),
        obs_means,
        "o-",
        color="grey",
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="Observations (pixel avg)",
        zorder=3,
    )
    ax.set_xticks(range(len(sorted_times)))
    ax.set_xticklabels([vt.strftime("%Y-%m-%d %H:%M") for vt in sorted_times], rotation=45, ha="right")
    ax.set_xlabel("Validation Time")
    ax.set_ylabel(variable)
    ax.set_title(f"Forecast distribution — {n_pixels} pixels, {n_init_times} init times")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / f"{variable}_plume_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()


def error_vs_leadtime(
    ds: xr.Dataset,
    variable: str,
    save_dir: Path,
    init_time: int = 0,
    n_pixels: int = 500,
    seed: int = 42,
    bootstrap_uncertainty: bool = True,
) -> None:
    """MAE vs lead time with optional bootstrap uncertainty bands.

    Args:
        ds: ForecastResult Dataset.
        variable: Variable name.
        save_dir: Directory in which to save the PNG.
        init_time: Integer index into init_time dimension.
        n_pixels: Number of random pixels to sample.
        seed: Random seed.
        bootstrap_uncertainty: If True, add 95% bootstrap confidence bands.
    """
    plt.rcParams.update(RC)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    obs = ds[f"{variable}_obs"].isel(init_time=init_time)
    pred = ds[f"{variable}_pred"].isel(init_time=init_time)

    valid_mask = (np.isfinite(obs) & np.isfinite(pred)).all(dim="lead_time")
    valid_lats, valid_lons = np.where(valid_mask.values)
    if len(valid_lats) == 0:
        print("No fully valid pixels found for error_vs_leadtime.")
        return

    rng = np.random.default_rng(seed)
    n_pixels = min(n_pixels, len(valid_lats))
    chosen = rng.choice(len(valid_lats), size=n_pixels, replace=False)
    lat_idx = valid_lats[chosen]
    lon_idx = valid_lons[chosen]

    obs_vals = obs.values[:, lat_idx, lon_idx]
    pred_vals = pred.values[:, lat_idx, lon_idx]
    abs_err = np.abs(pred_vals - obs_vals)
    mae_vals = abs_err.mean(axis=1)

    mae_std = np.zeros(len(ds.lead_time))
    if bootstrap_uncertainty:
        n_boot = 1000
        mae_boot = np.zeros((len(ds.lead_time), n_boot))
        for b in range(n_boot):
            idx = rng.choice(n_pixels, size=n_pixels, replace=True)
            mae_boot[:, b] = abs_err[:, idx].mean(axis=1)
        mae_std = mae_boot.std(axis=1)

    lead_x = np.arange(len(ds.lead_time))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lead_x, mae_vals, "s-", color="tab:blue", label="MAE")
    if bootstrap_uncertainty:
        ax.fill_between(lead_x, mae_vals - 1.96 * mae_std, mae_vals + 1.96 * mae_std, color="tab:blue", alpha=0.2)

    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("LST Error (K)")
    init_time_str = pd.Timestamp(ds.init_time.values[init_time]).strftime("%Y-%m-%d %H:%M")
    ax.set_title(f"Init: {init_time_str} — n={n_pixels} pixels")
    ax.set_xticks(lead_x)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"{variable}_error_vs_leadtime_it{init_time}.png", dpi=150, bbox_inches="tight")
    plt.close()
