"""Power spectrum plots for forecast evaluation.

Spatial power spectrum (grid data only)
----------------------------------------
spatial_power_spectrum_plot(spectra_by_case, obs_spectrum, ...)
    Log-log plot of spatially-averaged power spectral density vs. wavelength.
    Each case contributes one predicted spectrum; observed is plotted once.

How the spatial spectrum is computed
-------------------------------------
For each time step:
  1. Fill NaNs with the field mean (avoids spectral artefacts at cloud gaps).
  2. Subtract the spatial mean (remove DC component).
  3. Apply a 2D Hann window (reduce spectral leakage at domain edges).
  4. Compute the 2D FFT and power = |FFT|² / (Ny × Nx).
  5. Radially average into 1-D bins of wavenumber k = sqrt(kx²+ky²).
Steps 1-5 are repeated for each time step and averaged.
X-axis: wavelength in km, derived from the lat/lon grid spacing.
Y-axis: power spectral density (K² per wavenumber).

Temporal power spectrum (grid and sparse data)
-----------------------------------------------
temporal_power_spectrum_plot(spectra_by_case, obs_spectrum, ...)
    Log-log plot of pixel-averaged power spectral density vs. period in hours.
    Works on both sparse (n_time, n_pixel) and grid (n_time, ny, nx) data.

How the temporal spectrum is computed
--------------------------------------
Spatial dims are flattened to (n_time, n_pixel). For each pixel:
  1. Fill NaNs with pixel mean (handles cloud gaps).
  2. Subtract the temporal mean (remove DC component).
  3. Apply a 1D Hann window (reduce spectral leakage at series edges).
  4. Compute 1D rfft and power = |FFT|² / (N × window_norm).
Power is averaged across all valid pixels.
X-axis: period in hours (1 / frequency). Key diurnal periods are marked.
Y-axis: power spectral density (K² per frequency unit).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from deep_lst.evaluation.plots._design import (
    BASELINE_LINESTYLE,
    CASE_COLORS,
    CASE_LINESTYLES,
    DEFAULT_COLOR,
    OBS_COLOR,
    OBS_LINESTYLE,
    RC,
)

_OBS_COLOR = OBS_COLOR
_DEFAULT_COLOR = DEFAULT_COLOR


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _grid_spacing_km(lats: np.ndarray, lons: np.ndarray) -> float:
    """Estimate mean grid spacing in km from 1-D lat/lon arrays."""
    dlat = np.abs(np.diff(lats)).mean()
    dlon = np.abs(np.diff(lons)).mean()
    lat_c = np.deg2rad(np.mean(lats))
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(lat_c)
    return float(0.5 * (dlat * km_per_deg_lat + dlon * km_per_deg_lon))


def compute_spatial_power_spectrum(
    fields: np.ndarray,
    dx_km: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radially-averaged 1-D spatial power spectrum.

    Parameters
    ----------
    fields : ndarray, shape (n_time, ny, nx)
        LST values at each time step.  NaN = cloud / missing.
    dx_km : float
        Approximate grid spacing in km (used to set x-axis scale).

    Returns
    -------
    wavelengths_km : ndarray, shape (n_bins,)
        Wavelength in km corresponding to each spectral bin.
    power : ndarray, shape (n_bins,)
        Mean power spectral density (K² per wavenumber unit).
    """
    n_time, ny, nx = fields.shape
    n_bins = min(ny, nx) // 2

    # Radial wavenumber grid (cycles / pixel), then convert to cycles / km
    ky = np.fft.fftshift(np.fft.fftfreq(ny))  # cycles / pixel
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)  # cycles / pixel

    k_max = 0.5  # Nyquist
    k_edges = np.linspace(0.0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    # 2-D Hann window
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    window = np.outer(wy, wx)
    window_norm = np.mean(window**2)  # normalise for windowing loss

    power_accum = np.zeros(n_bins)
    n_valid = 0

    for t in range(n_time):
        f = fields[t]
        if np.isnan(f).all():
            continue

        # Fill NaNs and remove spatial mean
        f = np.where(np.isnan(f), np.nanmean(f), f)
        f = f - f.mean()

        # FFT → power
        fft2 = np.fft.fftshift(np.fft.fft2(f * window))
        power2d = (np.abs(fft2) ** 2) / (ny * nx * window_norm)

        # Radial average
        for i in range(n_bins):
            mask = (K >= k_edges[i]) & (K < k_edges[i + 1])
            if mask.sum() > 0:
                power_accum[i] += power2d[mask].mean()

        n_valid += 1

    if n_valid == 0:
        return np.array([]), np.array([])

    power_mean = power_accum / n_valid

    # Convert wavenumber from cycles/pixel → cycles/km, then to wavelength
    k_km = k_centers / dx_km  # cycles / km
    # avoid division by zero at k=0
    valid = k_km > 0
    wavelengths_km = np.full_like(k_km, np.nan)
    wavelengths_km[valid] = 1.0 / k_km[valid]

    return wavelengths_km, power_mean


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def spatial_power_spectrum_plot(
    pred_spectra_by_case: dict[str, tuple[np.ndarray, np.ndarray]],
    obs_spectrum: tuple[np.ndarray, np.ndarray] | None = None,
    var: str = "LST",
    save_path: Path = None,
    title: str = None,
    eval_region: str = "",
    ylabel: str | None = None,
) -> plt.Figure:
    """Log-log spatial power spectrum: observed vs. each forecast case.

    Parameters
    ----------
    pred_spectra_by_case : dict
        ``{case_label: (wavelengths_km, power)}`` — output of
        ``compute_spatial_power_spectrum`` for each model's predictions.
    obs_spectrum : tuple, optional
        ``(wavelengths_km, power)`` for the observed LST. If None (residual
        mode), the obs reference line is omitted.
    ylabel : str, optional
        Y-axis label. Defaults to "Power spectral density ({var}², K²)".
    """
    plt.rcParams.update(RC)

    fig, ax = plt.subplots(figsize=(8, 5))

    if obs_spectrum is not None:
        wav_obs, pwr_obs = obs_spectrum
        valid = np.isfinite(wav_obs) & np.isfinite(pwr_obs) & (pwr_obs > 0)
        ax.loglog(
            wav_obs[valid], pwr_obs[valid], color=_OBS_COLOR, linestyle=OBS_LINESTYLE, linewidth=2, label="SEVIRI obs"
        )

    for case, (wav, pwr) in pred_spectra_by_case.items():
        color = CASE_COLORS.get(case, _DEFAULT_COLOR)
        ls = CASE_LINESTYLES.get(case, BASELINE_LINESTYLE)
        valid = np.isfinite(wav) & np.isfinite(pwr) & (pwr > 0)
        ax.loglog(wav[valid], pwr[valid], color=color, linestyle=ls, linewidth=2, label=case, alpha=0.85)

    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel(ylabel or "Power spectral density")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_xaxis()  # large scales (long wavelengths) on the left

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Temporal power spectrum
# ---------------------------------------------------------------------------


def compute_temporal_power_spectrum(
    fields: np.ndarray,
    dt_hours: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1-D temporal power spectrum, averaged over pixels.

    Parameters
    ----------
    fields : ndarray, shape (n_time, ...)
        LST values. Any spatial dims beyond the first are flattened.
    dt_hours : float
        Time step in hours (used to convert frequency → period).

    Returns
    -------
    periods_hours : ndarray, shape (n_bins,)
        Period in hours for each spectral bin (NaN at DC / zero frequency).
    power : ndarray, shape (n_bins,)
        Mean power spectral density (K² per frequency unit), averaged over pixels.
    """
    n_time = fields.shape[0]
    pixels = fields.reshape(n_time, -1)  # (n_time, n_pixel)

    window = np.hanning(n_time)
    window_norm = np.mean(window**2)

    freqs = np.fft.rfftfreq(n_time, d=dt_hours)  # cycles / hour
    n_bins = len(freqs)

    power_accum = np.zeros(n_bins)
    n_valid = 0

    for col in pixels.T:
        if np.isnan(col).all():
            continue
        col = np.where(np.isnan(col), np.nanmean(col), col)
        col = col - col.mean()
        fft1d = np.fft.rfft(col * window)
        power_accum += (np.abs(fft1d) ** 2) / (n_time * window_norm)
        n_valid += 1

    if n_valid == 0:
        return np.array([]), np.array([])

    power_mean = power_accum / n_valid

    periods_hours = np.full_like(freqs, np.nan)
    valid = freqs > 0
    periods_hours[valid] = 1.0 / freqs[valid]

    return periods_hours, power_mean


def temporal_power_spectrum_plot(
    pred_spectra_by_case: dict[str, tuple[np.ndarray, np.ndarray]],
    obs_spectrum: tuple[np.ndarray, np.ndarray] | None = None,
    var: str = "LST",
    save_path: Path = None,
    eval_region: str = "",
    ylabel: str | None = None,
) -> plt.Figure:
    """Log-log temporal power spectrum: observed vs. each forecast case.

    Parameters
    ----------
    pred_spectra_by_case : dict
        ``{case_label: (periods_hours, power)}`` — output of
        ``compute_temporal_power_spectrum`` for each model's predictions.
    obs_spectrum : tuple, optional
        ``(periods_hours, power)`` for the observed LST. If None (residual
        mode), the obs reference line is omitted.
    ylabel : str, optional
        Y-axis label. Defaults to "Power spectral density ({var}², K²)".
    """
    plt.rcParams.update(RC)

    fig, ax = plt.subplots(figsize=(8, 5))

    if obs_spectrum is not None:
        wav_obs, pwr_obs = obs_spectrum
        valid = np.isfinite(wav_obs) & np.isfinite(pwr_obs) & (pwr_obs > 0)
        ax.plot(
            wav_obs[valid], pwr_obs[valid], color=_OBS_COLOR, linestyle=OBS_LINESTYLE, linewidth=2, label="SEVIRI LST"
        )

    for case, (wav, pwr) in pred_spectra_by_case.items():
        color = CASE_COLORS.get(case, _DEFAULT_COLOR)
        ls = CASE_LINESTYLES.get(case, BASELINE_LINESTYLE)
        valid = np.isfinite(wav) & np.isfinite(pwr) & (pwr > 0)
        ax.plot(wav[valid], pwr[valid], color=color, linestyle=ls, linewidth=2, label="Deep LST", alpha=0.85)

    # Mark key diurnal periods
    for period, label in [(24, "24 h"), (12, "12 h"), (8, "8 h")]:
        ax.axvline(period, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(
            period, 1.02, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="gray"
        )

    ax.set_xlabel("Period (hours)")
    ax.set_ylabel(ylabel or "Power spectral density")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_xaxis()  # long periods on the left

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Convenience wrapper: compute + plot directly from forecast datasets
# ---------------------------------------------------------------------------


def power_spectrum_from_forecasts(
    forecast_by_case: dict[str, xr.Dataset],
    var: str = "LST",
    lead_time: str = "3h",
    save_path: Path = None,
    eval_region: str = "",
) -> plt.Figure:
    """Compute spectra from raw forecast datasets and plot.

    Parameters
    ----------
    forecast_by_case : dict
        ``{case_label: forecast_xr.Dataset}`` — loaded forecast.nc files.
        Each dataset has dims (init_time, lead_time, lat, lon).
    lead_time : str
        Which lead time slice to use, e.g. "3h".
    """
    import pandas as pd

    pred_spectra = {}
    obs_spectrum = None

    for case, ds in forecast_by_case.items():
        lt = pd.Timedelta(lead_time)
        pred = ds[f"{var}_pred"].sel(lead_time=lt).values  # (n_time, lat, lon)
        obs = ds[f"{var}_obs"].sel(lead_time=lt).values

        lats = ds["lat"].values
        lons = ds["lon"].values
        dx_km = _grid_spacing_km(lats, lons)

        wav_pred, pwr_pred = compute_spatial_power_spectrum(pred, dx_km=dx_km)
        pred_spectra[case] = (wav_pred, pwr_pred)

        # Observed: compute once, reuse (overwriting is fine — same obs)
        wav_obs, pwr_obs = compute_spatial_power_spectrum(obs, dx_km=dx_km)
        obs_spectrum = (wav_obs, pwr_obs)

    return spatial_power_spectrum_plot(
        pred_spectra_by_case=pred_spectra,
        obs_spectrum=obs_spectrum,
        var=var,
        save_path=save_path,
        eval_region=eval_region,
    )
