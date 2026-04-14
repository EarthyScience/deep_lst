"""
Quality-control checks for sampled zarr stores.

Three checks
------------
1. Round-trip:   a random subset of pixels, values in sampled zarr == values in full zarr.
2. Coordinates:  spatial_idx array in sampled zarr matches npz source indices.
3. Spatial match: for each SEVIRI pixel, the mapped ERA5 pixel is within grid-spacing distance.

Additionally produces two diagnostic plots:
  - Scatter of pixel lat/lon coloured by modality (spatial coverage)
  - Time-series of SEVIRI LST and ERA5 SKT for 3 co-located pixels

Usage
-----
python qc_sampled_zarr.py
"""

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import zarr

# ── paths ─────────────────────────────────────────────────────────────────────
STORES = "/Net/Groups/BGI/tscratch/mwessel/wegen-stores-new"
ASSETS = "/Net/Groups/BGI/people/mwessel/deep_lst/assets"

SEV_FULL = f"{STORES}/seviri_long.zarr"
ERA5_FULL = f"{STORES}/era5_long.zarr"
SEV_SAMPLED = f"{STORES}/seviri_sampled_KGLC_L.zarr"
ERA5_SAMPLED = f"{STORES}/era5_sampled_KGLC_L.zarr"

SEV_IDX_FILE = f"{ASSETS}/seviri/spatial/indices_KGLC_L.npz"
ERA5_IDX_FILE = f"{ASSETS}/era5/spatial/indices_KGLC_L.npz"

SEV_COL_EXTENT = 3201
ERA5_COL_EXTENT = 1440

N_SPOT_CHECK = 20  # pixels to round-trip check
RNG = np.random.default_rng(42)


# ── helpers ───────────────────────────────────────────────────────────────────


def _rank_lookup(stored_indices: np.ndarray) -> dict:
    return {int(idx): pos for pos, idx in enumerate(stored_indices)}


def _grid_coords(spatial_indices, col_extent, lat_arr, lon_arr):
    li = spatial_indices // col_extent
    loi = spatial_indices % col_extent
    return lat_arr[li], lon_arr[loi]


# ── load stores ───────────────────────────────────────────────────────────────
print("Opening zarr stores...", flush=True)
sev_full = zarr.open(SEV_FULL, mode="r")
era5_full = zarr.open(ERA5_FULL, mode="r")
sev_samp = zarr.open(SEV_SAMPLED, mode="r")
era5_samp = zarr.open(ERA5_SAMPLED, mode="r")

sev_idx = np.load(SEV_IDX_FILE)
era5_idx = np.load(ERA5_IDX_FILE)

# New format: coordinate array is named 'spatial_idx' (flat source-grid positions)
sev_stored = np.asarray(sev_samp["spatial_idx"])  # (1500,) SEVIRI flat
era5_stored = np.asarray(era5_samp["spatial_idx"])  # (1500,) ERA5 flat

sev_rank = _rank_lookup(sev_stored)
era5_rank = _rank_lookup(era5_stored)

sev_lat_full = np.asarray(sev_full["latitude"])
sev_lon_full = np.asarray(sev_full["longitude"])
era5_lat_full = np.asarray(era5_full["latitude"])
era5_lon_full = np.asarray(era5_full["longitude"])

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


# ══════════════════════════════════════════════════════════════════════════════
# Check 1: Round-trip — sampled values match full zarr
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Check 1: Round-trip value match ──")

checks = [
    ("SEVIRI", "LST", sev_full, sev_samp, sev_stored, SEV_COL_EXTENT),
    ("ERA5", "skt", era5_full, era5_samp, era5_stored, ERA5_COL_EXTENT),
]

for modality, var, full, samp, stored, col_ext in checks:
    picks = RNG.choice(len(stored), size=N_SPOT_CHECK, replace=False)
    spatial_flat = stored[picks]
    li = (spatial_flat // col_ext).astype(np.intp)
    loi = (spatial_flat % col_ext).astype(np.intp)

    T_CHECK = 48
    t_start = RNG.integers(0, full[var].shape[0] - T_CHECK)
    t_sl = slice(int(t_start), int(t_start) + T_CHECK)

    # Values from full zarr at those spatial positions (loop to avoid fancy indexing)
    full_vals = np.stack(
        [np.asarray(full[var][t_sl, int(li[i]), int(loi[i])]) for i in range(N_SPOT_CHECK)],
        axis=1,
    )  # (T_CHECK, N_SPOT_CHECK)

    # Values from sampled zarr — new layout is (spatial_idx, time_idx) = (N, T)
    ranks = np.array(
        [sev_rank[int(s)] if modality == "SEVIRI" else era5_rank[int(s)] for s in spatial_flat], dtype=np.intp
    )
    samp_vals = np.stack(
        [np.asarray(samp[var][int(ranks[i]), t_sl]) for i in range(N_SPOT_CHECK)],
        axis=1,
    )  # (T_CHECK, N_SPOT_CHECK)

    # Compare (ignore NaN pairs)
    mask = np.isfinite(full_vals) & np.isfinite(samp_vals)
    if mask.sum() == 0:
        print(f"  {modality} {var}: all NaN in window — try larger T_CHECK or different pixel")
        continue
    match = np.allclose(full_vals[mask], samp_vals[mask], rtol=0, atol=0)
    max_diff = np.abs(full_vals[mask] - samp_vals[mask]).max()
    status = PASS if match else FAIL
    print(
        f"  {modality} {var}: {status}  max_diff={max_diff:.6f}  "
        f"(checked {mask.sum()} finite values across {N_SPOT_CHECK} pixels, {T_CHECK} timesteps)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Check 2: Coordinate consistency — spatial_idx in sampled store matches npz
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Check 2: spatial_idx coordinate consistency ──")

# SEVIRI: src_indices in npz should match spatial_idx in sampled store (same set, possibly reordered)
sev_npz_src = sev_idx["src_indices"].astype(np.int64)
era5_npz_src = np.unique(era5_idx["mapping"]).astype(np.int64)

for modality, stored, npz_src in [
    ("SEVIRI", sev_stored, sev_npz_src),
    ("ERA5", era5_stored, era5_npz_src),
]:
    set_stored = set(stored.tolist())
    set_npz = set(npz_src.tolist())
    same_set = set_stored == set_npz
    n_only_stored = len(set_stored - set_npz)
    n_only_npz = len(set_npz - set_stored)
    status = PASS if same_set else FAIL
    print(
        f"  {modality}: {status}  stored={len(stored)}  npz={len(npz_src)}  "
        f"only-in-stored={n_only_stored}  only-in-npz={n_only_npz}"
    )

    # Also verify lat/lon are physically plausible
    col_ext = SEV_COL_EXTENT if modality == "SEVIRI" else ERA5_COL_EXTENT
    lat_arr = sev_lat_full if modality == "SEVIRI" else era5_lat_full
    lon_arr = sev_lon_full if modality == "SEVIRI" else era5_lon_full
    lats, lons = _grid_coords(stored, col_ext, lat_arr, lon_arr)
    print(f"    lat range: [{lats.min():.2f}, {lats.max():.2f}]  lon range: [{lons.min():.2f}, {lons.max():.2f}]")


# ══════════════════════════════════════════════════════════════════════════════
# Check 3: SEVIRI↔ERA5 spatial match — mapped ERA5 pixel is close to SEVIRI pixel
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Check 3: SEVIRI↔ERA5 spatial proximity ──")

# era5_idx has correctly paired (src_indices[i], mapping[i]) — SEVIRI→ERA5 pairs.
# sev_idx['src_indices'] is the same pixel set but may be in a different order,
# so we must use era5_idx['src_indices'] as the SEVIRI side to keep pairs aligned.
sev_src = era5_idx["src_indices"]  # SEVIRI flat indices, paired with mapping below
era5_map = era5_idx["mapping"]  # corresponding ERA5 flat indices, (1500,)

sev_lat_s, sev_lon_s = _grid_coords(sev_src, SEV_COL_EXTENT, sev_lat_full, sev_lon_full)
era5_lat_s, era5_lon_s = _grid_coords(era5_map, ERA5_COL_EXTENT, era5_lat_full, era5_lon_full)

lat_dist = np.abs(sev_lat_s - era5_lat_s)
lon_dist = np.abs(sev_lon_s - era5_lon_s)
dist_deg = np.sqrt(lat_dist**2 + lon_dist**2)

ERA5_SPACING = 0.25  # degrees
within = (dist_deg <= ERA5_SPACING).mean() * 100
status = PASS if within > 99 else FAIL
print(f"  {status}  {within:.1f}% of SEVIRI pixels mapped within {ERA5_SPACING}° of ERA5 pixel")
print(f"  max distance: {dist_deg.max():.4f}°  median: {np.median(dist_deg):.4f}°")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots...")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot A: spatial coverage ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.scatter(sev_lon_s, sev_lat_s, s=8, c="steelblue", alpha=0.6, label="SEVIRI pixels")
ax.scatter(era5_lon_s, era5_lat_s, s=8, c="tomato", alpha=0.6, label="ERA5 pixels (mapped)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Spatial coverage: SEVIRI vs ERA5 sample positions")
ax.legend(markerscale=2, fontsize=8)

# ── Plot B: SEVIRI–ERA5 per-pixel spatial distance ────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.hist(dist_deg, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(ERA5_SPACING / 2, color="tomato", linestyle="--", label=f"ERA5 half-spacing ({ERA5_SPACING / 2}°)")
ax.set_xlabel("Distance SEVIRI → mapped ERA5 pixel (°)")
ax.set_ylabel("Count")
ax.set_title("SEVIRI↔ERA5 mapping proximity")
ax.legend(fontsize=8)

# ── Plot C & D: time series for 3 co-located pixels ──────────────────────────
# Pick 3 pixels with low NaN fraction in SEVIRI LST
# New layout: (N, T) — axis 0 = pixels, axis 1 = time
sev_lst_arr = sev_samp["LST"]
era5_skt_arr = era5_samp["skt"]

# daily sampling of the first year: take every 24th timestep along time axis
nan_frac = np.isnan(np.asarray(sev_lst_arr[:, : 365 * 24 : 24])).mean(axis=1)  # per pixel
good_pixels = np.argsort(nan_frac)[:3]  # 3 pixels with fewest NaNs

# Build SEVIRI→ERA5 rank mapping
sev_to_era5_flat = dict(zip(sev_idx["src_indices"].tolist(), era5_idx["mapping"].tolist()))
era5_ranks_for_sev = np.array(
    [era5_rank[sev_to_era5_flat[int(sev_stored[r])]] for r in range(len(sev_stored))], dtype=np.intp
)

# Take one year of data (hourly)
T_PLOT = 24 * 365
t_sl = slice(0, T_PLOT)
times = np.arange(T_PLOT)

ax_ts = fig.add_subplot(gs[1, :])
colors = ["steelblue", "forestgreen", "darkorange"]
for i, (rank_sev, col) in enumerate(zip(good_pixels, colors)):
    # New (N, T) layout: arr[rank, t_sl]
    lst = np.asarray(sev_lst_arr[rank_sev, t_sl]) - 273.15  # K → °C
    skt = np.asarray(era5_skt_arr[era5_ranks_for_sev[rank_sev], t_sl]) - 273.15

    sev_flat = int(sev_stored[rank_sev])
    lat = float(sev_lat_full[sev_flat // SEV_COL_EXTENT])
    lon = float(sev_lon_full[sev_flat % SEV_COL_EXTENT])
    label_base = f"Pixel {i + 1}  ({lat:.1f}°N, {lon:.1f}°E)"

    ax_ts.plot(times, lst, color=col, alpha=0.7, lw=0.6, label=f"{label_base} — SEVIRI LST")
    ax_ts.plot(times, skt, color=col, alpha=0.4, lw=0.6, linestyle="--", label=f"{label_base} — ERA5 SKT")

ax_ts.set_xlabel("Hours from start of store")
ax_ts.set_ylabel("Temperature (°C)")
ax_ts.set_title("SEVIRI LST (solid) vs ERA5 SKT (dashed) — first year, 3 pixels")
ax_ts.legend(fontsize=7, ncol=2)

out_path = "/Net/Groups/BGI/people/mwessel/deep_lst/plots/qc/qc_sampled_zarr.png"

os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out_path}")
plt.close()

print("\nQC complete.")
