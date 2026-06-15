import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

CLIM_PATH = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/climatology/hourly/climatology_train_KG_L.nc"
ZARR_PATH = "/Net/Groups/BGI/tscratch/mwessel/wegen-stores/sampled_stores/seviri_sampled_KG_L.zarr"
OUT_PATH = "/Net/Groups/BGI/people/mwessel/deep_lst/results/exploration/climatology.png"

PERIOD = slice("2024-11-04", "2024-11-17")

VALID_FLAGS = [101.0, 102.0, 301.0, 302.0]
INVALID_FLAGS = [103.0]

# --- climatology ---
clim = xr.open_dataset(CLIM_PATH)
clim_ts = clim["LST"].sel(valid_time=PERIOD).mean("pixel")

# --- observed ---
ds = xr.open_zarr(ZARR_PATH)
lst = ds["LST"].sel(time_idx=PERIOD)
qf = ds["quality_flag"].sel(time_idx=PERIOD)

lst_valid = lst.where(qf.isin(VALID_FLAGS)).mean("spatial_idx").compute()
lst_invalid = lst.where(qf.isin(INVALID_FLAGS)).mean("spatial_idx").compute()

# --- plot ---
fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(clim_ts["valid_time"].values, clim_ts.values, color="gray", linestyle="--", linewidth=1.2, label="Climatology")
ax.plot(lst_valid["time_idx"].values, lst_valid.values, color="red", linewidth=1.0, label="Observed valid (QF 1–2)")
ax.plot(
    lst_invalid["time_idx"].values,
    lst_invalid.values,
    color="lightblue",
    linewidth=1.0,
    label="Observed degraded (QF 3)",
)

ax.set_ylabel("LST (K)")
ax.set_title("LST climatology vs. observed — test_period_22 (2024-11-04 → 2024-11-17)")
ax.legend()
fig.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.show()


print(np.sqrt(np.mean((lst_valid - clim_ts) ** 2)))
print(np.mean(abs(lst_valid - clim_ts)))
