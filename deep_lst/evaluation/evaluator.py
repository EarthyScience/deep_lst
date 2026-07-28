import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.metrics import to_valid_time
from deep_lst.evaluation.products import align_baseline, compute_diagnostic_products, compute_products
from deep_lst.evaluation.spatial import pixel_grid
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.spatial import store_grid

_BASELINE_REGISTRY: dict[str, tuple[callable, str]] = {
    # era5: NOT re-extractable (the era5 store carries no skt); existing files are stale
    # (float32 coords, pre-fix extraction). era5_land is the reference reanalysis baseline.
    "era5": (lambda p, n: p.baseline_path(n), "skt"),
    "era5_land": (lambda p, n: p.modality_baseline_path("era5_land", n), "skt"),
    "clim_hourly": (lambda p, n: p.climatology_path(n, "hourly"), "LST"),
    "clim_smooth": (lambda p, n: p.climatology_path(n, "smoothed_hourly"), "LST"),
    "persistence": (lambda p, n: p.persistence_path(n), "LST"),
}

# Single source of truth for what gets computed and plotted.
# Change these tuples to add/remove metrics across the whole pipeline.
RAW_METRICS = ("mae", "rmse", "bias", "nse")
ANOMALY_METRICS = ("acc", "activity")
CLIMATE_BASELINES = ("clim_smooth", "clim_hourly")


class Evaluator:
    """Shared loading and the product contract: load() / load_baseline() / compute() / materialise().

    Finished products land in self.products / self.diag_products (see
    evaluation/products.py for both catalogues).
    """

    def __init__(
        self,
        experiment,
        version,
        eval_name,
        var,
        baseline,
        temporal,
        aggregation,
        consistent_mask,
        overwrite,
    ):
        self.experiment = experiment
        self.version = version
        self.eval_name = eval_name
        self.var = var
        self.baseline = baseline
        self.overwrite = overwrite
        self.temporal = temporal
        self.aggregation = aggregation
        self.consistent_mask = consistent_mask
        self.paths = ExperimentPaths()

    def load(self) -> None:
        """Load the forecast, apply the temporal filter, attach class coords. No computation."""
        self.forecast_ds = self._attach_class_coords(self._filter(self._load_forecast()))
        self.pred = self.forecast_ds[f"{self.var}_pred"]
        self.obs = self.forecast_ds[f"{self.var}_obs"]

    def load_baseline(self, name: str) -> xr.DataArray:
        """Aligned (init_time, lead_time, space) cube for a registry baseline, cached per name."""
        if not hasattr(self, "_baseline_cubes"):
            self._baseline_cubes = {}
        if name in self._baseline_cubes:
            return self._baseline_cubes[name]
        if name not in _BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline: {name!r}. Available: {list(_BASELINE_REGISTRY)}")
        path_fn, var_name = _BASELINE_REGISTRY[name]
        path = path_fn(self.paths, self.eval_name)
        if not path.exists():
            raise FileNotFoundError(f"Baseline '{name}' not found: {path}")
        baseline = xr.open_dataset(path, chunks={})[var_name]
        baseline = self._subset_pixels(baseline, self.pred.isel(lead_time=0))
        self._check_baseline_coords(baseline, name)
        cube = align_baseline(baseline, self.pred)
        if not bool(cube.isel(init_time=0).notnull().any()):
            print(f"  [warn] baseline '{name}' aligned to no data at the first init — check grid/time coverage")
        self._baseline_cubes[name] = cube
        return cube

    def compute(
        self,
        products: list,
        baselines: tuple = (),
        clim_baseline: str | None = None,
        drop_lead0: bool = True,
    ) -> dict:
        """Evaluate the named PRODUCTS on shared stats tables; stores and returns them.

        `baselines` feed the with_skill variants; `clim_baseline` feeds the anomaly
        tables. Rollout stores carry a NaN lead-0 placeholder, dropped by default.
        """
        pred, obs = self.pred, self.obs
        if drop_lead0:
            pred = pred.isel(lead_time=slice(1, None))
            obs = obs.isel(lead_time=slice(1, None))
        baseline_cubes = {name: self.load_baseline(name).isel(lead_time=slice(1, None)) for name in baselines}
        clim = None
        if clim_baseline is not None:
            clim = self.load_baseline(clim_baseline).isel(lead_time=slice(1, None))
        self.products = compute_products(pred, obs, products, baselines=baseline_cubes, clim=clim)
        return self.products

    def materialise(self) -> None:
        """Eager-ify the finished products (small); pred/obs stay lazy."""
        self.products = {name: ds.compute() for name, ds in self.products.items()}

    def compute_diagnostics(
        self,
        lead_time,
        products: list,
        baselines: tuple = (),
        clim_baseline: str | None = None,
        class_coords: tuple = (),
        tod: str = "all",
    ) -> dict:
        """Single-lead diagnostic products on the valid_time view; stored in self.diag_products.

        Baseline cubes are the cached load_baseline() alignments sliced at the lead, so
        verification and diagnostics share one load and one alignment per baseline.
        `tod` conditions every product on the local-solar day/night window.
        """
        pred_lead = to_valid_time(self.pred.sel(lead_time=lead_time), lead_time)
        obs_lead = to_valid_time(self.obs.sel(lead_time=lead_time), lead_time)
        baseline_slices = {
            name: to_valid_time(self.load_baseline(name).sel(lead_time=lead_time), lead_time) for name in baselines
        }
        clim = None
        if clim_baseline is not None:
            clim = to_valid_time(self.load_baseline(clim_baseline).sel(lead_time=lead_time), lead_time)
        self.diag_products = compute_diagnostic_products(
            pred_lead,
            obs_lead,
            products,
            baselines=baseline_slices,
            clim=clim,
            class_coords=class_coords,
            tod=tod,
        )
        return self.diag_products

    def materialise_diagnostics(self) -> None:
        """Eager-ify the finished diagnostic products (small)."""
        self.diag_products = {name: ds.compute() for name, ds in self.diag_products.items()}

    # Shared helpers
    def _load_forecast(self) -> xr.Dataset:
        path = self.paths.forecast_path(self.experiment, self.version, self.eval_name)
        if not path.exists():
            raise FileNotFoundError(f"Forecast not found: {path}")
        return xr.open_zarr(path, chunks={"init_time": 500})

    def _filter(self, ds: xr.Dataset) -> xr.Dataset:
        filtered = ds
        if self.temporal is not None:
            origin = pd.Timestamp("2016-01-01")
            time_idx = np.concatenate(
                [np.load(self.paths.temporal_indices_path(p))["time_indices"].astype(np.int64) for p in self.temporal]
            )
            keep_times = origin + pd.to_timedelta(np.unique(time_idx), unit="h")
            filtered = ds.sel(init_time=ds["init_time"].isin(keep_times))

        if self.consistent_mask:
            pred_raw = filtered[f"{self.var}_pred"]
            obs_raw = filtered[f"{self.var}_obs"]
            all_finite = pred_raw.notnull().all("lead_time") & obs_raw.notnull().all("lead_time")
            filtered = filtered.assign(
                {
                    f"{self.var}_pred": pred_raw.where(all_finite),
                    f"{self.var}_obs": obs_raw.where(all_finite),
                }
            )

        return filtered

    def _attach_class_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Attach per-pixel class coords (Köppen-Geiger etc.) from the eval npz — no sampled store.

        The forecast pixels are the anchors in sorted order (Forecast writes np.unique(spatial_idx));
        the npz class arrays are aligned to its src_indices order, so a single argsort re-orders them
        to the forecast pixels. Dense box regions scatter into the (lat, lon) grid via pixel_grid on
        the seviri cell centres — the same geometry the writer used.
        """
        path = self.paths.eval_indices_path("seviri", self.eval_name)
        if not path.exists():
            raise FileNotFoundError(f"Eval indices not found for {self.eval_name}: {path}")
        indices_ds = np.load(path)
        class_keys = ["kg_class", "kg_major", "kg_classes", "biome_classes", "modis_lc_classes", "ecoregion_classes"]
        order = np.argsort(indices_ds["src_indices"], kind="stable")  # npz order -> sorted-anchor order

        coords = {}
        if "pixel" in ds.dims:
            for key in class_keys:
                if key in indices_ds.files:
                    coords[key] = (("pixel",), indices_ds[key][order])
        else:
            sorted_anchors = indices_ds["src_indices"][order]
            centers = store_grid("seviri").centers[sorted_anchors]  # (n, 2) lon/lat
            _, _, local_rows, local_cols, grid_shape = pixel_grid(centers[:, 0], centers[:, 1])
            if grid_shape != (ds.sizes["lat"], ds.sizes["lon"]):
                raise ValueError(
                    f"Grid shape mismatch: expected {grid_shape}, got {(ds.sizes['lat'], ds.sizes['lon'])}"
                )
            for key in class_keys:
                if key in indices_ds.files:
                    class_values = indices_ds[key][order]
                    grid = np.full(grid_shape, -1, dtype=class_values.dtype)
                    grid[local_rows, local_cols] = class_values
                    coords[key] = (("lat", "lon"), grid)
        ds = ds.assign_coords(coords)
        return ds

    def _load_baseline(self) -> xr.DataArray | None:
        if self.baseline is None:
            return None
        if self.baseline not in _BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline: {self.baseline!r}. Available: {list(_BASELINE_REGISTRY)}")

        path_fn, var_name = _BASELINE_REGISTRY[self.baseline]
        path = path_fn(self.paths, self.eval_name)
        if not path.exists():
            raise FileNotFoundError(f"Baseline '{self.baseline}' not found: {path}")

        # chunks={} -> dask-backed, so the per-lead reindex/concat in baseline_full
        # streams instead of materialising the full aligned cube in numpy.
        return xr.open_dataset(path, chunks={})[var_name]

    def _check_baseline_coords(self, baseline: xr.DataArray, name: str) -> None:
        """Baseline coords are float64 BY CONTRACT (see BaselineExtractor._make_dataset).

        float32 coords mean a stale pre-2026-07 extraction: exact-value alignment against
        the float64 forecast grid silently yields an empty intersection (dense grids) or
        drops the conflicting coords (pixel stores).
        """
        for coord in ("lat", "lon"):
            if coord in baseline.coords and baseline[coord].dtype != np.float64:
                raise ValueError(
                    f"Baseline {name!r} has {baseline[coord].dtype} {coord} coords — stale "
                    "pre-2026-07 extraction. Re-extract with scripts/baselines/run_baselines.py "
                    "(era5 itself is not extractable: the era5 store carries no skt)"
                )

    def _align_baseline(self, valid_times: np.ndarray) -> xr.DataArray:
        return self.baseline_ds.reindex(valid_time=valid_times)

    def _subset_pixels(self, baseline_ds: xr.DataArray, forecast_ds: xr.DataArray) -> xr.DataArray:
        """Subset da's pixel dim to match ref's lat/lon coordinates.

        No-op when pixel counts already match or lat coords are absent.
        Used when a baseline covers all store pixels but the forecast uses a subset.
        """
        if (
            "pixel" not in baseline_ds.dims
            or "pixel" not in forecast_ds.dims
            or baseline_ds.sizes["pixel"] == forecast_ds.sizes["pixel"]
            or "lat" not in baseline_ds.coords
            or "lat" not in forecast_ds.coords
        ):
            return baseline_ds
        bl_lats = baseline_ds["lat"].values.astype(np.float32)
        bl_lons = baseline_ds["lon"].values.astype(np.float32)
        lookup = {(float(la), float(lo)): i for i, (la, lo) in enumerate(zip(bl_lats, bl_lons))}
        p_lats = forecast_ds["lat"].values.astype(np.float32)
        p_lons = forecast_ds["lon"].values.astype(np.float32)
        idx = [lookup[(float(la), float(lo))] for la, lo in zip(p_lats, p_lons)]
        return baseline_ds.isel(pixel=idx)

    def bootstrap_summarise(
        self,
        stats: xr.Dataset,
        metric: str = "rmse",
        block: int = 24,
        n_boot: int = 500,
        quantiles=(0.05, 0.50, 0.95),
        seed: int = 0,
    ) -> xr.Dataset:
        """Block-bootstrap CI for a pooled metric over init_time, per lead_time.

        `stats` carries additive sufficient statistics (se_sum, count) with dims
        (init_time, lead_time) — see metrics.compute_rmse_sufficient_stats. We moving-block
        resample init_time (blocks of `block` consecutive inits, to respect temporal
        autocorrelation — block=1 recovers the iid/climpred bootstrap), pool each resample via
        sqrt(Σse_sum / Σcount), and take percentiles over the bootstrap curves.

        Returns the SAME schema as `summarise` ({metric}_q05/q50/q95 over lead_time) so the plots
        are unchanged. q50 is the point estimate (pooled over ALL inits); the outer quantiles are
        bootstrap percentiles. NOTE: `block` is in init-INDEX units, so its wall-clock span depends
        on the init spacing (e.g. 6-hourly inits → block=4 ≈ 1 day).
        """
        se = stats["se_sum"].transpose("init_time", "lead_time").values
        cnt = stats["count"].transpose("init_time", "lead_time").values
        n_init = se.shape[0]

        def pooled(rows):
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.sqrt(se[rows].sum(0) / cnt[rows].sum(0))

        block = min(block, n_init)
        n_blocks = int(np.ceil(n_init / block))
        starts = np.arange(n_init - block + 1)
        rng = np.random.default_rng(seed)
        curves = np.empty((n_boot, se.shape[1]))
        for b in range(n_boot):
            rows = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n_blocks)])[:n_init]
            curves[b] = pooled(rows)

        point = pooled(np.arange(n_init))
        data_vars = {}
        for q in quantiles:
            label = f"{metric}_q{int(q * 100):02d}"
            if abs(q - 0.5) < 1e-9:
                data_vars[label] = ("lead_time", point)
            else:
                data_vars[label] = ("lead_time", np.nanpercentile(curves, q * 100, axis=0))
        return xr.Dataset(data_vars, coords={"lead_time": stats["lead_time"].values})


class EvaluatorVerification(Evaluator):
    """Full (init_time, lead_time) verification on evaluation products.

    Use load() / load_baseline() / compute(products) / materialise(); the finished
    datasets land in self.products (see evaluation/products.py for the catalogue).
    """
