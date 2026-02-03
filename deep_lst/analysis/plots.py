import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.utils import ExperimentPaths


class PlotResults:
    def __init__(self, results: xr.Dataset, experiment: str, version: str):
        self.results = results
        self.experiment = experiment
        self.version = version
        self.paths = ExperimentPaths()

        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.labelsize": 16,
                "axes.titlesize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 12,
                "figure.titlesize": 18,
            }
        )

    def regression_parity(self, n_lead_times: int = None):
        """
        Create a single parity plot with points colored by lead time.
        n_lead_times : int, optional
            Number of lead times to include. If None, uses all available lead times.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Select lead times and get number
        lead_times = self.results.lead_time if n_lead_times is None else self.results.lead_time[:n_lead_times]
        colors = plt.cm.viridis(np.linspace(0, 1, len(lead_times)))

        # Scatter plot colored by lead time
        for i, lt in enumerate(lead_times):
            ax.scatter(
                self.results["LST_obs"].isel(init_time=0, lead_time=i),
                self.results["LST_pred"].isel(init_time=0, lead_time=i),
                c=[colors[i]],
                label=f"Lead Time {i}",
                alpha=0.6,
                s=20,
            )

        # Add 45-degree diagonal
        lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
        lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5, linewidth=1.5)

        # Formatting
        ax.set_xlabel("SEVIRI")
        ax.set_ylabel("Forecast")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.paths.plots_path(self.experiment, self.version, "regression_parity"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    def prediction_maps(self, lead_time):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot observations
        self.results["LST_obs"].isel(init_time=0, lead_time=lead_time).plot(
            ax=axes[0], cmap="RdBu_r", add_colorbar=False
        )
        axes[0].set_title("SEVIRI")

        # Plot forecast
        im = (
            self.results["LST_pred"]
            .isel(init_time=0, lead_time=lead_time)
            .plot(ax=axes[1], cmap="RdBu_r", add_colorbar=False)
        )
        axes[1].set_title("Forecast")

        for ax in axes:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        # Add shared colorbar to the right
        fig.subplots_adjust(wspace=0.4, right=0.88)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Temperature (K)", rotation=270, labelpad=20)

        plt.savefig(
            self.paths.plots_path(self.experiment, self.version, f"prediction_maps_lt_{lead_time}"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def plume_diagram_lts(self, n_pixels: int = 10, n_init_times: int = 4, seed: int = 42):
        """Plot forecast distributions as boxplots for each valid time.

        Shows the distribution of forecasts across all lead times for each valid time,
        with observations overlaid as a reference line.

        n_pixels : int
            Number of random pixel locations to sample.
        n_init_times : int
            Number of sequential init times to plot (starting from the first).
        seed : int
            Random seed for pixel selection reproducibility.
        """
        ds = self.results
        init_times = ds.init_time.values[:n_init_times]
        lead_times = pd.TimedeltaIndex(ds.lead_time.values)

        # Find pixels that have valid data across all selected init times and lead times
        valid_mask = np.isfinite(ds["LST_pred"].sel(init_time=init_times)).all(
            dim=["init_time", "lead_time"]
        ) & np.isfinite(ds["LST_obs"].sel(init_time=init_times)).all(dim=["init_time", "lead_time"])
        valid_lats, valid_lons = np.where(valid_mask.values)
        if len(valid_lats) == 0:
            print("No fully valid pixels found for forecast_timeseries plot.")
            return

        rng = np.random.default_rng(seed)
        n_pixels = min(n_pixels, len(valid_lats))
        chosen = rng.choice(len(valid_lats), size=n_pixels, replace=False)
        pixel_lat_idx = valid_lats[chosen]
        pixel_lon_idx = valid_lons[chosen]

        # Collect all forecasts and observations by valid time
        valid_time_data = {}  # {valid_time: {'forecasts': [], 'obs': []}}

        for li, lo in zip(pixel_lat_idx, pixel_lon_idx):
            for it in init_times:
                for j, lt in enumerate(lead_times):
                    vt = pd.Timestamp(it) + lt

                    # Get forecast value
                    pred_val = float(ds["LST_pred"].sel(init_time=it).isel(lead_time=j, lat=li, lon=lo))
                    # Get observation value
                    obs_val = float(ds["LST_obs"].sel(init_time=it).isel(lead_time=j, lat=li, lon=lo))

                    if np.isfinite(pred_val):
                        if vt not in valid_time_data:
                            valid_time_data[vt] = {"forecasts": [], "obs": []}
                        valid_time_data[vt]["forecasts"].append(pred_val)
                        if np.isfinite(obs_val):
                            valid_time_data[vt]["obs"].append(obs_val)

        # Sort by valid time
        sorted_times = sorted(valid_time_data.keys())
        forecast_distributions = [valid_time_data[vt]["forecasts"] for vt in sorted_times]
        obs_means = [
            np.mean(valid_time_data[vt]["obs"]) if valid_time_data[vt]["obs"] else np.nan for vt in sorted_times
        ]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Create boxplots
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

        # Overlay observation means
        ax.plot(
            range(len(sorted_times)),
            obs_means,
            "o-",
            color="grey",
            linewidth=2,
            markersize=6,
            alpha=0.7,
            label="Observations (pixel average)",
            zorder=3,
        )

        # Format x-axis with valid times
        ax.set_xticks(range(len(sorted_times)))
        ax.set_xticklabels([vt.strftime("%Y-%m-%d %H:%M") for vt in sorted_times], rotation=45, ha="right")

        ax.set_xlabel("Validation Time")
        ax.set_ylabel("LST")
        ax.set_title(f"Forecast distribution : {n_pixels} pixels; {n_init_times} init times; all lead times")
        # ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            self.paths.plots_path(self.experiment, self.version, "plume_diagram_lts"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
