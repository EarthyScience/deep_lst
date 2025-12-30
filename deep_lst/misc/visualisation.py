import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def plot_predictions(ds, vars, outpath = "./plots"):

    # Compute a common color scale across both fields
    vmin = np.nanmin([np.nanmin(ds[vars[0]]), np.nanmin(ds[vars[1]])])
    vmax = np.nanmax([np.nanmax(ds[vars[0]]), np.nanmax(ds[vars[1]])])
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].pcolormesh(ds[vars[0]].coords['longitude'], ds[vars[0]].coords['latitude'], ds[vars[0]], cmap='pink_r', shading='auto', norm=norm)
    axes[0].set_ylabel('Latitude')
    axes[0].set_xlabel('Longitude')
    axes[0].set_title('Target LST')
    axes[1].set_ylabel('Latitude')
    axes[1].set_xlabel('Longitude')
    axes[1].set_title('Predicted LST')

    im1 = axes[1].pcolormesh(ds[vars[1]].coords['longitude'], ds[vars[1]].coords['latitude'], ds[vars[1]], cmap='pink_r', shading='auto', norm=norm)
    cbar = fig.colorbar(im1, ax=axes, location='right', fraction=0.046, pad=0.02)
    cbar.set_label('LST (K)')

    plt.savefig(f'{outpath}/predictions_plot.pdf')

def plot_bias(bias, ds, vars, outpath = "./plots"):

    fig, axes = plt.subplots(1,1, figsize=(5, 5), constrained_layout=True)
    im0 = axes.pcolormesh(ds[vars[0]].coords['longitude'], ds[vars[0]].coords['latitude'], bias.values, cmap='pink_r', shading='auto')
    axes.set_ylabel('Latitude')
    axes.set_xlabel('Longitude')
    #axes.set_title('LST Bias (Target - Predicted)')
    axes.set_aspect('equal', adjustable='box')

    cbar = fig.colorbar(im0, ax=axes, location='right', fraction=0.046, pad=0.02)
    cbar.set_label('LST (K)')

    plt.savefig(f'{outpath}/bias_plot.pdf')

def plot_anomalies(ds, vars, bias, anomaly_targ, anomaly_pred, outpath = "./plots"):

    vmin = np.nanmin([np.nanmin(bias.values), np.nanmin(anomaly_pred.values)])
    vmax = np.nanmax([np.nanmax(bias.values), np.nanmax(anomaly_pred.values)])
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].pcolormesh(ds[vars[0]].coords['longitude'], ds[vars[0]].coords['latitude'], anomaly_targ.values, cmap='pink_r', shading='auto', norm=norm)
    axes[0].set_ylabel('Latitude')
    axes[0].set_xlabel('Longitude')
    axes[0].set_title('Target Anomaly')
    axes[1].set_ylabel('Latitude')
    axes[1].set_xlabel('Longitude')
    axes[1].set_title('Prediction Anomaly')

    im1 = axes[1].pcolormesh(ds[vars[1]].coords['longitude'], ds[vars[1]].coords['latitude'], anomaly_pred.values, cmap='pink_r', shading='auto', norm=norm)
    cbar = fig.colorbar(im1, ax=axes, location='right', fraction=0.046, pad=0.02)
    cbar.set_label('LST (K)')

    plt.savefig(f'{outpath}/anomalies_plot.pdf')

def plot_correlation(ds, outpath = "./plots"):

    fig, axes = plt.subplots(1,1, figsize=(5, 5), constrained_layout=True)
    axes.scatter(ds.targ.values, ds.pred.values, color = "gray", alpha=0.5, edgecolors='none')

    min_val = np.nanmin([np.nanmin(ds.targ.values), np.nanmin(ds.pred.values)])
    max_val = np.nanmax([np.nanmax(ds.targ.values), np.nanmax(ds.pred.values)])
    axes.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2, color="black")

    axes.set_xlabel("Target LST")
    axes.set_ylabel("Predicted LST")
    plt.savefig(f'{outpath}/correlation_plot.pdf')