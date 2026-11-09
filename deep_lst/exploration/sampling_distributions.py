"""
Mahalanobis distance of spatial evaluation regions from the ERA5 forcing
training distribution.

Mahalanobis distance measures how far each evaluation pixel (or region
centroid) sits from the training distribution in covariate feature space.
Unlike Euclidean distance, it accounts for the variance *and* the
correlation structure of the training features: it is equivalent to
Euclidean distance in the PCA-whitened space (all principal components
equally weighted, scaled by their variance).  A distance of 0 means the
point lies at the training centroid; the 95th percentile of
within-training distances defines the "training envelope" — eval samples
beyond it occupy feature-space regions the model has not seen during
training.

Each sample is the temporal mean of the forcing at one anchor site over a
configurable time window (default: 2024), read from the full era5 and
era5_land stores via the same nearest-neighbour anchor mapping the training
preload uses (coarse cells repeat across nearby anchors).  Training samples
are the 3000 MTG training sites; evaluation samples are the dense pixel
sets within each evaluation region.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eodata.grids import T1H
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA

from deep_lst.utils import ExperimentPaths
from deep_lst.utils.full_store import ERA5_LAND_STORE, ERA5_STORE, load_store_columns
from deep_lst.utils.spatial import store_grid

# ── configuration ─────────────────────────────────────────────────────────────
PATHS = ExperimentPaths()

TRAIN_SPATIAL = "mtg"
EVAL_NAMES = [
    "dahra",
    "gobabeb",
    "yangambi",
    "bouamir",
    "west_africa",
    "east_africa",
    "johannesburg",
    "antananarivo",
]
# cp (convective precipitation) is not in the era5 full store — dropped from the original list.
FEATURE_VARS = [
    "stl1",
    "swvl1",
    "u10",
    "v10",
    "sp",
    "t2m",
    "d2m",
    "sf",
    "strd",
    "ssrd",
    "tp",
    "tcwv",
    "skt",
    "tvh",
    "tvl",
]
MODALITY_LOAD = {
    "era5": {
        "store": ERA5_STORE,
        "data": ["u10", "v10", "sp", "sf", "strd", "ssrd", "tp", "tcwv"],
        "data_static_categorical": ["tvh", "tvl"],
    },
    "era5_land": {
        "store": ERA5_LAND_STORE,
        "data": ["stl1", "swvl1", "t2m", "d2m", "skt"],
    },
}

RIDGE_ALPHA = 1e-6  # added to Sigma diagonal when the condition number is high
N_SUBSAMPLE = 500  # training points sampled for the within-training reference
YEAR_FILTER = 2024  # time window for the temporal means

OUT_DIR = PATHS.results_root / "exploration" / "mahalanobis_distance"
SAVE_CSV = True
SAVE_FIGURES = True

YEAR_DAY_RANGE = (
    T1H.date_to_index(f"{YEAR_FILTER}-01-01") // 24,
    T1H.date_to_index(f"{YEAR_FILTER + 1}-01-01") // 24,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def load_feature_matrix(spatial, anchor):
    """Return (X, lonlat): X is (n_anchors, n_features) of temporal means, lonlat the
    per-anchor era5 cell centres (lon, lat).

    Each anchor gets its nearest era5 and era5_land cell's features; hourly variables
    are averaged over YEAR_FILTER, static categoricals (tvh, tvl) pass through.
    """
    per_var = {}
    lonlat = None
    for modality, spec in MODALITY_LOAD.items():
        load = {leaf: names for leaf, names in spec.items() if leaf != "store"}
        columns, unique_rows, inverse, _ = load_store_columns(
            modality, spec["store"], spatial, load, anchor=anchor, day_range=YEAR_DAY_RANGE, num_workers=16
        )
        data = columns["data"].numpy().astype(np.float64)  # (n_unique, n_hours, C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(data, axis=1)  # (n_unique, C)
        for channel, var in enumerate(load["data"]):
            per_var[var] = means[inverse, channel]
        if "data_static_categorical" in load:
            static = columns["data_static_categorical"].numpy().astype(np.float64)
            for channel, var in enumerate(load["data_static_categorical"]):
                per_var[var] = static[inverse, channel]
        if modality == "era5":
            lonlat = store_grid("era5").centers[unique_rows][inverse]  # (n_anchors, 2)

    X = np.stack([per_var[var] for var in FEATURE_VARS], axis=1)
    return X, lonlat


def fit_training_distribution(X_train, ridge_alpha=1e-6):
    """
    Fit the multivariate Gaussian characterising the training distribution.

    Returns:
        mu        -- mean vector, shape (p,)
        Sigma_inv -- precision matrix (inverse covariance), shape (p, p)
                     This is the key object: multiplying by Sigma_inv is
                     what "whitens" each direction by its variance and
                     removes redundancy from correlated features.
        cond      -- condition number of the raw covariance matrix
    """
    mu = X_train.mean(axis=0)  # (p,)
    Sigma = np.cov(X_train, rowvar=False)  # (p, p)

    cond = np.linalg.cond(Sigma)
    if cond > 1e6:
        warnings.warn(
            f"Covariance condition number {cond:.2e} > 1e6; applying ridge α={ridge_alpha}",
            UserWarning,
            stacklevel=2,
        )

    # Ridge regularisation keeps the matrix invertible when features are
    # nearly collinear (adds a small constant to each eigenvalue).
    Sigma_reg = Sigma + ridge_alpha * np.eye(Sigma.shape[0])
    Sigma_inv = np.linalg.inv(Sigma_reg)

    return mu, Sigma_inv, cond


def mahalanobis_distances(X, mu, Sigma_inv):
    """
    Vectorised Mahalanobis distance from each row of X to the distribution
    defined by (mu, Sigma_inv).

    Formula: d(x) = sqrt( (x - mu)^T  Sigma^{-1}  (x - mu) )

    Steps:
        1. Centre:   diff  = X - mu           -- shape (n, p)
        2. Whiten:   left  = diff @ Sigma_inv  -- shape (n, p); rotates and
                                                  scales each direction by
                                                  inverse variance
        3. Quadform: sq_d  = sum(left * diff, axis=1)  -- (n,); row-wise dot
                                                           product = quadratic
                                                           form per sample
        4. Distance: sqrt(sq_d)
    """
    diff = X - mu  # (n, p): centred samples
    left = diff @ Sigma_inv  # (n, p): whitened
    sq_d = (left * diff).sum(axis=1)  # (n,):   quadratic form, one value per sample
    return np.sqrt(np.clip(sq_d, 0, None))


# ── load & fit ────────────────────────────────────────────────────────────────

print(f"Loading training data ({TRAIN_SPATIAL}, year={YEAR_FILTER}) …")
X_train, _ = load_feature_matrix(TRAIN_SPATIAL, "mtg")
n_train, n_feat = X_train.shape
print(f"  X_train: {X_train.shape}  ({n_feat} features)")

print("Fitting training distribution …")
mu, Sigma_inv, cond = fit_training_distribution(X_train, RIDGE_ALPHA)
print(f"  Condition number of Σ: {cond:.2e}")

rng = np.random.default_rng(42)
ref_idx = rng.choice(n_train, size=min(N_SUBSAMPLE, n_train), replace=False)
d_train_ref = mahalanobis_distances(X_train[ref_idx], mu, Sigma_inv)
ref_med = np.median(d_train_ref)
ref_q25 = np.percentile(d_train_ref, 25)
ref_q75 = np.percentile(d_train_ref, 75)
p95_train = np.percentile(d_train_ref, 95)
print(f"  Within-training: median={ref_med:.2f}, IQR=[{ref_q25:.2f}, {ref_q75:.2f}], 95th={p95_train:.2f}")

# ── per-region analysis ───────────────────────────────────────────────────────

eval_results = {}
summary_rows = []

for name in EVAL_NAMES:
    print(f"Loading eval region '{name}' …")
    X_eval, eval_lonlat = load_feature_matrix(f"eval_{name}", "seviri")
    n_eval = len(X_eval)

    if n_eval < n_feat:
        warnings.warn(
            f"Region '{name}': n_eval={n_eval} < n_features={n_feat}; Figure 2 spatial panel will be skipped.",
            UserWarning,
        )

    d_eval = mahalanobis_distances(X_eval, mu, Sigma_inv)
    centroid_d = mahalanobis_distances(X_eval.mean(axis=0, keepdims=True), mu, Sigma_inv)[0]
    frac_outside = (d_eval > p95_train).mean()
    lons, lats = eval_lonlat[:, 0], eval_lonlat[:, 1]

    eval_results[name] = dict(
        X=X_eval,
        d=d_eval,
        lats=lats,
        lons=lons,
        centroid_d=centroid_d,
        frac_outside=frac_outside,
    )
    summary_rows.append(
        dict(
            region=name,
            n_samples=n_eval,
            centroid_distance=centroid_d,
            mean_distance=d_eval.mean(),
            median_distance=np.median(d_eval),
            frac_outside_95pct_training=frac_outside,
        )
    )
    print(f"  n={n_eval}, centroid_d={centroid_d:.2f}, mean={d_eval.mean():.2f}, frac_outside={frac_outside:.1%}")

summary = pd.DataFrame(summary_rows).sort_values("centroid_distance").reset_index(drop=True)
region_order = summary["region"].tolist()

# ── summary table ─────────────────────────────────────────────────────────────
pd.set_option("display.float_format", "{:.3f}".format)
print("\n── Mahalanobis distance summary ──────────────────────────────────────────")
print(summary.to_string(index=False))

if SAVE_CSV:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "mahalanobis_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

# ── Figure 1: per-region box-whisker summary ──────────────────────────────────
n_regions = len(region_order)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_regions))

fig1, ax = plt.subplots(figsize=(max(9, n_regions * 1.3), 6))

ax.axhspan(ref_q25, ref_q75, color="grey", alpha=0.2, label="Training IQR")
ax.axhline(ref_med, color="grey", lw=1.5, ls="--", label=f"Training median ({ref_med:.1f})")
ax.axhline(p95_train, color="grey", lw=1, ls=":", label=f"Training 95th pct ({p95_train:.1f})")

for i, rname in enumerate(region_order):
    d = eval_results[rname]["d"]
    q5, q25, q50, q75, q95 = np.percentile(d, [5, 25, 50, 75, 95])
    c = colors[i]
    w = 0.35

    ax.bar(i, q75 - q25, bottom=q25, width=w, color=c, alpha=0.75)
    ax.plot([i - w / 2, i + w / 2], [q50, q50], color="k", lw=2)
    ax.plot([i, i], [q5, q25], color=c, lw=1.5)
    ax.plot([i, i], [q75, q95], color=c, lw=1.5)
    ax.plot([i - w / 4, i + w / 4], [q5, q5], color=c, lw=1.5)
    ax.plot([i - w / 4, i + w / 4], [q95, q95], color=c, lw=1.5)
    ax.scatter(i, eval_results[rname]["centroid_d"], color=c, edgecolors="k", s=80, zorder=5)

ax.set_xticks(np.arange(n_regions))
ax.set_xticklabels(region_order, rotation=30, ha="right", fontsize=13)
ax.set_ylabel("Mahalanobis distance", fontsize=14)
ax.set_title(
    "Mahalanobis distance of evaluation regions from training distribution\n"
    "(box = IQR, whiskers = 5th/95th pct; dot = centroid distance)",
    fontsize=14,
)
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=12)
fig1.tight_layout()

# ── Figure 2: per-sample spatial context ──────────────────────────────────────
# PCA on the training set is used as a fallback x/y axis when the eval region
# is too spatially compact to use lat/lon directly.
pca = PCA(n_components=2).fit(X_train)

valid_regions = [r for r in region_order if len(eval_results[r]["X"]) >= n_feat]
skipped = [r for r in region_order if r not in valid_regions]
if skipped:
    print(f"\nFigure 2: skipping {skipped} (n_samples < n_features)")

if valid_regions:
    ncols = min(3, len(valid_regions))
    nrows = (len(valid_regions) + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    vmax = max(eval_results[r]["d"].max() for r in valid_regions)
    norm = TwoSlopeNorm(vcenter=ref_med, vmin=0.0, vmax=max(vmax, ref_med + 0.01))

    for ax_i, rname in enumerate(valid_regions):
        row, col = divmod(ax_i, ncols)
        ax = axes[row][col]
        res = eval_results[rname]

        lats, lons = res["lats"], res["lons"]
        if np.ptp(lons) > 0.01 and np.ptp(lats) > 0.01:
            x_plot, y_plot = lons, lats
            xlabel, ylabel = "Longitude", "Latitude"
        else:
            pc = pca.transform(res["X"])
            x_plot, y_plot = pc[:, 0], pc[:, 1]
            xlabel, ylabel = "PC1", "PC2"

        sc = ax.scatter(x_plot, y_plot, c=res["d"], cmap="RdYlBu_r", norm=norm, s=40, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Mahalanobis dist.", pad=0.02)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{rname}\ncentroid d = {res['centroid_d']:.2f}", fontsize=13)
        ax.tick_params(labelsize=11)

    for ax_i in range(len(valid_regions), nrows * ncols):
        row, col = divmod(ax_i, ncols)
        axes[row][col].set_visible(False)

    fig2.suptitle(
        "Per-sample Mahalanobis distance within evaluation regions\n"
        f"(colormap anchored at training median = {ref_med:.1f})",
        fontsize=15,
    )
    fig2.tight_layout()

# ── save / show ───────────────────────────────────────────────────────────────
if SAVE_FIGURES:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = OUT_DIR / "mahalanobis_summary.pdf"
    fig1.savefig(p1, bbox_inches="tight")
    print(f"Saved: {p1}")
    if valid_regions:
        p2 = OUT_DIR / "mahalanobis_spatial.pdf"
        fig2.savefig(p2, bbox_inches="tight")
        print(f"Saved: {p2}")
else:
    plt.show()
