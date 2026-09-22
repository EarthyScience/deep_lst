"""KG climate class sampling representativeness plots.

plot_kg_bars(df, out_path)
    Grouped bar chart: area fraction (grey) vs sample fraction (blue) per KG class,
    sorted by area fraction descending. Secondary axis shows sampling ratio as scatter
    with a dashed reference line at 1. Each group annotated with absolute n.

plot_kg_map(df, africa, africa_grid, africa_kg, valid, grid, npz_path, out_path)
    Africa map of KG classes coloured by sampling ratio (white=0, dark blue=high).
    Training sites overlaid as points coloured by MODIS IGBP land-cover class.
    Eval region centres drawn as labelled circles (3-degree radius, black outline).
    No frame, axes, or title; background transparent.
"""

import os

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

from deep_lst.exploration.plots._design import RC

# MODIS IGBP Land Cover Type 1 class colours and names
IGBP_COLORS = {
    1: "#006400",  # Evergreen Needleleaf Forest
    2: "#00a000",  # Evergreen Broadleaf Forest
    3: "#7ec850",  # Deciduous Needleleaf Forest
    4: "#b4e04b",  # Deciduous Broadleaf Forest
    5: "#3a9c3a",  # Mixed Forests
    6: "#b5a02a",  # Closed Shrublands
    7: "#d7c752",  # Open Shrublands
    8: "#c8a020",  # Woody Savannas
    9: "#ede840",  # Savannas
    10: "#a8d015",  # Grasslands
    11: "#00b060",  # Permanent Wetlands
    12: "#cc2222",  # Croplands
    13: "#c0c0c0",  # Urban and Built-up
    14: "#f09050",  # Cropland/Natural Veg. Mosaic
    15: "#d4eef4",  # Snow and Ice
    16: "#d2b48c",  # Barren
    17: "#4169e1",  # Water Bodies
}
IGBP_NAMES = {
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up",
    14: "Cropland/Natural Veg. Mosaic",
    15: "Snow and Ice",
    16: "Barren",
    17: "Water Bodies",
}


EVAL_CENTERS: dict[str, tuple[float, float]] = {
    "Dahra": (-15.43, 15.35),
    "West\n Africa": (-3.00, 8.00),
    "Sahel": (9.50, 15.50),
    "Gobabeb": (15.04, -23.56),
    "Johannesburg": (28.05, -26.20),
    "Antananarivo": (46.50, -19.50),
    "Passage": (37.79, 3.36),
    "Yangambi": (24.5, 0.81),
    "Bouamir": (12.82, 3.19),
}


def _lc_rgb(lc_array):
    return np.array([to_rgb(IGBP_COLORS.get(int(c), "#808080")) for c in lc_array])


def _plot_lc_histogram(ax, lc_array):
    """Training-site frequency per IGBP land-cover class, bars coloured by class.

    Doubles as the land-cover legend (bar colour = class), replacing the patch legend.
    """
    counts = np.bincount(lc_array.astype(int))
    present = [c for c in range(len(counts)) if c > 0 and counts[c] > 0]
    x = np.arange(len(present))
    ax.bar(x, [counts[c] for c in present], width=0.6, color=[IGBP_COLORS.get(c, "#808080") for c in present])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}: {IGBP_NAMES.get(c, '')}" for c in present], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("n sites", fontsize=10)
    ax.set_title("Land cover class", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_kg_bars(df, out_path):
    plt.rcParams.update(RC)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = len(df)
    x = np.arange(n)
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(max(10, n * 0.9), 5))
    ax2 = ax1.twinx()

    ax1.bar(x - w / 2, df["area_frac"], width=w, color="#888888")
    ax1.bar(x + w / 2, df["sample_frac"], width=w, color="#2196F3")

    ax2.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax2.scatter(x, df["ratio"], color="tomato", s=25, zorder=5)
    ax2.set_ylabel("Sampling ratio", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], rotation=45, ha="right")
    ax1.set_ylabel("Fraction")

    for i, row in enumerate(df.itertuples()):
        top = max(row.area_frac, row.sample_frac) + ax1.get_ylim()[1] * 0.01
        ax1.text(i, top, f"n={row.n_samples}", ha="center", va="bottom", fontsize=7)

    handles = [
        mpatches.Patch(color="#888888", label="Area fraction (Africa)"),
        mpatches.Patch(color="#2196F3", label="Sample fraction"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato", markersize=6, label="Sampling ratio"),
        Line2D([0], [0], color="k", ls="--", lw=0.8, label="ratio = 1"),
    ]
    ax1.legend(handles=handles, loc="upper right", fontsize=8)

    ax1.set_title("Climate class sampling vs. continental area coverage")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def plot_kg_map(df, africa, africa_grid, africa_kg, valid, grid, npz_path, out_path):
    plt.rcParams.update(RC)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    class_to_ratio = {row.cls: (row.ratio if not np.isnan(row.ratio) else 0.0) for row in df.itertuples()}
    class_to_label = {row.cls: row.label for row in df.itertuples()}

    africa_lons = africa_grid.centers[valid, 0]
    africa_lats = africa_grid.centers[valid, 1]
    africa_kg_valid = africa_kg[valid]
    africa_ratios = np.array([class_to_ratio.get(int(c), 0.0) for c in africa_kg_valid])

    coords = np.column_stack([africa_lons, africa_lats])

    train_npz = np.load(npz_path)
    train_lons = grid.centers[train_npz["mapping"], 0]
    train_lats = grid.centers[train_npz["mapping"], 1]
    train_lc = train_npz["modis_lc_classes"]

    cmap = LinearSegmentedColormap.from_list("wblue", ["white", "#08306b"])
    vmax = float(np.nanmax(africa_ratios))

    fig, ax = plt.subplots(figsize=(12, 10))

    sc = ax.scatter(
        africa_lons,
        africa_lats,
        c=africa_ratios,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        s=1,
        linewidths=0,
        rasterized=True,
    )
    gpd.GeoDataFrame(geometry=[africa.geom], crs="EPSG:4326").boundary.plot(
        ax=ax,
        color="black",
        lw=0.5,
    )
    ax.scatter(
        train_lons,
        train_lats,
        c=_lc_rgb(train_lc),
        s=18,
        linewidths=0.2,
        edgecolors="#303030",
        zorder=5,
    )

    for name, (lon, lat) in EVAL_CENTERS.items():
        dists = (coords[:, 0] - lon) ** 2 + (coords[:, 1] - lat) ** 2
        kg_label = class_to_label.get(int(africa_kg_valid[np.argmin(dists)]), "?")
        ax.add_patch(
            mpatches.Rectangle(
                (lon - 2.5, lat - 2.5),
                width=5,
                height=5,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
                zorder=6,
            )
        )
        ax.text(
            lon,
            lat,
            f"{name}\n({kg_label})",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            zorder=7,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
        )

    plt.colorbar(sc, ax=ax, label="Sampling ratio", shrink=0.6)

    axh = ax.inset_axes([0.03, 0.11, 0.30, 0.22])
    _plot_lc_histogram(axh, train_lc)

    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)
    print(f"Map saved to {out_path}")


def plot_kg_map_grey(df, africa, africa_grid, africa_kg, valid, grid, npz_path, out_path):
    """Like plot_kg_map but background coloured by KG class in shades of grey.

    Classes 1–30 (Af→EF) map linearly from light grey to dark grey.
    Training sample points have thin dark-grey outlines.
    """

    plt.rcParams.update(RC)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    class_to_label = {row.cls: row.label for row in df.itertuples()}

    africa_lons = africa_grid.centers[valid, 0]
    africa_lats = africa_grid.centers[valid, 1]
    africa_kg_valid = africa_kg[valid]

    # Classes 1–30 (Af→EF); map linearly to grey range [0.15, 0.75]
    def _kg_grey(cls):
        return 0.15 + (int(cls) - 1) / 29 * 0.60

    africa_grey = np.array([_kg_grey(c) for c in africa_kg_valid])

    train_npz = np.load(npz_path)
    train_lons = grid.centers[train_npz["mapping"], 0]
    train_lats = grid.centers[train_npz["mapping"], 1]
    train_lc = train_npz["modis_lc_classes"]

    coords = np.column_stack([africa_lons, africa_lats])

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        africa_lons,
        africa_lats,
        c=africa_grey,
        cmap="Greys",
        vmin=0,
        vmax=1,
        s=1,
        linewidths=0,
        rasterized=True,
    )
    gpd.GeoDataFrame(geometry=[africa.geom], crs="EPSG:4326").boundary.plot(
        ax=ax,
        color="black",
        lw=0.5,
    )
    ax.scatter(
        train_lons,
        train_lats,
        c=_lc_rgb(train_lc),
        s=18,
        linewidths=0.2,
        edgecolors="#303030",
        zorder=5,
    )

    for name, (lon, lat) in EVAL_CENTERS.items():
        dists = (coords[:, 0] - lon) ** 2 + (coords[:, 1] - lat) ** 2
        kg_label = class_to_label.get(int(africa_kg_valid[np.argmin(dists)]), "?")
        ax.add_patch(
            mpatches.Rectangle(
                (lon - 2.5, lat - 2.5),
                width=5,
                height=5,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
                zorder=6,
            )
        )
        ax.text(
            lon,
            lat,
            f"{name}\n({kg_label})",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            zorder=7,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
        )

    axh = ax.inset_axes([0.03, 0.11, 0.30, 0.22])
    _plot_lc_histogram(axh, train_lc)

    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Map saved to {out_path}")
