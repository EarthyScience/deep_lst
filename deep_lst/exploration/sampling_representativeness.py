"""
Exploratory analysis: KG climate class sampling representativeness.
Continental area proxy = 5 km MajorTomGrid cells within Africa, sampled from KG GeoTIFF.
"""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless cluster script
from eodata.grids.majortom import MajorTomGrid
from eodata.util.raster import sample_raster_on_grid
from eodata.util.spatial import get_admin_geometry

from deep_lst.exploration.plots.kg_sampling import plot_kg_bars, plot_kg_map, plot_kg_map_grey

# ── paths ─────────────────────────────────────────────────────────────────────
KG_TIFF = "/Net/Groups/BGI/people/mwessel/auxilary_data/Beck_KG_V1/Beck_KG_V1_present_0p083.tif"
NPZ_PATH = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/spatial/indices_mtg.npz"
EVAL_NPZ_PATH = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/spatial/indices_eval_mtg.npz"
META_PATH = "/Net/Groups/BGI/tscratch/weathergenerator/indices/africa/indices_meta.json"
OUT_PATH = "/Net/Groups/BGI/people/mwessel/deep_lst/results/exploration/sampling_representativeness/kg_classes.pdf"
OUT_PATH_MAP = (
    "/Net/Groups/BGI/people/mwessel/deep_lst/results/exploration/sampling_representativeness/kg_classes_map.pdf"
)
OUT_PATH_MAP_GRAY = (
    "/Net/Groups/BGI/people/mwessel/deep_lst/results/exploration/sampling_representativeness/kg_classes_map_gray.pdf"
)

# ── step 1: continental area proxy ────────────────────────────────────────────
print("Building Africa MajorTomGrid (5 km) ...")
grid = MajorTomGrid(grid_distance=5)
africa = get_admin_geometry(continent="Africa")
africa_grid = grid.intersect(africa, as_grid=True)

print(f"  Africa grid cells: {len(africa_grid):,}")

print("Sampling KG classes on Africa grid ...")
africa_kg = sample_raster_on_grid(KG_TIFF, africa_grid)

valid = africa_kg > 0
classes, cnts = np.unique(africa_kg[valid], return_counts=True)
area_counts = dict(zip(classes.tolist(), cnts.tolist()))

print(f"  KG classes found in Africa pool: {sorted(area_counts)}")
print(f"  Total valid Africa cells: {sum(area_counts.values()):,}")

# ── step 2: sample counts ─────────────────────────────────────────────────────
print("Loading sample counts from metadata ...")
with open(META_PATH) as f:
    meta = json.load(f)

kg_labels = meta["kg_class_labels"]
sample_counts = {int(k): v for k, v in meta["kg_class_counts"]["sites"].items()}
n_samples_total = meta["n_sites"]

print(f"  Total training sites: {n_samples_total}")
print(f"  KG classes in sample: {sorted(sample_counts)}")

# ── step 3: fractions and ratios ──────────────────────────────────────────────
all_classes = sorted(set(area_counts) | set(sample_counts))
n_area_total = sum(area_counts.values())

rows = []
for c in all_classes:
    n_area = area_counts.get(c, 0)
    n_samp = sample_counts.get(c, 0)
    area_frac = n_area / n_area_total
    samp_frac = n_samp / n_samples_total
    ratio = samp_frac / area_frac if area_frac > 0 else np.nan
    rows.append(
        dict(
            cls=c,
            label=kg_labels[str(c)],
            n_area=n_area,
            n_samples=n_samp,
            area_frac=area_frac,
            sample_frac=samp_frac,
            ratio=ratio,
        )
    )

df = pd.DataFrame(rows).sort_values("area_frac", ascending=False).reset_index(drop=True)

# ── step 4: summary table ─────────────────────────────────────────────────────
print("\n── KG class sampling representativeness ──────────────────────────────")
print(f"{'class':>6}  {'label':>5}  {'area_frac':>10}  {'sample_frac':>11}  {'ratio':>6}  {'n':>5}")
for row in df.itertuples():
    ratio_str = f"{row.ratio:.3f}" if not np.isnan(row.ratio) else "  n/a"
    print(
        f"{row.cls:>6}  {row.label:>5}  {row.area_frac:>10.4f}  {row.sample_frac:>11.4f}  {ratio_str:>6}  {row.n_samples:>5}"
    )

# ── step 5: plots ─────────────────────────────────────────────────────────────
plot_kg_bars(df, OUT_PATH)
plot_kg_map(df, africa, africa_grid, africa_kg, valid, grid, NPZ_PATH, OUT_PATH_MAP)
plot_kg_map_grey(df, africa, africa_grid, africa_kg, valid, grid, NPZ_PATH, OUT_PATH_MAP_GRAY)
