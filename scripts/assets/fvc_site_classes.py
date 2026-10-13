"""Annual-mean FVC per site and its FVC class, written into the selection npz files.

The annual mean is the mean of the twelve monthly climatology means (work_5 derived
FVC_monthly_climatology, 2016-2024) at each site's seviri pixel. Class 1 is bare (annual
mean exactly 0, one fifth of the mtg_new sites, so plain deciles cannot split it); classes
2-10 are nine equal-count bins of the vegetated sites. Bin edges are computed once on the
reference selection (mtg_new) and applied to every selection, so a class means the same
vegetation level on training and holdout sites. Adds the keys `fvc_annual_mean` and
`fvc_classes` (1-10, 0 where the FVC is missing) next to kg_classes etc.; the verification
attaches them as the `fvc_class` pixel coordinate (cubes.attach_classes, masks.FVC_GROUPS).
Also adds `lst_validity_annual_mean`: the site's mean share of valid clear-sky LST hours (%)
over the same years, from the LSTvalidity_monthly_climatology sibling, for the validity
scatter (plots.plot_fvc_pixel_scatter with x_attribute="lst_validity").

    python scripts/assets/fvc_site_classes.py                      # mtg_new (reference) + eval_mtg
    python scripts/assets/fvc_site_classes.py --selections mtg eval_mtg_val
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from deep_lst.utils.io import ExperimentPaths
from deep_lst.utils.spatial import anchor_grid, map_to_store_rows, store_grid

PATHS = ExperimentPaths()
CLIMATOLOGY_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/FVC_monthly_climatology")
VALIDITY_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/LSTvalidity_monthly_climatology")
REFERENCE_SELECTION = "mtg_new"
EDGES_PATH = PATHS.assets_root / "spatial" / "fvc_class_edges.json"
N_CLASSES = 10  # 1 = bare, 2-10 = equal-count bins of the vegetated sites


def annual_mean_fvc():
    """(n_seviri_pixels,) mean of the twelve monthly climatology means, NaN where every month is."""
    monthly = []
    for month in range(1, 13):
        field = xr.open_dataset(CLIMATOLOGY_DIR / f"fvc_climatology_{month:02d}.nc")["fvc_mean"]
        monthly.append(field.values.ravel())
    return np.nanmean(np.stack(monthly), axis=0)


def annual_mean_validity():
    """(n_seviri_pixels,) mean of the twelve monthly LST validity climatologies [%], NaN where never observed."""
    monthly = []
    for month in range(1, 13):
        field = xr.open_dataset(VALIDITY_DIR / f"lst_validity_climatology_{month:02d}.nc")["lst_validity"]
        monthly.append(field.values.ravel())
    return np.nanmean(np.stack(monthly), axis=0)


def site_values(selection, by_pixel):
    """A per-seviri-pixel field sampled at every site of a selection, in the npz's own site order."""
    npz = np.load(PATHS.spatial_indices_path(selection))
    anchor_ids = npz["src_indices"].astype(np.int64)
    rows = map_to_store_rows(anchor_grid("mtg"), anchor_ids, store_grid("seviri"))
    return by_pixel[rows]


def vegetated_edges(values):
    """Eight interior edges splitting the vegetated (FVC > 0) values into nine equal-count bins."""
    vegetated = values[np.isfinite(values) & (values > 0)]
    return np.quantile(vegetated, np.linspace(0, 1, N_CLASSES)[1:-1])


def classify(values, edges):
    """1 for bare (FVC <= 0), 2..10 by vegetated bin, 0 where the value is missing."""
    classes = 2 + np.searchsorted(edges, values, side="right")
    classes[values <= 0] = 1
    classes[~np.isfinite(values)] = 0
    return classes.astype(np.int16)


def write_selection(selection, fvc, classes, validity):
    """Add the three keys to the selection npz, keeping a one-off backup of the original."""
    path = PATHS.spatial_indices_path(selection).resolve()  # follow symlinks (eval_mtg_val -> eval_mtg)
    backup = path.with_suffix(".npz.bak_fvc")
    if not backup.exists():
        shutil.copy2(path, backup)
    contents = dict(np.load(path))
    contents["fvc_annual_mean"] = fvc.astype(np.float32)
    contents["fvc_classes"] = classes
    contents["lst_validity_annual_mean"] = validity.astype(np.float32)
    np.savez(path, **contents)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selections", nargs="+", default=[REFERENCE_SELECTION, "eval_mtg"])
    args = parser.parse_args()

    fvc_by_pixel = annual_mean_fvc()
    validity_by_pixel = annual_mean_validity()

    if EDGES_PATH.exists():
        edges = np.array(json.loads(EDGES_PATH.read_text())["edges"])
        print(f"class edges from {EDGES_PATH}")
    else:
        edges = vegetated_edges(site_values(REFERENCE_SELECTION, fvc_by_pixel))
        EDGES_PATH.write_text(
            json.dumps(
                {
                    "reference": REFERENCE_SELECTION,
                    "classes": "1 = bare (FVC <= 0); 2-10 = vegetated sites binned by these interior edges",
                    "edges": edges.tolist(),
                },
                indent=2,
            )
        )
        print(f"class edges from {REFERENCE_SELECTION} -> {EDGES_PATH}")
    print("vegetated bin edges:", np.round(edges, 3))

    for selection in args.selections:
        fvc = site_values(selection, fvc_by_pixel)
        validity = site_values(selection, validity_by_pixel)
        classes = classify(fvc, edges)
        path = write_selection(selection, fvc, classes, validity)
        counts = np.bincount(classes, minlength=N_CLASSES + 1)
        print(
            f"{selection}: {len(fvc)} sites, missing {counts[0]}, class counts {counts[1:].tolist()}, "
            f"validity median {np.nanmedian(validity):.1f} % -> {path.name}"
        )


if __name__ == "__main__":
    main()
