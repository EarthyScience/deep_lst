# Onboarding

The workflow for training on a new spatial selection, and a first project. It builds on
`docs/RUNBOOK.md`.

---

## 1. Workflow: training on a new spatial selection

The loop for any sampling ablation. Call the new selection `<sel>`.

**Step 1: draw the selection** (eodata environment, interactive, minutes):

```
cd ../eodata_slim && mamba activate eodata
uv run scripts/site_selection/s2_majortom/spatial_stratified_africa.py \
    --out_dir <somewhere>/<sel> --n_sites 3000 --seed 0 --min_distance 10 --buffer_deg 0 \
    --grid MajorTom --axes KG MODIS_LC ELEVATION FVC \
    --class_exclude '{"MODIS_LC": [-1, 0, 15], "ECO": [-1, 69, 759]}'
```

Knobs: `--n_sites` (sample size), `--axes` (stratification axes among
`KG KG_MAJOR MODIS_LC ECO BIOME ELEVATION FVC AI`; the draw is inverse-frequency weighted over the joint
strata of the chosen axes), `--seed`, `--min_distance` (in grid cells), `--buffer_deg`
(exclusion radius around the field-site evaluation regions). The docstring of the script
(`--help`) documents every option.

Things to know:

- **`--grid MajorTom` is mandatory for deep_lst.** The script's default writes ids of the
  land-only subset grid, which deep_lst would silently misread as global ids.
- The script writes `indices_<name>.npz` + `indices_<name>_meta.json` into `--out_dir`
  (`--name`, default `sites`); copy them to `assets/spatial/` of the deep_lst clone.
- `--fixed_sites <existing .npz>` extends a selection: the new sites are disjoint from the
  existing ones and keep `--min_distance` from them; `--n_sites` then counts only the new
  sites. This is how the spatial holdout `indices_eval_mtg_new_val.npz` was drawn from
  `indices_mtg_new.npz` (seed 1, same axes).
- A selection stores the classes of its own stratification axes only. The verification reads
  `kg_classes` from every selection, so one drawn without the `KG` axis needs
  `add_site_classes.py` (next to the selection script) before it is evaluated.
- **Check disjointness from the holdouts**, once for the evaluation set you will score on and
  once for the validation pixels of training (`_spatial_val`, default `eval_mtg_new_val`):
  ```
  python -c "import numpy as np; a=np.load('assets/spatial/indices_<sel>.npz')['src_indices']; \
  b=np.load('assets/spatial/indices_eval_<holdout>.npz')['src_indices']; print(np.isin(a,b).sum(),'shared')"
  ```
  Redraw with another seed if it is not 0.

**Step 2: statistics** (three GPU jobs, `docs/RUNBOOK.md` section 4.4, ~1 h wall):

```
bash scripts/assets/launch_statistics_mtg.sh <sel>
```

(It submits `scripts/assets/compute_statistics_distributed.sh` once per modality.)

**Step 3: warm the cache with a smoke run, then train** (`docs/RUNBOOK.md` sections 5.2, 5.3), with
`_spatial=<sel>` and `<sel>` in `tag`.

**Step 4: inference on the holdout, then verification** (`docs/RUNBOOK.md` sections 5.5, 5.6).

What is keyed on the selection: `indices_<sel>.npz`, three `*_stats_<sel>.zarr`,
`preload_<sel>_*.pt`. The evaluation side (holdout indices, baselines) stays fixed.

---

## 2. First project: sampling strategy ablations

Goal: understand how the **stratification** (which axes) and the **sample size** (`--n_sites`)
of the training selection affect (a) the balance of the training set and (b) the forecast
skill of the two backbones.

**Part A: balance of the selection, without training.** Cheap, deterministic, and it defines
the design grid for part B. For each selection (axes x n_sites x seed) compute, per axis, the
class shares of the sample against the class shares of the Africa pool (the "sampling
ratio"), an entropy or effective-number-of-classes summary, how many joint strata are
covered, and the nearest-neighbour distance distribution of the sites. Plot balance against
n_sites per stratification. Starting points:

- `deep_lst/exploration/sampling_representativeness.py` and
  `deep_lst/exploration/plots/kg_sampling.py`: sampling-ratio bars and Africa maps for the
  KG axis of one selection (hard-coded paths, single axis; generalise rather than copy).
- `selection_diagnostics.py` next to the selection script in eodata: class marginals, overlap,
  minimum separation and a map of drawn selections against a reference selection.
- The `*_class_counts` and `*_class_labels` in every `indices_<name>_meta.json`.
- The pool itself: `MajorTomLandGrid(5).intersect(get_admin_geometry(continent="Africa"))`
  plus `sample_raster_on_grid` on the auxiliary rasters (paths in
  `eodata_slim/scripts/site_selection/s2_majortom/common.py`).

Selections drawn with the same seed and increasing `--n_sites` should be nested (the sampler
takes prefixes of one weighted permutation); verify with `np.isin` before relying on it.

**Part B: learning curves with training.** Pick a small grid from part A (for example three
sample sizes x two stratifications, both backbones) and run section 1 for each. Two kinds of
curves: validation loss against epoch per run (`metrics.csv`), and final validation loss (or
holdout skill per KG class) against n_sites per backbone. Design caveats:

- An epoch is a fixed number of batches (`training.limit_train_batches`), independent of
  n_sites, so runs at different sample sizes see the same number of gradient steps; the
  sample-size effect is site diversity, not training length. Keep it that way.
- Each configuration costs three statistics jobs, a 20-30 GB cache and one GPU run per
  backbone (hours). Budget before launching a grid.
- Statistics recomputed per selection versus fixed to one reference selection is a design
  choice (the former is the clean default; the latter isolates the sampling effect). Decide
  together with the maintainer.
- The seed is the config key `seed` (default 42); for several seeds per configuration pass
  `seed=<n>` and put it into `tag`.

Whether part A alone answers the balance question, or model training is needed to see
whether balance matters for skill, is open; do part A first, it is needed either way.
