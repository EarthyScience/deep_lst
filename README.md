# deep_lst

Data-driven forecasting of land surface temperature (LST) from satellite observations and
reanalysis. LST is estimated diagnostically and prognostically from medium-range, large-scale
synoptic weather conditions (ERA5, ERA5-Land) and small- to medium-scale surface state (SEVIRI).

Built on PyTorch Lightning, configured with Hydra, logged with Weights & Biases.

> **Status:** research code, single-maintainer. Interfaces and configs change; when this README
> and the code disagree, the code wins. Please fix the README when you notice.

---

## 1. Before you start (cluster prerequisites)

Everything runs on the BGI cluster under SLURM. You need:

- **A cluster account** with access to the `gpu`, `work` and `big` partitions.
- **Outbound network via the proxy.** Nothing on the cluster reaches the internet unless the
  proxy module is loaded. Add this to your `~/.bashrc`; it is needed for `git`, `uv`, `mamba`
  and `wandb`:
  ```
  ml proxy
  ```
- **A conda/mamba installation** (e.g. Miniforge) in your scratch space. It provides the native
  libraries (GDAL, PROJ, HDF) that the eodata data stack needs; `uv` itself comes from that
  environment.
- **Two storage areas.** Code and small outputs live in your people directory. Large
  intermediate files (the preload cache, tens of GB per training selection) go to your
  `tscratch` space, see section 6.6.

---

## 2. Installation

deep_lst and eodata are two repositories, cloned side by side, **each installed with its own
environment**. deep_lst imports eodata as an editable dependency from the sibling directory:
`pyproject.toml` pins `eodata = { path = "../eodata_slim", editable = true }`, so the sibling
must be named `eodata_slim`.

```
cd /Net/Groups/BGI/people/<you>
git clone git@github.com:MWesselkamp/deep_lst.git
git clone git@github.com:EarthyScience/eodata.git eodata_slim
```

**2.1 eodata** (branch `v0.3_DeepLST`), following its own README:

```
cd eodata_slim
git checkout v0.3_DeepLST
mamba env create -n eodata -f environment.yaml
mamba activate eodata
uv sync
```

**2.2 deep_lst.** Keep the `eodata` conda environment active while syncing: the Python GDAL
bindings are compiled against the `libgdal` it provides. `uv sync` downloads the pinned
Python 3.12 itself and creates `deep_lst/.venv` with everything, including eodata from
`../eodata_slim`.

```
cd ../deep_lst
uv sync
```

**Which environment for what.** Anything under `eodata_slim/` (site selection, temporal
selection, store conversion) runs in the eodata environment, i.e. `mamba activate eodata`
and `uv run` from inside `eodata_slim`. Anything under `deep_lst/` runs in the deep_lst venv,
i.e. `source .venv/bin/activate` (what the SLURM jobs do) or `uv run` from inside `deep_lst`.

Verify:

```
cd deep_lst
uv run python -c "import deep_lst, eodata; from osgeo import gdal; print('ok')"
uv run pytest            # tests that need the data stores are skipped without them
```

Development tooling: `uv run pre-commit run --all-files` (ruff format + lint).

---

## 3. Directory structure and what the repository does not contain

The repository tracks the Python package, `conf/`, `tests/` and the Python entry points under
`scripts/`. Everything else is gitignored and has to be set up per clone:

| Directory | Content | Where it comes from |
|---|---|---|
| `assets/` | index files, normalisation statistics, baselines (section 6) | **provided by the maintainer on request**, or computed (statistics, baselines) |
| `scripts/**/*.sh` | SLURM launchers | not tracked; section 7 gives the underlying commands and a template |
| `scripts/out/` | SLURM job logs | create it yourself; SLURM does not create it and jobs die silently without it |
| `experiments/` | per-run outputs: config, metrics, checkpoints | created by training |
| `results/` | forecasts, verification, tables, plots | created by inference and evaluation |
| `wandb/` | local W&B run files | created by training |

Create the skeleton after cloning:

```
cd deep_lst
mkdir -p assets/spatial assets/temporal assets/statistics assets/baselines \
         experiments results scripts/out
```

and place the provided files as

```
assets/spatial/indices_<name>.npz         pixel selections (training selections and evaluation sets)
assets/temporal/time_indices_<split>.npz  hourly index sets for the train / val / test / statistics splits
assets/statistics/<modality>_stats_<name>.zarr   z-score statistics per (modality, training selection)
assets/baselines/<kind>/<name>.nc         persistence / climatology / ERA5(-Land) reference forecasts
```

Two hard-coded paths need attention in a new clone:

1. `ExperimentPaths.preload_cache_root` in `deep_lst/utils/io.py` points at the maintainer's
   tscratch directory, which you cannot write to. Point it at your own tscratch directory.
   The data stores (`ExperimentPaths.eozarr_root`, `/Net/Groups/BGI/tscratch/weathergenerator/`)
   are group-readable and stay as they are.
2. If you receive SLURM launchers, each sets `WORKDIR=` near the top; replace it with your
   clone's path.

All other paths in `ExperimentPaths` (`assets/`, `experiments/`, `results/`) are relative to
the current working directory. **Always run from the repository root**; the launchers `cd`
there and Hydra pins its run dir to it.

---

## 4. Repository layout

```
deep_lst/
  train.py            training entry point (Hydra)          -> experiments/<run>/version_N/
  finetune.py         warm-start from a trained run
  lit_module.py       LightningModule wrapper
  dataset/            preload of the zarr stores into RAM, MultimodalDataset, LitDataModule
  models/             MLP and transformer backbones
  losses/             rollout loss (state + teacher-forcing + increment terms), diagnostic losses
  preprocessing/      derived static inputs
  evaluation/         inference (Forecast), metrics, tables, plots
  evaluation_climpred/ verification against baselines (verif_init / verif_pixel / verif_anom -> evaluate)
  baselines/          climatology / persistence / ERA5(-Land) reference forecast extraction
  exploration/        one-off analyses (sampling representativeness, diurnal cycle, ...)
  utils/              io.py (ExperimentPaths), spatial.py (anchor-to-store mapping), config.py
conf/                 Hydra config groups model / loss / horizon / data / eval; train.yaml, infer.yaml, statistics.yaml
scripts/
  assets/             compute_statistics.py, run_baselines.py
  inference/          run_inference.py
tests/                pytest suite
```

---

## 5. Configuration and run naming (Hydra)

`conf/train.yaml` is the composition root. Its `defaults` pick one file from each group:

| Group | Options | Sets |
|---|---|---|
| `model` | `mlp_rollout`, `transformer_rollout`, `mlp_diag` | backbone and its hyper-parameters, loss terms, `_model_name` |
| `loss` | `smoothl1`, `mse`, `mae` | `model.loss_fn`, `_loss` |
| `horizon` | `h4 h8 h12 h24 h48 h72` | forecast horizon in hours for every modality, `_horizon` |
| `data` | `full`, `full_glofas`, `noera5`, `noera5land`, `nosevaux`, ... | which variables the model sees as inputs / prognostics, `_data` |

Scalar keys you will override often (all in `conf/train.yaml`):

| Key | Meaning |
|---|---|
| `_spatial` | name of the training pixel selection, resolves to `assets/spatial/indices_<_spatial>.npz`. `stats_spatial` follows it, so the statistics for that name must exist. |
| `_temporal_train`, `_temporal_val` | names of the temporal split files under `assets/temporal/` |
| `_lookback` | hours of input history |
| `tag` | free suffix appended to the run name |
| `training.max_epochs` | epochs |
| `training.batch_size` | per-process batch (DDP multiplies by the number of GPUs) |
| `training.limit_train_batches` | **an epoch is a fixed number of batches** (default 2500), not a pass over the data |
| `smoke` | `smoke=true`: 1 epoch of 3 batches, no W&B, run name gets `_smoke` |
| `wandb_logger`, `csv_logger` | loggers, both on by default |
| `resume_ckpt` | path to a `last.ckpt` to continue a walltime-killed run |

Any key can be overridden on the command line, nested keys with dots:

```
python deep_lst/train.py model=transformer_rollout loss=smoothl1 horizon=h24 data=full \
    _spatial=<selection> _lookback=72 training.batch_size=128 training.max_epochs=40 tag=<tag>
```

The **run name** is assembled in `deep_lst/train.py` as

```
<_model_name>_<_loss>_<_horizon>_<_data>_v14[_<tag>][_smoke]
```

(`v14` is a data-version label set in `conf/train.yaml`). `_spatial` and `_lookback` are
**not** part of the name, so put them into `tag`; otherwise two runs land in the same
`experiments/<run>/` directory as `version_0`, `version_1`, ... and are hard to tell apart.

Every run writes `experiments/<run>/version_N/` with:

- `config.yml`: the fully resolved config (the ground truth of what was run)
- `metadata.json`: git hash, SLURM job id, node, timestamp
- `metrics.csv`: the learning curves (section 7.4)
- `checkpoints/best.ckpt`, `checkpoints/last.ckpt`

The temporal split is chronological: training years, a validation year, a test year; which
hours belong to which split is defined by the temporal index files, not by dates in the config.

---

## 6. Data and assets

### 6.1 The data stores

Training reads the full `.eozarr` stores written by eodata, hourly:

| Modality | Native grid | Role |
|---|---|---|
| `seviri` | 0.05 deg equiangular, Africa disc | LST (the prognostic), FVC, DEM, land/water mask, land cover |
| `era5` | 0.25 deg | synoptic forcing, static soil/vegetation classes |
| `era5_land` | 0.1 deg | near-surface state (soil temperature/moisture, t2m, d2m) |
| `glofas_v4` | daily | optional, only with `data=full_glofas` |

Store names are set in `conf/train.yaml` (`_seviri_store`, ...), located under
`ExperimentPaths.eozarr_root`. Every store is indexed by its own grid cell id and an hourly
time axis. The `_<modality>_load` blocks in `conf/train.yaml` list what is read from each
store into RAM; the `data=` group decides which of those the model actually sees.

### 6.2 Asset overview and dependencies

```
assets/
  spatial/    indices_<name>.npz              which pixels   (per selection / evaluation set)
  temporal/   time_indices_<split>.npz        which hours    (fixed)
  statistics/ <modality>_stats_<name>.zarr    normalisation  (per training selection)
  baselines/  <kind>/<name>.nc                reference forecasts (per evaluation set)
```

The important dependency: **statistics and the preload cache are keyed on the training
selection.** A new selection needs new statistics (section 6.4) and builds a new cache
(section 6.6). Baselines are keyed on evaluation sets and are untouched by a new training
selection.

### 6.3 Spatial indices

One `assets/spatial/indices_<name>.npz` per selection. deep_lst reads a single key,
`src_indices`, holding **anchor-grid cell ids**:

- **Training selections** are ids on the global 5 km MajorTom grid (`MajorTomGrid(5)` in
  eodata); `anchor: mtg` in the config.
- **Evaluation sets** are either MajorTom selections as well (a holdout of sites) or dense
  SEVIRI pixel ids for a region (`anchor: seviri`). Their config lives in `conf/eval/<name>.yaml`
  and is selected with `eval=<name>` at inference; `ExperimentPaths.eval_indices_path` resolves
  `indices_eval_<name>.npz`, or `indices_<name>.npz` when the eval name starts with `train_`
  (evaluating on a training selection).

At preload, every modality maps the anchor ids to its own store rows by nearest neighbour
(`deep_lst/utils/spatial.py::map_to_store_rows`), so one file serves all modalities and no
per-modality index files exist. The other keys in the file (`kg_classes`, `modis_lc_classes`,
`dem_values`, ...) are per-pixel stratification classes used for evaluation-by-class and for
balance analysis; class labels, seed, axes, `n_sites` and per-class counts are recorded in
the sidecar `indices_<name>_meta.json` written by the selection script.

A training selection must not contain sites of the holdout evaluation set it is scored on;
section 8 has the check.

### 6.4 Normalisation statistics

`assets/statistics/<modality>_stats_<name>.zarr`: per-variable mean/std/min/max computed on
the **raw store values over the selection's pixels and the `statistics` timesteps**, plus the
increment statistics used to weight the loss. Computed by `scripts/assets/compute_statistics.py`
with `conf/statistics.yaml`, one modality per run:

```
python scripts/assets/compute_statistics.py modality=seviri    spatial=<name>
python scripts/assets/compute_statistics.py modality=era5      spatial=<name>
python scripts/assets/compute_statistics.py modality=era5_land spatial=<name>
```

Submit each as a SLURM job on the `gpu` partition (1 GPU, 8 CPUs, 32 GB, 2 h is enough for
3000 sites; 20-70 min per modality). Training resolves `stats_spatial` from `_spatial`, so
all three zarrs must exist before you train on a selection. Recomputing statistics changes
the preload fingerprint (below), so the cache is rebuilt afterwards.

### 6.5 Temporal indices

`assets/temporal/time_indices_<split>.npz`, key `time_indices` (hourly positions on the store
time axis). The names used by the config are `_temporal_train`, `_temporal_val`,
`test_dataset.temporal` in `conf/train.yaml` and `temporal: statistics` in
`conf/statistics.yaml`. They are generated with the temporal-selection scripts in
`eodata_slim/scripts/time_selection/` and provided together with the other assets; you should
not need to regenerate them.

### 6.6 The preload cache

The datasets do not read zarr during training. At start-up `MultimodalDataset.preload_shared`
(`deep_lst/dataset/dataset.py`) loads every needed column for every selected pixel into RAM,
normalised, and saves that dict once as

```
<preload_cache_root>/preload_<spatial>_<fingerprint>.pt
```

The fingerprint hashes the anchor ids, the store names, the `load` lists and the modification
time of the statistics zarrs. Any change to those rebuilds the cache (30-60 min); changes to
model, loss, lookback, horizon or `data=` inputs reuse it. A 3000-site selection produces a
20-30 GB file, which is why training jobs request 256 GB of RAM. There is no separate
"precache" command: the first training or inference job on a selection builds the cache as a
side effect, all later jobs load it in seconds.

**Race condition:** two jobs that start on the same missing cache both write the same
`<cache>.tmp` and one of them crashes with `FileNotFoundError`. Run one job first (a
`smoke=true` run is enough to warm the cache), or chain jobs with `sbatch --dependency=afterany:<jobid>`.

Old caches are never deleted automatically; watch your tscratch quota.

### 6.7 Baselines

`assets/baselines/{persistence,climatology_hourly,climatology_smoothed_hourly,era5_land}/<name>.nc`
are physical reference forecasts per evaluation set, extracted by
`scripts/assets/run_baselines.py --region <eval name> --test_year <year>`. Only verification
needs them; training does not.

---

## 7. Running the pipeline

The stages are: statistics (once per training selection) -> training -> inference on an
evaluation set -> verification. All entry points are Python scripts taking Hydra overrides,
run from the repository root inside the deep_lst venv. On the cluster, wrap them in SLURM
jobs; a minimal launcher that the campaigns use looks like this
(`scripts/training/train.sh`, adapt `WORKDIR`):

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=scripts/out/train.%j

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKDIR=/Net/Groups/BGI/people/<you>/deep_lst
cd $WORKDIR
source $WORKDIR/.venv/bin/activate
export PYTHONPATH=$WORKDIR

srun python deep_lst/train.py "$@"
```

Everything after the script name is passed through as Hydra overrides:
`sbatch scripts/training/train.sh model=mlp_rollout tag=<tag>`.

### 7.1 Weights & Biases

Training logs to the W&B project **`deep_lst`** (set in `deep_lst/train.py`) under the account
you are logged in as; the run name is the one from section 5. One-time setup:

```
ml proxy
uv run wandb login          # paste your API key; it is stored in ~/.netrc
```

The code sets neither entity nor mode, but the W&B SDK reads environment variables: export
`WANDB_ENTITY=<team>` in the launcher if the runs should land in a shared team, and
`WANDB_MODE=offline` to log locally only. Local run files go to `wandb/` in the repo.
`smoke=true` or `wandb_logger=false` disable W&B; the CSV logger stays on.

### 7.2 Smoke test

Before anything else, check that your clone works end to end on a selection whose statistics
exist:

```
sbatch scripts/training/train.sh model=mlp_rollout _spatial=<selection> smoke=true
squeue -u $USER
tail -f scripts/out/train.<jobid>
```

This is a 1-epoch, 3-batch run; the first one on a selection also builds the preload cache,
so expect 30-60 min. Success is `experiments/<run>_smoke/version_0/` containing
`metrics.csv` and `checkpoints/`.

### 7.3 Training

```
sbatch scripts/training/train.sh model=mlp_rollout _spatial=<selection> \
       training.max_epochs=40 tag=<tag>
sbatch --gres=gpu:4 scripts/training/train.sh model=transformer_rollout _spatial=<selection> \
       training.batch_size=128 training.max_epochs=40 tag=<tag>
```

Conventions of the current campaigns: global batch 512 everywhere, the transformer as DDP on
4 GPUs with per-rank batch 128 (it does not fit on one A40 at long lookbacks), the MLP on one
GPU at 512, `max_epochs=40`. Measured: ~28 min/epoch transformer, ~8 min/epoch MLP. Pass
`_lookback` and any model override explicitly and encode them in `tag`.

### 7.4 Monitoring and learning curves

- `squeue -u $USER`; job log in `scripts/out/`.
- W&B: the run appears under its run name.
- `experiments/<run>/version_N/metrics.csv`: sparse CSV with per-step rows
  (`train_loss_step_step`) and per-epoch rows (`train_loss_step_epoch`,
  `val_loss_step_epoch`, `val_loss_logits`, `val_loss_rollout`, ...). Drop NaNs per column
  before plotting; group by `epoch`. The checkpoint monitor is `val_loss_step_epoch_log`
  (the log of `val_loss_step_epoch`).

There is no plotting utility for learning curves in the package yet. The random seed is fixed
to 42 in `deep_lst/train.py` (`pl.seed_everything`); repeat runs with different seeds need
that line parameterised.

### 7.5 Inference

Roll a trained model out over an evaluation set (`conf/eval/<eval>.yaml`):

```
python scripts/inference/run_inference.py eval=<eval> experiment=<run> version=version_0 \
    best=true inference_horizon=240 'test_dataset.init_hours=[0, 6, 12, 18]'
```

Run it as a GPU job (1 GPU, 32 CPUs, 256 GB, a few hours). The existing inference launchers
add `unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU` before `srun`; keep that line.
Output: `results/<run>/<version>/<eval>/forecast.zarr`. The inference preload
cache is keyed on the evaluation set, so the first job per set builds it (same race caveat as
training). Use a validation-period evaluation set for model selection and keep the test-year
sets for final numbers.

### 7.6 Verification

Three verification jobs per run, then one evaluation job depending on them:

```
python -m deep_lst.evaluation_climpred.verif_init  --experiment <run> --eval_name <eval> --version version_0 --windows all day night
python -m deep_lst.evaluation_climpred.verif_pixel --experiment <run> --eval_name <eval> --version version_0 --windows all day night
python -m deep_lst.evaluation_climpred.verif_anom  --experiment <run> --eval_name <eval> --version version_0 --windows all day night
python -m deep_lst.evaluation_climpred.evaluate    --experiment <run> --eval_name <eval> --version version_0
```

The verification jobs are memory-bound (`big` partition, ~400 GB for dense regions; a
holdout of a few hundred sites needs much less). `evaluate` derives skill against
climatology and persistence, tables and figures into
`results/<run>/<version>/<eval>/{verif,scores,tables,plots}`. Windows such as `kg_arid`,
`kg_tropical`, `kg_temperate` give per-climate-class skill, which is what relates model
performance to the composition of the training set.

---

## 8. Workflow: training on a new spatial selection

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
`KG MODIS_LC ECO BIOME ELEVATION FVC`; the draw is inverse-frequency weighted over the joint
strata of the chosen axes), `--seed`, `--min_distance` (in grid cells), `--buffer_deg`
(exclusion radius around the field-site evaluation regions). The dedicated README in
`eodata_slim/scripts/site_selection/` describes the selection module in detail.

Things to know:

- **`--grid MajorTom` is mandatory for deep_lst.** The script's default writes ids of the
  land-only subset grid, which deep_lst would silently misread as global ids.
- The script writes fixed names `indices_sites.npz` + `indices_meta.json` into `--out_dir`;
  copy them to `deep_lst/assets/spatial/indices_<sel>.npz` and `indices_<sel>_meta.json`.
- **Check disjointness from the holdout evaluation set** you will score on:
  ```
  python -c "import numpy as np; a=np.load('assets/spatial/indices_<sel>.npz')['src_indices']; \
  b=np.load('assets/spatial/indices_eval_<holdout>.npz')['src_indices']; print(np.isin(a,b).sum(),'shared')"
  ```
  Redraw with another seed if it is not 0.

**Step 2: statistics** (three GPU jobs, section 6.4, ~1 h wall):

```
for m in seviri era5 era5_land; do
  sbatch --job-name=stats_${m}_<sel> scripts/assets/stats.sh modality=$m spatial=<sel>
done
```

(`stats.sh` being the launcher template of section 7 with `compute_statistics.py` as entry
point and the resources of section 6.4.)

**Step 3: warm the cache with a smoke run, then train** (section 7.2, 7.3), with
`_spatial=<sel>` and `<sel>` in `tag`.

**Step 4: inference on the holdout, then verification** (sections 7.5, 7.6).

What is keyed on the selection: `indices_<sel>.npz`, three `*_stats_<sel>.zarr`,
`preload_<sel>_*.pt`. The evaluation side (holdout indices, baselines) stays fixed.

---

## 9. First project: sampling strategy ablations

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
- The `*_class_counts` and `*_class_labels` in every `indices_<name>_meta.json`.
- The pool itself: `MajorTomLandGrid(5).intersect(get_admin_geometry(continent="Africa"))`
  plus `sample_raster_on_grid` on the auxiliary rasters (paths in
  `eodata_slim/scripts/site_selection/s2_majortom/common.py`).

Selections drawn with the same seed and increasing `--n_sites` should be nested (the sampler
takes prefixes of one weighted permutation); verify with `np.isin` before relying on it.

**Part B: learning curves with training.** Pick a small grid from part A (for example three
sample sizes x two stratifications, both backbones) and run section 8 for each. Two kinds of
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
- The seed is fixed at 42 in `train.py`; multiple seeds per configuration need that changed.

Whether part A alone answers the balance question, or model training is needed to see
whether balance matters for skill, is open; do part A first, it is needed either way.

---

## 10. Gotchas

- `FileNotFoundError: ... .tmp` at start-up: the preload-cache race (6.6). Run the first job
  on a new selection or evaluation set alone.
- `FileNotFoundError` / missing group for `<modality>_stats_<sel>.zarr`: statistics not yet
  computed for that selection (6.4).
- Job dies immediately with no log: `scripts/out/` does not exist.
- Two runs overwrite each other's `version_0`: `_spatial` / `_lookback` were not put into `tag`.
- The transformer at long lookback does not fit on one A40 at batch 512: 4 GPUs x 128.
- Inference or baseline jobs fail on memory settings: add `unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU`.
