# Runbook

Operating deep_lst on the BGI cluster of the MPI for Biogeochemistry: prerequisites,
installation, directories, data and assets, and every stage of the pipeline under SLURM.
The overview is in `README.md`.

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
  `tscratch` space, see section 4.6.

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
Python 3.12 itself and creates `.venv` in the clone with everything, including eodata from
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

The repository tracks the Python package, `conf/`, `tests/` and `scripts/` (Python entry points
and SLURM launchers). The directories below are gitignored and have to be set up per clone:

| Directory | Content | Where it comes from |
|---|---|---|
| `assets/` | index files, normalisation statistics, baselines (section 4) | **provided by the maintainer on request**, or computed (statistics, baselines) |
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

Check these locations in a new clone:

1. `ExperimentPaths.preload_cache_root` in `deep_lst/utils/io.py` uses `DEEP_LST_CACHE_ROOT`
   when it is nonempty, otherwise `/Net/Groups/BGI/tscratch/$USER/tmp` (using the login name if
   `USER` is unset or empty). For training, `paths.preload_cache_root=<your-tscratch>` on the command
   line or in the config file takes priority; `paths.eozarr_root` can also override the store
   root. Other entry points use the shared `ExperimentPaths` defaults, including this cache
   fallback. See `docs/CONFIGURATION.md`.
   The data stores (`ExperimentPaths.eozarr_root`,
   `/Net/Groups/BGI/work_5/scratch/WeatherGenerator/data/lowres_old`) are group-readable and
   stay as they are.
2. The SLURM launchers under `scripts/` hard-code the path of the maintainer's clone, most of
   them as `WORKDIR=` near the top; replace it with the path of your clone.

All other paths in `ExperimentPaths` (`assets/`, `experiments/`, `results/`) are relative to
the current working directory. **Always run from the repository root**; the launchers `cd`
there and Hydra pins its run dir to it.

---

## 4. Data and assets

### 4.1 The data stores

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

### 4.2 Asset overview and dependencies

```
assets/
  spatial/    indices_<name>.npz              which pixels   (per selection / evaluation set)
  temporal/   time_indices_<split>.npz        which hours    (fixed)
  statistics/ <modality>_stats_<name>.zarr    normalisation  (per training selection)
  baselines/  <kind>/<name>.nc                reference forecasts (per evaluation set)
  static/     <kind>_init_state_<name>.nc     site states at init (optional, verification only)
```

The important dependency: **statistics and the preload cache are keyed on the training
selection.** A new selection needs new statistics (section 4.4) and builds a new cache
(section 4.6). Baselines are keyed on evaluation sets and are untouched by a new training
selection.

`assets/static/` is optional: the vegetation and hydrological states at init that the
verification attaches where they exist, written by `scripts/assets/fvc_init_state.py` and
`scripts/assets/hydro_init_state.py`.

### 4.3 Spatial indices

One `assets/spatial/indices_<name>.npz` per selection. deep_lst reads a single key,
`src_indices`, holding **anchor-grid cell ids**:

- **Training selections** are ids on the global 5 km MajorTom grid (`MajorTomGrid(5)` in
  eodata); `anchor: mtg` in the config.
- **Evaluation sets** are either MajorTom selections as well (a holdout of sites) or dense
  SEVIRI pixel ids for a region (`anchor: seviri`). Their config lives in `conf/eval/<name>.yaml`
  and is selected with `eval=<name>` at inference; its `splits.test.spatial` key names the
  selection, which `ExperimentPaths.spatial_indices_path` resolves to `indices_<spatial>.npz`
  (a training selection such as `mtg_new` is evaluated by naming it there).

At preload, every modality maps the anchor ids to its own store rows by nearest neighbour
(`deep_lst/utils/spatial.py::map_to_store_rows`), so one file serves all modalities and no
per-modality index files exist. The other keys in the file (`kg_classes`, `modis_lc_classes`,
`dem_values`, ...) are per-pixel stratification classes used for evaluation-by-class and for
balance analysis; class labels, seed, axes, `n_sites` and per-class counts are recorded in
the sidecar `indices_<name>_meta.json` written by the selection script.

A training selection must not contain sites of the holdout it is validated on (`_spatial_val`
in `conf/train.yaml`) or of the holdout evaluation set it is scored on;
`docs/ONBOARDING.md`, section 1, has the check.

### 4.4 Normalisation statistics

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
3000 sites; 20-70 min per modality): `scripts/assets/compute_statistics_distributed.sh` is the
launcher for one modality, and `bash scripts/assets/launch_statistics_mtg.sh <name>` submits
all three. Training resolves `stats_spatial` from `_spatial`, so all three zarrs must exist
before you train on a selection, and a fourth from `modality=glofas_v4` for `data=full_glofas`.

The store, the value transform and the variables that get increment statistics come from
`conf/sources/eozarr.yaml`, the same file training composes. The transform is recorded in the
zarr; training stops before the preload if it differs from the transform of the run's source.
Recomputing statistics changes the preload fingerprint (below), so the cache is rebuilt
afterwards.

### 4.5 Temporal indices

`assets/temporal/time_indices_<split>.npz`, key `time_indices` (hourly positions on the store
time axis). The names used by the config are `_temporal_train`, `_temporal_val`,
`splits.test.temporal` in `conf/train.yaml` and `temporal: statistics` in
`conf/statistics.yaml`. They are generated with the temporal-selection scripts in
`eodata_slim/scripts/time_selection/` and provided together with the other assets; you should
not need to regenerate them.

### 4.6 The preload cache

The datasets do not read zarr during training. At start-up `MultimodalDataset.preload_shared`
(`deep_lst/dataset/dataset.py`) loads every needed column for every selected pixel into RAM,
normalised, and saves that dict once as

```
<preload_cache_root>/preload_<spatial>_<fingerprint>.pt
```

The fingerprint hashes the anchor ids and, per modality, the store name, the `load` list, the
`transform` where one is set and the modification time of the statistics zarr. Any change to
those rebuilds the cache (30-60 min); changes to model, loss, lookback, horizon or `data=`
inputs reuse it, unless the `data=` option adds a modality, as `full_glofas` does. A 3000-site selection produces a
20-30 GB file, which is why training jobs request 256 GB of RAM. There is no separate
"precache" command: the first training or inference job on a selection builds the cache as a
side effect, all later jobs load it in seconds.

**Race condition:** two jobs that start on the same missing cache both write the same
`<cache>.tmp` and one of them crashes with `FileNotFoundError`. Run one job first (a
`smoke=true` run is enough to warm the cache), or chain jobs with `sbatch --dependency=afterany:<jobid>`.

Old caches are never deleted automatically; watch your tscratch quota.

### 4.7 Baselines

`assets/baselines/{persistence,climatology_hourly,climatology_smoothed_hourly,era5_land}/<name>.nc`
are physical reference forecasts per evaluation set, extracted by
`scripts/assets/run_baselines.py --region <eval name> --test_year <year>` (launcher:
`scripts/assets/run_baselines.sh`). Only verification needs them; training does not.

---

## 5. Running the pipeline

The stages are: statistics (once per training selection) -> training -> inference on an
evaluation set -> verification. All entry points are Python scripts taking Hydra overrides,
run from the repository root inside the deep_lst venv. On the cluster, wrap them in SLURM
jobs; the launchers are tracked under `scripts/`. The training launcher looks like this
(`scripts/training/train_mlp.sh`, abridged; `scripts/training/train_transformer.sh` differs in
walltime and job name only; adapt `WORKDIR`):

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=scripts/out/train_mlp.%j

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKDIR=/Net/Groups/BGI/people/<you>/deep_lst
cd $WORKDIR
source $WORKDIR/.venv/bin/activate
export PYTHONPATH=$WORKDIR

srun python3 ${WORKDIR}/deep_lst/train.py "$@"
```

Everything after the script name is passed through as Hydra overrides:
`sbatch scripts/training/train_mlp.sh model=mlp tag=<tag>`.

### 5.1 Weights & Biases

Training logs to the W&B project **`deep_lst`** (set in `deep_lst/train.py`) under the account
you are logged in as; the run name is the one from `docs/ARCHITECTURE.md`, section 1. One-time setup:

```
ml proxy
uv run wandb login          # paste your API key; it is stored in ~/.netrc
```

The code sets neither entity nor mode, but the W&B SDK reads environment variables: export
`WANDB_ENTITY=<team>` in the launcher if the runs should land in a shared team, and
`WANDB_MODE=offline` to log locally only. Local run files go to `wandb/` in the repo.
`smoke=true` or `wandb_logger=false` disable W&B; the CSV logger stays on.

### 5.2 Smoke test

Before anything else, check that your clone works end to end on a selection whose statistics
exist:

```
sbatch scripts/training/train_mlp.sh model=mlp _spatial=<selection> smoke=true
squeue -u $USER
tail -f scripts/out/train_mlp.<jobid>
```

This is a 1-epoch, 3-batch run; the first one on a selection also builds the preload cache,
so expect 30-60 min. Success is `experiments/<run>_smoke/version_0/` containing
`metrics.csv` and `checkpoints/`.

### 5.3 Training

```
sbatch scripts/training/train_mlp.sh model=mlp _spatial=<selection> \
       training.max_epochs=40 tag=<tag>
sbatch --gres=gpu:4 scripts/training/train_transformer.sh model=transformer _spatial=<selection> \
       training.batch_size=128 training.max_epochs=40 tag=<tag>
```

Conventions of the current campaigns: global batch 512 everywhere, the transformer as DDP on
4 GPUs with per-rank batch 128 (it does not fit on one A40 at long lookbacks), the MLP on one
GPU at 512. Since v18 the recipe (`window.lookback=168`, `model.static_encoding=integer`,
`objective.terms.increment=0`, `training.max_epochs=70`) is the config default; encode any
deviation from it in `tag`. Measured: ~28 min/epoch transformer, ~8 min/epoch MLP (~7 h for
70 epochs with the preload cache present).

### 5.4 Monitoring and learning curves

- `squeue -u $USER`; job log in `scripts/out/`.
- W&B: the run appears under its run name.
- `experiments/<run>/version_N/metrics.csv`: sparse CSV with per-step rows
  (`train_loss_step_step`) and per-epoch rows (`train_loss_step_epoch`,
  `val_loss_step_epoch`, `val_loss_logits`, `val_loss_rollout`, ...). Drop NaNs per column
  before plotting; group by `epoch`. The checkpoint monitor is `val_loss_step_epoch_log`
  (the log of `val_loss_step_epoch`).

There is no plotting utility for learning curves in the package yet. The random seed is the
config key `seed` (default 42, passed to `pl.seed_everything` in `deep_lst/train.py`); for repeat
runs pass `seed=<n>` and put it into `tag`, because the seed is not part of the run name.

### 5.5 Inference

Roll a trained model out over an evaluation set (`conf/eval/<eval>.yaml`):

```
python scripts/inference/run_inference.py eval=<eval> experiment=<run> version=version_0 \
    best=true inference_horizon=240 'splits.test.init_hours=[0, 6, 12, 18]'
```

Run it as a GPU job (1 GPU, 32 CPUs, 256 GB, a few hours). The existing inference launchers
add `unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU` before `srun`; keep that line.
Output: `results/<run>/<version>/<eval>/forecast.zarr`. The inference preload
cache is keyed on the evaluation set, so the first job per set builds it (same race caveat as
training). Use a validation-period evaluation set for model selection and keep the test-year
sets for final numbers.

### 5.6 Verification

Three verification jobs per run, then one evaluation job depending on them:

```
python -m deep_lst.evaluation_climpred.verif_init  --experiment <run> --eval_name <eval> --version version_0 --windows all day night
python -m deep_lst.evaluation_climpred.verif_pixel --experiment <run> --eval_name <eval> --version version_0 --windows all day night
python -m deep_lst.evaluation_climpred.verif_anom  --experiment <run> --eval_name <eval> --version version_0 --windows all day night
python -m deep_lst.evaluation_climpred.evaluate    --experiment <run> --eval_name <eval> --version version_0
```

A fourth pass, `verif_diurnal` (called with `--experiment`, `--eval_name` and `--version`), feeds
the diurnal figures; `evaluate` skips them when it has not run.
`scripts/evaluation/run_climpred.sh` submits the four passes for every evaluation set and
`evaluate` as a dependent job, with the memory request of each set.

The verification jobs are memory-bound (`big` partition, several hundred GB to 1 TB for dense
regions; a holdout of a few hundred sites needs much less). `evaluate` derives skill against
climatology and persistence, tables and figures into
`results/<run>/<version>/<eval>/{verif,scores,tables,plots}`. Windows such as `kg_arid`,
`kg_tropical`, `kg_temperate` give per-climate-class skill, which is what relates model
performance to the composition of the training set.

---

### 5.7 Diagnostic regression, the joint head and init fill

The model is a one-step function: a window ending at hour tau, the forcings at tau+1, outputs
for tau+1. A prognostic (LST in `prognostics`) is carried forward as `state(tau) + increment`, so
the rollout needs a number at the init hour and a missing observation there makes the whole
forecast NaN. A diagnostic (`diagnostics`) is an immediate value with no prior. A diagnostic's
observed past enters the state window like a prognostic's (`model.diagnostic_history`, default
on; off = drivers only), hidden from t0+1 on because it is never carried.

- `prognostics: []`, `diagnostics: [LST]` (`conf/data/full_lstdiag.yaml`): the diagnostic
  regression, LST at every lead as an immediate estimate from the LST history and the drivers;
  `deep_lst/losses/rollout.py` skips the state and rollout terms (`objective=diagnostic` is the
  rollout objective with the rollout term off, for this data option). Forecasts exist at every init.
- LST in both lists (`conf/data/full_joint.yaml`): the joint head. The diagnostic copy is written
  as `LST_diag`; its history is not added to the window twice.
- `init_fill` (`none | gated | always`) is one question asked in two phases: where does the
  rollout take its first prior? In training (`objective.init_fill`, variants `objective=rollout_gated` and
  `objective=rollout_always`) gated starts from the observation where present and from the analysis
  elsewhere, so cloudy inits enter the rollout term; always starts every sample from the
  analysis. Both train the analysis head, the diagnostic copy at t0 from the window ending at
  t0, with the t0 slot hidden in `init_hide_fraction` of the samples so filling has a target. At
  inference (`init_fill` in `conf/infer.yaml`, null = as trained) the nowcast is the analysis for
  such models and the history nowcast (the lead-1 diagnostic one hour early) for any other joint
  model, which allows `gated` without retraining; `always` needs the analysis. An inference whose
  `init_fill` deviates from the trained one writes `forecast_<init_fill>.zarr`, verified with
  `--init_fill <value>` (outputs in `<kind>_<value>`).

Every rollout store written since 2026-10-23 has a uniform lead axis 0..H: lead 0 holds the state
the rollout started from (`LST_pred`), the init observation (`LST_obs`), the nowcast (`LST_diag_pred`,
NaN without filling) and a boolean `LST_init_filled` per init and pixel. The evaluation drops lead 0,
so scores are unchanged, and older stores remain readable. Compare filling runs to observation-
anchored runs in the `init_obs` window (`--windows all init_obs init_gap init_filled`), and read the
dependence on the gap length from the `gap_*` windows (tables `scores_gap`, `skill_gap`).
