# deep_lst

Neural forecasting of land surface temperature (LST) over Africa from geostationary
satellite observations (SEVIRI) and reanalysis (ERA5, ERA5-Land). An autoregressive MLP or
transformer rolls the hourly LST state forward for up to ten days, driven by the large-scale
atmospheric forcing. Forecasts are verified against climatology, persistence and ERA5-Land
skin temperature.

Code for: `<paper reference / DOI>`. Research code with a single maintainer. The resolved
`config.yml` that every run writes is the ground truth of what was run.

## 1. Installation

Two repositories are cloned side by side: `deep_lst` (this one) and `eodata`, which provides
the data stores and grids. `pyproject.toml` pins `eodata = { path = "../eodata_slim", editable = true }`,
so the sibling directory must be named `eodata_slim`.

```
git clone https://github.com/MWesselkamp/deep_lst.git
git clone -b v0.3_DeepLST https://github.com/EarthyScience/eodata.git eodata_slim
```

eodata needs native GDAL and PROJ libraries. The supported way is its conda environment,
kept active while the deep_lst environment is created so that the Python GDAL bindings link
against it:

```
cd eodata_slim
mamba env create -n eodata -f environment.yaml
mamba activate eodata
uv sync

cd ../deep_lst
uv sync            # creates .venv (Python 3.12) with torch, lightning, climpred and eodata
```

Verify the installation and run the unit tests (tests that need the data stores are skipped):

```
uv run python -c "import deep_lst, eodata; print('ok')"
uv run pytest
```

Run everything from the repository root inside the deep_lst environment (`uv run <cmd>` or
`source .venv/bin/activate`). Hardware: a CUDA GPU for training and inference, about 256 GB of
RAM for the in-memory preload of a 3000-site training selection, and several hundred GB for the
verification of dense regions. All commands below are plain Python entry points; wrap them in
the job scheduler of your system as needed.

## 2. Data and assets

Nothing under `assets/`, `experiments/` and `results/` is tracked. Reproducing the paper needs:

| Item | Content | Source |
|---|---|---|
| data stores | `.eozarr` stores `seviri_2016_2024`, `era5_2014_2025`, `era5_land_2016_2025` (optional `glofas_v4_2014_2025`), hourly, one per modality | built with eodata from LSA SAF SEVIRI LST, ERA5, ERA5-Land and GloFAS; `<DATA_ARCHIVE or "available from the authors on request">` |
| `assets/spatial/indices_<name>.npz` | pixel selections: training selection `mtg_new` (3000 sites) and the evaluation sets | `<ASSET_ARCHIVE>` |
| `assets/temporal/time_indices_<split>.npz` | hour sets of the train, validation, test and statistics splits | `<ASSET_ARCHIVE>` |
| `assets/statistics/<modality>_stats_<name>.zarr` | normalisation statistics per training selection | `<ASSET_ARCHIVE>`, or computed (3.1) |
| `assets/baselines/<kind>/<name>.nc` | reference forecasts per evaluation set | computed (3.4) |
| `experiments/<run>/version_0/checkpoints/best.ckpt` | trained models | `<CHECKPOINT_ARCHIVE>`, or trained (3.2) |

Two locations are set in code, in `ExperimentPaths` in `deep_lst/utils/io.py`: `eozarr_root`,
the directory that holds the stores, and `preload_cache_root`, a scratch directory for the
preload cache (20 to 30 GB per selection, built once by the first run on a selection). Point
both at your machine. Every other path is relative to the repository root:

```
mkdir -p assets/spatial assets/temporal assets/statistics assets/baselines experiments results
```

The temporal split is chronological: training 2016 to 2022, validation blocks in 2023, test
year 2024. Which hours belong to which split is defined by the temporal index files, not by
dates in the config.

## 3. Reproducing the paper

Stages: statistics, training, inference, baselines, verification. Every entry point takes
Hydra overrides (`key=value`, nested keys with dots).

### 3.1 Statistics (once per training selection)

```
python scripts/assets/compute_statistics.py modality=seviri    spatial=mtg_new
python scripts/assets/compute_statistics.py modality=era5      spatial=mtg_new
python scripts/assets/compute_statistics.py modality=era5_land spatial=mtg_new
```

### 3.2 Training the two final models

Both backbones use the smooth-L1 rollout loss, a 24 h training horizon, 168 h of lookback and
70 epochs (an epoch is a fixed number of batches, `training.limit_train_batches`):

```
python deep_lst/train.py model=transformer_rollout loss=smoothl1 horizon=h24 data=full \
    _lookback=168 model.static_encoding=integer model.incremental_loss=false \
    training.max_epochs=70 training.batch_size=128 tag=lb168_int_noinc_e70

python deep_lst/train.py model=mlp_rollout loss=smoothl1 horizon=h24 data=full \
    _lookback=168 model.static_encoding=integer model.incremental_loss=false \
    training.max_epochs=70 tag=lb168_int_noinc_e70
```

The transformer was trained with distributed data parallel on four GPUs (four ranks of batch
128), the MLP on one GPU at batch 512. Runs are named
`<model>_<loss>_<horizon>_<data>_<label>[_<tag>]` (the label is set in `conf/train.yaml`) and
write `experiments/<run>/version_N/` with the resolved `config.yml`, `metrics.csv` and
`checkpoints/{best,last}.ckpt`. Weights & Biases logging is on by default; `wandb_logger=false`
keeps only the CSV logger. `smoke=true` runs one epoch of three batches. The seed is fixed to 42.

The ablations are the same command with one override each, for example `_lookback=24`,
`model.static_encoding=embedding`, `loss=mae`, `model.rollout_loss=false`,
`model.teacher_forcing_loss=false`, `data=noera5`, `horizon=h72`; see the config groups in
section 4.

### 3.3 Inference

```
python scripts/inference/run_inference.py eval=<eval> experiment=<run> version=version_0 best=true
```

Evaluation sets are the files in `conf/eval/`: `train_mtg_new` (the training sites in the test
year), `eval_mtg_new` (the same sites in the 2023 validation blocks, used for the ablations),
and the dense regions
`sahel_niger`, `yangambi` and `zambia`. Defaults in `conf/infer.yaml`: 240 h rollout,
initialisations at 00, 06, 12 and 18 UTC. Output: `results/<run>/<version>/<eval>/forecast.zarr`.

### 3.4 Baselines (once per evaluation set)

```
python scripts/assets/run_baselines.py --region <eval> --test_year 2024
```

Writes persistence, hourly and smoothed climatology (2016 to 2023) and ERA5-Land skin
temperature to `assets/baselines/<kind>/`.

### 3.5 Verification, tables and figures

Three verification passes, named after the dimension they keep, then one derivation step that
needs no cluster:

```
python -m deep_lst.evaluation_climpred.verif_init  --experiment <run> --eval_name <eval> --windows <windows>
python -m deep_lst.evaluation_climpred.verif_pixel --experiment <run> --eval_name <eval> --windows <windows>
python -m deep_lst.evaluation_climpred.verif_anom  --experiment <run> --eval_name <eval> --windows <windows>
python -m deep_lst.evaluation_climpred.evaluate    --experiment <run> --eval_name <eval>
```

Windows condition the scores: `all day night` (time of day, the default), `kg_arid kg_tropical
kg_temperate` and their `_day` / `_night` variants (Köppen-Geiger groups, the paper's tables),
and `--months <m ...>` adds a calendar-month window (give it to `evaluate` as well). Output:
`results/<run>/<version>/<eval>/{verif,scores,tables,plots}`; tables are written as `.tex` and
`.csv`. Ablation tables compare several runs on one evaluation set:

```
python -m deep_lst.evaluation_climpred.tables_ablation --eval_name eval_mtg_new --experiments <run ...> --out tables/<name>
```

## 4. Running your own experiment

`conf/train.yaml` is the composition root. Its `defaults` pick one file from each group:

| Group | Options | Sets |
|---|---|---|
| `model` | `mlp_rollout`, `transformer_rollout`, `mlp_diag` | backbone, hyper-parameters, loss terms |
| `loss` | `smoothl1`, `mse`, `mae` | the step loss |
| `horizon` | `h4` ... `h72` | training horizon in hours |
| `data` | `full`, `full_glofas`, `noera5`, `noera5land`, `nosevaux`, ... | which variables the model sees |

Scalar keys to override: `_spatial` (training selection, resolves to
`assets/spatial/indices_<_spatial>.npz`; its statistics must exist), `_lookback` (hours of input
history), `tag` (appended to the run name; put `_spatial` and `_lookback` there, they are not
part of the name), `training.max_epochs`, `training.batch_size`, `resume_ckpt`. A new training
selection needs new statistics (3.1) and builds its own preload cache; evaluation sets and
baselines are independent of it. `deep_lst/finetune.py` warm-starts from a trained run
(`from_run=<run>`).

## 5. Repository layout

```
deep_lst/
  train.py, finetune.py    training entry points (Hydra)
  inference.py             Forecast: model rollout over an evaluation set, writes forecast.zarr
  lit_module.py            LightningModule wrapper
  dataset/                 preload of the stores into RAM, dataset and data module
  models/                  MLP and transformer backbones
  losses/                  rollout loss (state, teacher-forcing and increment terms)
  baselines/               climatology, persistence and ERA5-Land reference forecasts
  evaluation_climpred/     verification (verif_init, verif_pixel, verif_anom), derived scores, tables, figures
  utils/                   io.py (ExperimentPaths), spatial.py (anchor-to-store mapping), config.py
conf/                      Hydra config groups model / loss / horizon / data / eval; train.yaml, infer.yaml, statistics.yaml
scripts/
  assets/                  compute_statistics.py, run_baselines.py
  inference/               run_inference.py
tests/                     pytest suite
```
