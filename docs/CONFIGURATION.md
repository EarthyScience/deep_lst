# Configuration

Every entry point is configured with [Hydra](https://hydra.cc) from the files in `conf/`. This
document describes every root file, every group, every option and every key. The yaml files
themselves carry no explanation; this is the place for it.

Related: how a run is named and how a `config.yml` of an older layout is read is in
`docs/ARCHITECTURE.md`, section 1. The assets that keys point to (index files, statistics,
preload cache) are in `docs/RUNBOOK.md`, section 4.

---

## 1. Layout

One root file per entry point:

| Root file | Entry point | Composes |
|---|---|---|
| `conf/train.yaml` | `deep_lst/train.py` | the groups `sources`, `model`, `objective`, `loss`, `optim`, `horizon`, `data` |
| `conf/finetune.yaml` | `deep_lst/finetune.py` | all of `conf/train.yaml`, plus its own keys |
| `conf/infer.yaml` | `scripts/inference/run_inference.py` | the group `eval` |
| `conf/sensitivity.yaml` | `scripts/sensitivity/run_sensitivity.py` | all of `conf/infer.yaml`, plus `sensitivity` |
| `conf/statistics.yaml` | `scripts/assets/compute_statistics.py` | the group `sources` |

A group is a directory under `conf/`; an option is one file in it. Each group writes one
section of the config and nothing else:

| Group | Section it writes | Default | Options |
|---|---|---|---|
| `sources` | `sources` | `eozarr` | `eozarr` |
| `model` | `model` | `mlp` | `mlp`, `transformer` |
| `objective` | `objective` | `rollout` | `rollout`, `rollout_gated`, `rollout_always`, `diagnostic` |
| `loss` | `objective.pointwise` | `smoothl1` | `smoothl1`, `mse`, `mae` |
| `optim` | `optim` | `adamw_cosine` | `adamw_cosine` |
| `horizon` | `window` (the key `horizon`) | `h24` | `h4`, `h8`, `h12`, `h24`, `h48`, `h72`, `h120` |
| `data` | `features` (one option also `modality_ids`) | `full` | section 3.2 |
| `eval` | `splits.test` and `eval_name` | `gobabeb` | section 5.1 |

On the command line, `group=option` picks a file and `key=value` overrides one value, with dots
for nested keys:

```
python deep_lst/train.py model=transformer objective=rollout loss=smoothl1 horizon=h48 data=full \
    window.lookback=72 training.max_epochs=40 tag=<tag>
```

`python deep_lst/train.py --cfg job --resolve` prints the composed config without running
anything, and `--help` lists the options of every group. Every run saves its composed config as
`config.yml` in its run directory; that file, not `conf/`, is the record of what a run did.

Every root file ends with a `hydra:` block that keeps Hydra from changing the working directory,
from writing its own output directory and from logging below warning level. It is not meant to
be edited.

---

## 2. `conf/train.yaml`

### Training data and cache paths

`paths.eozarr_root` and `paths.preload_cache_root` override the data-store and preload-cache
roots used by `deep_lst/train.py`, for all three training/validation/test splits. The defaults
are defined in `ExperimentPaths` in `deep_lst/utils/io.py`, also used by the `PATHS` singleton
in `deep_lst/dataset/dataset.py`. Both config values default to `null`, so these defaults apply.

The cache default uses `DEEP_LST_CACHE_ROOT` when it is nonempty; otherwise it uses
`/Net/Groups/BGI/tscratch/$USER/tmp` (the login name is used if `USER` is unset or empty).
An explicit `paths.preload_cache_root` takes priority over this default. Relative paths remain
relative to the repository working directory. The overrides are saved in the run's `config.yml`.

```
uv run python deep_lst/train.py paths.eozarr_root=/data/eozarr paths.preload_cache_root=/scratch/deep_lst-cache
```

Cache keys are unchanged; use a separate cache root when the source data content differs.

These Hydra overrides apply to training. Standalone inference, statistics and fine-tuning use
the shared `ExperimentPaths` defaults, including the cache default described above.
Model, loss and data-processing settings are unchanged.

### 2.1 Selections and splits

| Key | Default | Meaning |
|---|---|---|
| `config_version` | `2` | layout of the config; written into every `config.yml`. Not to be overridden. |
| `_spatial` | `mtg_new` | training pixel selection: the index file `indices_<_spatial>.npz` under `assets/spatial/` |
| `_spatial_val` | `eval_mtg_new_val` | pixel selection that training validates and tests on. The default is a spatial holdout of 500 sites disjoint from the default `_spatial`. Set it to the value of `_spatial` to validate on the training pixels. |
| `_temporal_train` | `train_full_2023b10` | hours of the training split: the index file `time_indices_<name>.npz` under `assets/temporal/`. The default is every hour of 2016 to 2022. |
| `_temporal_val` | `val_2023b10` | hours of the validation split. The default is days 11 to 20 of every month of 2023 (2880 hours), so every season is validated. |
| `_version` | `v18` | data and selection generation; first token of the run name |

These five keys are shorthands. They are used through `splits`, which says for each of `train`,
`val` and `test` which pixels and hours it reads:

| Key under `splits.<split>` | Meaning |
|---|---|
| `spatial` | pixel selection of the split |
| `stats_spatial` | selection whose statistics normalise the data. It is the training selection for all three splits, and it stays the training selection when an evaluation set replaces `spatial` at inference. |
| `anchor` | grid the selection's cell ids live on: `mtg` (MajorTom 5 km grid, used for site selections) or `seviri` (SEVIRI pixel ids, used for dense regions) |
| `temporal` | hours of the split (name of a temporal index file) |
| `init_hours` | optional; keep only forecast starts at these UTC hours. Set by `conf/infer.yaml`, not used in training. |

Defaults: `train` reads `_spatial` and `_temporal_train`; `val` reads `_spatial_val` and
`_temporal_val`; `test` reads `_spatial_val` and `test_period_00` (the first 14 days of the test
year 2024), which is the short test pass at the end of training. The full test year is
evaluated by inference (section 5).

Train, validation and test are separated by calendar year, so no gap for lookback or horizon is
needed between them.

### 2.2 `window`

| Key | Default | Meaning |
|---|---|---|
| `window.lookback` | `168` | hours of history the model sees before the forecast start |
| `window.horizon` | `24` (group `horizon`) | hours forecast during training. Every training sample spans `lookback + horizon + 1` hours. |

Both values hold for every modality. `horizon=h48` sets `window.horizon` to 48; the options are
listed in section 1. Inference can roll out further than the trained horizon (section 5).

### 2.3 `modality_ids`

Default `[era5, era5_land, seviri]`: the modalities that are loaded and preloaded. A modality
that is defined in `sources` but missing here is not read. `data=full_glofas` adds `glofas_v4`.

### 2.4 `training`

| Key | Default | Meaning |
|---|---|---|
| `training.max_epochs` | `70` | number of epochs. Also sets the length of the learning-rate schedule (section 3.6). |
| `training.batch_size` | `512` | batch size per process. With several GPUs every GPU gets a batch of this size. |
| `training.limit_train_batches` | `2500` | batches per epoch. **An epoch is this fixed number of batches, not a pass over the data.** Keep it an integer: the schedule length is `max_epochs` x `limit_train_batches`. |
| `training.limit_val_batches` | `500` | validation batches per epoch |
| `training.limit_test_batches` | `1.0` | share of the test split used after training (`1.0` = all of it) |
| `training.overfit_batches` | `0.0` | debugging: train and validate on this share of the data only |
| `training.num_workers` | `24` | data-loading worker processes per split |
| `training.pin_memory` | `true` | page-locked memory for the transfer to the GPU |
| `training.prefetch_factor` | `2` | batches each worker prepares ahead |

Set in `deep_lst/train.py` and not configurable: mixed precision (bf16), gradient clipping at
norm 1.0, and one process per visible GPU.

### 2.5 Logging and run flags

| Key | Default | Meaning |
|---|---|---|
| `wandb_logger` | `true` | log to the W&B project `deep_lst` |
| `csv_logger` | `true` | write the run directory with `config.yml`, `metadata.json`, `metrics.csv` and the checkpoints. Leave it on. |
| `early_stopping` | `false` | stop when `objective.checkpoint_monitor` has not improved for 7 epochs. Off because the learning-rate schedule is sized to `max_epochs` and the best epoch of a run is usually among its last; stopping early cuts the decay short. The best checkpoint is kept either way. |
| `profiler` | `false` | not read by any code |
| `smoke` | `false` | `smoke=true`: 1 epoch of 3 batches per split, no W&B, run name gets `_smoke`. For checking that a setup works. |
| `tag` | `null` | free suffix of the run name. Put into it whatever distinguishes the run and is not in the name already (selection, seed, lookback). |
| `seed` | `42` | random seed of the initialisation and the data order |
| `hpo` | `false` | `hpo=true` replaces `optim.learning_rate`, `optim.weight_decay`, `model.dropout`, `model.hidden_sizes` and `training.batch_size` by the result of a hyper-parameter search (`deep_lst/nas/hpo.py`) |
| `hpo_path` | `nas/hpo_best.json` | file that `hpo=true` reads |
| `resume_ckpt` | `null` | path to a `last.ckpt`; continues a run that hit the walltime, with its epoch, optimiser and schedule. The continued run logs to the next `version_N` directory. |

---

## 3. Groups of `conf/train.yaml`

### 3.1 `sources`

The data stores, one block per modality. `conf/sources/eozarr.yaml` is the single definition
that training, inference, the baselines and the statistics script share.

| Key under `sources.<modality>` | Meaning |
|---|---|
| `store` | name of the `.eozarr` store |
| `load` | variables read from the store, by data type (below). This is what reaches memory, not what the model sees; the model's variables are chosen in the `data` group and have to be listed here. |
| `transform` | optional; a value transform applied before the normalisation. `signed_log1p` is sign(x) * log(1 + \|x\|), for variables that span orders of magnitude. The statistics have to be computed under the same transform; training checks this before it reads data. |
| `increment_vars` | variables for which the statistics script also computes the statistics of the 1-hour increment. Read by `scripts/assets/compute_statistics.py` only. Listing a variable here does not make it a prognostic of the model. |
| `quality_mask` | optional and not set: `{field, se_max, target}` sets `target` to missing where the metadata variable `field` exceeds `se_max` |

Data types under `load`:

| Data type | Holds |
|---|---|
| `data` | variables on the store's own time step (hourly; daily for `glofas_v4`) |
| `data_daily` | daily variables of an hourly store |
| `data_static` | continuous fields without time axis |
| `data_static_categorical` | class codes without time axis |
| `metadata` | quality fields of the observations, kept as stored |

What `eozarr` defines:

| Modality | Store | Loaded | `increment_vars` |
|---|---|---|---|
| `era5` | `era5_2014_2025` | `data`: sp, u10, v10, ssrd, strd, tp, tcwv, tcc; `data_static_categorical`: tvh, tvl, lsm, slt | none |
| `era5_land` | `era5_land_2016_2025` | `data`: stl1, stl2, swvl1, swvl2, t2m, d2m, skt | swvl1, swvl2, stl1, stl2 |
| `glofas_v4` | `glofas_v4_2014_2025` | `data`: rowe, sd, swir, dis24; `transform: signed_log1p` | none |
| `seviri` | `seviri_2016_2024` | `data`: LST, MLE, MH; `data_daily`: FVC; `data_static`: DEM; `data_static_categorical`: LWMASK, LANDCOV; `metadata`: quality_flag, standard_error | LST, MLE, MH |

`glofas_v4` is a daily store; its values are repeated over the 24 hours of a day. ERA5-Land
`skt` is loaded for the ERA5-Land reference forecast of the verification and is a model input
only in the data options that name it.

Changing `store`, `load` or `transform` changes the preload cache key, so the cache is rebuilt
(`docs/RUNBOOK.md`, section 4.6). Changing the `data` option does not.

### 3.2 `data`

The role of every variable in the model, per modality, written to `features.<modality>`:

| Key | Meaning |
|---|---|
| `inputs` | variables the model receives for the hour it predicts: the forcings at that hour, the static fields and the computed features |
| `context_vars` | variables the model receives over the whole lookback window. A prognostic listed here is not passed twice; its history is in the state window already. |
| `prognostics` | state variables. The model predicts their 1-hour increment, and the forecast is carried forward as state + increment, so a forecast needs an observation (or an estimate) at its start. |
| `diagnostics` | variables predicted directly at every lead, without a previous state |

A variable has to be in the `load` list of its source, or be one of the computed features
`solar_zenith`, `doy_cos`, `doy_sin`, `hod_cos`, `hod_sin`, which are derived from position and
time when a sample is served (`docs/ARCHITECTURE.md`, section 2). The order of the modalities in
`sources` and of the variables in each list fixes the channel order of the model's tensors.

Options. "As `full`" means the lists of `full` with the stated difference.

| Option | Content |
|---|---|
| `full` | `era5`: sp, u10, v10, ssrd, strd, tp, tcwv, tcc, solar_zenith as inputs and context. `era5_land`: stl1, stl2, swvl1, swvl2, t2m, d2m, solar_zenith as inputs and context. `seviri`: FVC, DEM, LWMASK, LANDCOV, solar_zenith, doy_cos, doy_sin, hod_cos, hod_sin as inputs; LST as prognostic and context; no diagnostic. |
| `full_era5static` | as `full`, plus the ERA5 static fields tvh, tvl, lsm, slt as `era5` inputs |
| `full_skt` | as `full_era5static`, plus ERA5-Land skt as input and context |
| `full_glofas` | as `full_era5static`, plus the modality `glofas_v4` with swir, rowe, dis24 as inputs and context. Sets `modality_ids` to include `glofas_v4`. |
| `full_joint` | as `full`, with LST as prognostic and as diagnostic (the joint head). The diagnostic copy is the model's estimate of the start state, needed by `init_fill`. |
| `full_lstdiag` | as `full`, with LST as diagnostic only: no prognostic, and no `seviri` context variable |
| `noera5` | as `full`, without any `era5` input or context |
| `noera5land` | as `full_era5static`, without any `era5_land` input or context |
| `nosevaux` | as `full_era5static`, without the `seviri` inputs FVC, DEM, LWMASK, LANDCOV |
| `noera5static_nostl` | as `full`, without ERA5-Land stl1, stl2 |
| `noera5static_skt` | as `noera5static_nostl`, plus ERA5-Land skt as input and context |

All options except `full_glofas` read the same stores and variables, so they share one preload
cache per selection.

### 3.3 `model`

The architecture only. Nothing about the loss or the optimiser lives here.

Keys of both backbones:

| Key | `mlp` | `transformer` | Meaning |
|---|---|---|---|
| `model.name` | `mlp` | `transformer` | the backbone |
| `model.dropout` | `0.005` | `0.005` | dropout probability |
| `model.activation` | `gelu` | `gelu` | activation function: `gelu` or `relu` |
| `model.static_encoding` | `integer` | `integer` | how class codes (LANDCOV, LWMASK, the ERA5 statics) enter the model: `integer` = as numbers, `embedding` = through a learnable embedding per variable |
| `model.diagnostic_history` | `true` | `true` | `true`: the observed past of a diagnostic enters the state window like that of a prognostic, and is hidden from the forecast start on because a diagnostic is never carried forward. `false`: the window holds the prognostics only, and a diagnostic is predicted from the forcings alone. Has no effect on a diagnostic that is a prognostic as well. |

Keys of `mlp`:

| Key | Default | Meaning |
|---|---|---|
| `model.hidden_sizes` | `[1024, 512, 256, 256, 128]` | width of every hidden layer |
| `model.use_residual` | `true` | adds a skip connection around the first hidden layer that has the same width as the one before it (here the second 256) |
| `model.use_layer_norm` | `true` | layer normalisation in every hidden layer |

Keys of `transformer`:

| Key | Default | Meaning |
|---|---|---|
| `model.d_model` | `256` | width of a token |
| `model.n_heads` | `4` | attention heads |
| `model.n_layers` | `4` | encoder layers |
| `model.dim_feedforward` | `512` | width of the feed-forward block of a layer |
| `model.attention` | `causal` | `causal`: a history token attends to its past only. `full`: every token attends to the whole window. The token that makes the prediction sees the whole window either way. |
| `model.norm_first` | `false` | `true` normalises before attention and feed-forward instead of after |
| `model.encoder_weights` | `None` | not read by any code |

### 3.4 `objective`

What training minimises. The section is turned into the objective object as it stands
(`deep_lst.lit_module.build_objective`); `deep_lst/losses/rollout.py` has no default for any of
these keys.

For every sample the objective scores the leads 1 to `window.horizon`, in two branches:

- **teacher-forced**: every lead is predicted on its own from the observed window that ends
  one hour before it.
- **rollout**: the leads are predicted in sequence, and each prediction of a prognostic is
  written into the window of the next lead, as at inference. This branch needs a prognostic;
  without one it is empty.

| Key | Default | Meaning |
|---|---|---|
| `objective._target_` | `deep_lst.losses.RolloutLoss` | class of the objective |
| `objective.kind` | `rollout` | `rollout` or `diagnostic`. Read by the check of objective against data option (section 6) and by inference. |
| `objective.terms.teacher_forcing` | `1.0` | weight of the state and diagnostic errors in the teacher-forced branch |
| `objective.terms.rollout` | `1.0` | weight of the state and diagnostic errors in the rollout branch. `0` skips the branch altogether. |
| `objective.terms.increment` | `0.0` | weight of the error of the predicted 1-hour increment, divided by the increment's standard deviation from the statistics; counted in both branches |
| `objective.pointwise` | group `loss` | function every term is scored with (section 3.5) |
| `objective.detach_rollout` | `false` | `false`: gradients flow back through all earlier steps of the rollout. `true`: a prediction is cut from the graph before it enters the next window, so every step is trained for its own error only. |
| `objective.min_valid_fraction` | `0.05` | a term counts as zero for a batch in which less than this share of its prediction-target pairs is valid (target observed, prediction not missing) |
| `objective.init_fill` | `none` | where the training rollout takes its first state from. `none`: the observation at the forecast start; samples without one drop out of the rollout branch. `gated`: the observation where there is one, else the model's own estimate. `always`: the model's estimate for every sample. |
| `objective.init_hide_fraction` | `0.5` | used with `init_fill` `gated` or `always`: share of samples in which the observation at the forecast start is hidden from the window when the model estimates the start state, so that it learns to fill a missing one |
| `objective.checkpoint_monitor` | `val_loss_step_epoch_log` | validation metric that selects `best.ckpt` (lower is better) and that early stopping follows |

A weight multiplies its term, and a weight of 0 skips the term. At least one weight has to be
non-zero.

With `init_fill` `gated` or `always` the objective has one more part, the **analysis**: the
diagnostic copy of the prognostic is predicted for the forecast start itself, from the window
that ends there. That estimate is what fills a missing start state, in training and at
inference.

Logged metrics, per stage (`train`, `val`, `test`): `<stage>_loss_step` is the total,
`<stage>_loss_logits` the teacher-forced branch, `<stage>_loss_rollout` the rollout branch and
`<stage>_loss_analysis` the analysis. `val_loss_step_epoch` is the epoch mean of the validation
total and `val_loss_step_epoch_log` its logarithm.

Options:

| Option | Differs from `rollout` in | Needs |
|---|---|---|
| `rollout` | | a data option with a prognostic |
| `rollout_gated` | `init_fill: gated` | `data=full_joint` |
| `rollout_always` | `init_fill: always` | `data=full_joint` |
| `diagnostic` | `kind: diagnostic`, `terms.rollout: 0.0`, `checkpoint_monitor: val_loss_step_epoch` | `data=full_lstdiag` |

`diagnostic` is the regression of a variable that is never carried forward: only the
teacher-forced diagnostic term remains.

### 3.5 `loss`

The function that turns predictions and targets into one number. It is applied to the entries
where prediction and target are both present and returns their mean.

| Option | Function | Keys |
|---|---|---|
| `smoothl1` | smooth L1 (Huber): quadratic for errors below `beta`, linear above | `beta: 1.0`. Values are normalised, so 1.0 is one standard deviation. |
| `mse` | mean squared error | |
| `mae` | mean absolute error | |

`_target_` names the function and `_partial_: true` tells Hydra to hand it over uncalled; any
other key of the option is passed to the function as an argument.

### 3.6 `optim`

`adamw_cosine` is AdamW with a linear warm-up followed by a cosine decay.

| Key | Default | Meaning |
|---|---|---|
| `optim.learning_rate` | `3.7e-4` | learning rate at the end of the warm-up |
| `optim.weight_decay` | `0.0001` | weight decay of AdamW |
| `optim.warmup_steps` | `20000` | batches over which the learning rate rises linearly from a tenth of `learning_rate` to `learning_rate`. `0` = no warm-up. |
| `optim.eps` | `1.0e-8` | epsilon of AdamW |

After the warm-up the learning rate follows a cosine down to 1e-6, which it reaches at batch
`training.max_epochs` x `training.limit_train_batches`. Changing either of the two therefore
changes the schedule.

### 3.7 `horizon`

`h<N>` sets `window.horizon` to N hours (section 2.2).

---

## 4. `conf/finetune.yaml`

Continues training from the weights of an existing run. It contains all of `conf/train.yaml`,
so every key and group above applies, with these additions and changed defaults:

| Key | Default | Meaning |
|---|---|---|
| `from_run` | required | `<run>/<version>` of the run to start from, e.g. `from_run=<run>/version_0` |
| `checkpoint` | `best` | `best` or `last`: which checkpoint of that run |
| `optim.learning_rate` | `5e-5` | lower than in training |
| `optim.warmup_steps` | `0` | no warm-up |
| `training.max_epochs` | `25` | |

`model.hidden_sizes` and `model.use_residual` are taken from the source run, whatever the
command line says, so that its weights fit.

---

## 5. `conf/infer.yaml`

Rolls a trained run out over an evaluation set. The run's own `config.yml` supplies the model,
the objective, the sources and the features; `conf/infer.yaml` adds what is specific to the
inference and replaces the test split.

| Key | Default | Meaning |
|---|---|---|
| `experiment` | required | name of the run |
| `version` | `version_0` | version directory of the run |
| `eval_name` | set by the `eval` option | name of the evaluation set; the forecasts are stored under it |
| `best` | `true` | `true` loads `best.ckpt`, `false` loads `last.ckpt` |
| `land_water_mask` | `true` | keep land pixels only (LANDCOV classes 1 to 16); drops ocean and inland water |
| `inference_horizon` | `240` | length of the rollout in hours. Has to be at least the horizon the run was trained with. |
| `init_fill` | `null` | what a forecast starts from when the observation at its start is missing. `null`: as the run was trained (`objective.init_fill`). `none`: no forecast for such a start. `gated`: the model's own estimate where the observation is missing. `always`: the model's estimate for every start. |
| `splits.test.init_hours` | `[0, 6, 12, 18]` | forecasts start at these UTC hours only. `null` = every hour. |
| `analysis.batch_size` | `24768` | samples per batch |
| `analysis.num_workers` | `16` | data-loading worker processes |
| `analysis.prefetch_factor` | `4` | batches each worker prepares ahead |

`init_fill` `gated` and `always` need a run with LST as prognostic and diagnostic
(`data=full_joint`). The estimate is the analysis if the run was trained with
`objective.init_fill` `gated` or `always`. For a joint run trained with `none` it is the
diagnostic prediction for the first lead, used one hour early; `always` is not available for
such a run. An `init_fill` that differs from the trained one writes its forecasts to a store of
its own, `forecast_<init_fill>.zarr`, next to the run's `forecast.zarr`.

In every forecast store, lead 0 holds the state the forecast started from, the observation at
the start and, per start and pixel, whether the start state was filled.

### 5.1 `eval`

An option names the evaluation set and overrides `splits.test`. The `spatial` name needs its
index file under `assets/spatial/`, the `temporal` name its file under `assets/temporal/`.

| Option | `spatial` | `anchor` | `temporal` |
|---|---|---|---|
| `eval_mtg_new_val` | `eval_mtg_new_val` | `mtg` | `test` |
| `eval_mtg` | `eval_mtg` | `mtg` | `test` |
| `eval_mtg_val` | `eval_mtg` | `mtg` | `val_2023b10` |
| `eval_mtg_new` | `mtg_new` | `mtg` | `val_2023b10` |
| `train_mtg_new` | `mtg_new` | `mtg` | `test` |
| `train_mtg` | `mtg` | `mtg` | `test` |
| `antananarivo`, `bouamir`, `dahra`, `gobabeb`, `johannesburg`, `passage`, `sahel_niger`, `west_africa`, `yangambi` | `eval_<option>` | `seviri` | `test` |
| `zambia` | `eval_zambia_drought` | `seviri` | `test` |
| `zambia_drought_seasonal` | `eval_zambia_drought` | `seviri` | `test_zambia_seasonal`, with `init_hours: [0]` |

`test` is the test year 2024 (8760 hours from 1 January). The options with `anchor: mtg` are site selections (holdouts, or
the training sites over the test year); those with `anchor: seviri` are dense regions.

---

## 6. `conf/sensitivity.yaml`

Measures how much a trained run depends on each of its inputs: one input at a time is
perturbed, the forecast is repeated and compared with the unperturbed one. It contains all of
`conf/infer.yaml`, plus:

| Key | Default | Meaning |
|---|---|---|
| `sensitivity.method` | `permute_days` | the perturbation: whole days of the input are shuffled, per pixel |
| `sensitivity.block` | `month` | `month`: days are shuffled within their calendar month, which keeps the seasonal cycle and destroys the day-to-day anomalies. `all`: days are shuffled over the whole record, which destroys the seasonal cycle as well. |
| `sensitivity.variables` | `null` | inputs to perturb, as `["<modality>.<variable>", ...]`. `null` = every input that is read from a store and varies in time. |
| `sensitivity.seed` | `0` | random seed of the shuffle |
| `sensitivity.score` | `true` | score the perturbed forecasts against the run's own forecast after writing them |

---

## 7. `conf/statistics.yaml`

Computes the normalisation statistics of one modality over one selection. The store, the
transform and the `increment_vars` come from the `sources` group (section 3.1).

| Key | Default | Meaning |
|---|---|---|
| `modality` | required | `seviri`, `era5`, `era5_land` or `glofas_v4`; one modality per run |
| `spatial` | `mtg_new` | selection the statistics are computed on. Training looks the statistics up by `splits.<split>.stats_spatial`, so this has to be the training selection of the runs that use them. |
| `temporal` | `statistics` | hours the statistics are computed on (name of a temporal index file) |
| `anchor` | `mtg` | grid of the selection |
| `num_workers` | `8` | data-loading workers |

---

## 8. What has to fit together

`deep_lst/train.py` and `deep_lst/finetune.py` stop before reading any data when:

- `objective=diagnostic` is combined with a data option that has a prognostic or no diagnostic;
- `objective.init_fill` is `gated` or `always` and a prognostic has no diagnostic copy
  (anything but `data=full_joint`);
- the statistics of a modality were computed under another `transform` than its source names.

Not checked, and to be kept in mind:

- The statistics for `_spatial` have to exist for every modality in `modality_ids`.
- `_spatial`, `seed`, `window.lookback` and any other override are not part of the run name;
  put them into `tag` (`docs/ARCHITECTURE.md`, section 1).
- `inference_horizon` at inference is independent of `window.horizon` at training, but not
  smaller.
