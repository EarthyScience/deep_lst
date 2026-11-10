# Architecture

How a run is configured and named, what the computed input features mean for existing
checkpoints, and the pitfalls that follow from the design. The role of every package is in
`README.md`.

---

## 1. Configuration and run naming (Hydra)

`conf/train.yaml` is the composition root. Its `defaults` pick one file from each group; a
group writes one section of the config and nothing else:

| Group | Options | Default | Section | Sets |
|---|---|---|---|---|
| `model` | `mlp`, `transformer` | `mlp` | `model` | the architecture only: backbone, sizes, dropout, activation, `static_encoding`, `diagnostic_history` |
| `objective` | `rollout`, `rollout_gated`, `rollout_always`, `diagnostic` | `rollout` | `objective` | the training objective: class (`_target_`), `kind`, the weight of each loss term (`terms`), `detach_rollout`, `min_valid_fraction`, `init_fill`, `init_hide_fraction`, `checkpoint_monitor` |
| `loss` | `smoothl1`, `mse`, `mae` | `smoothl1` | `objective.pointwise` | the function every loss term is scored with |
| `optim` | `adamw_cosine` | `adamw_cosine` | `optim` | `learning_rate`, `weight_decay`, `warmup_steps`, `eps` |
| `horizon` | `h4` ... `h120` | `h24` | `window` | `window.horizon`, the forecast horizon in hours |
| `data` | the files in `conf/data/` | `full` | `features` | per modality the `inputs`, `prognostics`, `diagnostics` and `context_vars` of the model; `full_glofas` also sets `modality_ids` |
| `sources` | `eozarr` | `eozarr` | `sources` | per modality the `store`, what is `load`ed from it, the value `transform` and the `increment_vars`; shared with `conf/statistics.yaml` |

`python deep_lst/train.py --help` lists the options of every group; `docs/CONFIGURATION.md`
describes every key and option of every config file.

The objective is built from its section by `hydra.utils.instantiate`
(`deep_lst.lit_module.build_objective`); the entry points pass it no loss setting, and
`deep_lst/losses/rollout.py` has no default for any of them, so `conf/objective/` is the only
place a value is set. `terms` holds one weight per loss term: `teacher_forcing`, `rollout`,
`increment`. A weight multiplies its term and 0 skips it; `rollout: 0` skips the whole rollout
branch, `increment` weights the increment term in both branches. `objective=diagnostic` is the
rollout objective with the rollout term off, for a data option without a prognostic
(`data=full_lstdiag`). `train.py` checks the objective against the data option before any data is
read (`deep_lst.utils.config.validate_config`): `objective=diagnostic` needs no prognostic and at
least one diagnostic, `init_fill` gated or always needs a diagnostic copy of every prognostic
(`data=full_joint`).

`splits` in `conf/train.yaml` says which pixels and hours `train`, `val` and `test` read
(`spatial`, `stats_spatial`, `anchor`, `temporal`); the eval group (`conf/eval/`) overrides
`splits.test` at inference. The dataset dict of a split, which the datasets, the collator and
the baselines consume, is not in the config: `deep_lst.utils.config.split_views` builds it from
`splits` x `sources` x `window` x `features`.

Scalar keys you will override often (all in `conf/train.yaml`):

| Key | Meaning |
|---|---|
| `_spatial` | name of the training pixel selection, resolves to `assets/spatial/indices_<_spatial>.npz`. `stats_spatial` follows it, so the statistics for that name must exist. |
| `_spatial_val` | name of the pixel selection that training validates and tests on; the default is a spatial holdout disjoint from the default `_spatial`. Normalisation stays on the statistics of `_spatial`. |
| `_temporal_train`, `_temporal_val` | names of the temporal split files under `assets/temporal/` |
| `window.lookback` | hours of input history |
| `tag` | free suffix appended to the run name |
| `seed` | random seed, default 42; not part of the run name, so put it into `tag` |
| `training.max_epochs` | epochs |
| `training.batch_size` | per-process batch (DDP multiplies by the number of GPUs) |
| `training.limit_train_batches` | **an epoch is a fixed number of batches** (default 2500), not a pass over the data |
| `smoke` | `smoke=true`: 1 epoch of 3 batches, no W&B, run name gets `_smoke` |
| `wandb_logger`, `csv_logger` | loggers, both on by default |
| `resume_ckpt` | path to a `last.ckpt` to continue a walltime-killed run |

Any key can be overridden on the command line, nested keys with dots:

```
python deep_lst/train.py model=transformer objective=rollout loss=smoothl1 horizon=h24 data=full \
    _spatial=<selection> window.lookback=72 training.batch_size=128 training.max_epochs=40 tag=<tag>
```

The **run name** is built from the chosen options by `deep_lst.utils.config.experiment_name`,
extended by `tag` and `_smoke` in `deep_lst/train.py`:

```
<_version>_<model>_<objective>_<loss>_<horizon>_<data>[_<tag>][_smoke]
```

for example `v18_mlp_rollout_smoothl1_h24_full`. `_version` is the data/selection generation
set in `conf/train.yaml`. Existing run directories keep their names: runs before the objective
became a group are named `<_version>_<model>_<loss>_<horizon>_<data>` (with `mlp_gated` and
`mlp_always` for what is now `objective=rollout_gated` and `objective=rollout_always`), runs
before v18 `<model>_<loss>_<horizon>_<data>_<_version>`. The
recipe (lookback 168, integer statics, no incremental loss, 70 epochs) is the config default
and not part of the name. `_spatial`, `seed` and any override are **not** part of the name either,
so put them into `tag`; otherwise two runs land in the same `experiments/<run>/` directory
as `version_0`, `version_1`, ... and are hard to tell apart.

Every run writes `experiments/<run>/version_N/` with:

- `config.yml`: the fully resolved config (the ground truth of what was run)
- `metadata.json`: git hash, seed, tag, SLURM job id, node, timestamp
- `metrics.csv`: the learning curves (`docs/RUNBOOK.md`, section 5.4)
- `checkpoints/best.ckpt`, `checkpoints/last.ckpt`

The temporal split is chronological: training years, a validation year, a test year; which
hours belong to which split is defined by the temporal index files, not by dates in the config.

### Config versions and old runs

`config_version: 2` in `conf/train.yaml` marks the layout above and is written into every
`config.yml`. A `config.yml` without it was written by the layout before: loss terms and
optimiser keys under `model`, three dataset dicts `train_dataset` / `val_dataset` /
`test_dataset` with the stores, the window and the role lists repeated per split and per
modality. Such a file is never rewritten. Inference, finetuning, the sensitivity runs and
`scripts/assets/hydro_init_state.py` pass it through `deep_lst.utils.config.migrate_config`
when they load it, which moves the keys as follows and checks that `split_views` of the result
gives back the three saved dataset dicts:

| Before | Now |
|---|---|
| `model.step_loss: RolloutLoss` | `objective._target_`, `objective.kind: rollout` |
| `model.loss_fn` | `objective.pointwise` (`loss` group) |
| `model.teacher_forcing_loss`, `model.rollout_loss`, `model.incremental_loss` | `objective.terms.teacher_forcing`, `.rollout`, `.increment` (true = 1.0, false = 0.0) |
| `model.detach_rollout`, `model.init_fill`, `model.init_hide_fraction`, `model.checkpoint_monitor` | the same names under `objective` |
| `model.learning_rate`, `model.weight_decay`, `model.warmup_steps` | the same names under `optim` |
| `_lookback`, `<split>_dataset.<modality>.lookback` / `.horizon` | `window.lookback`, `window.horizon` |
| `<split>_dataset.<modality>.store` / `.load` / `.transform` | `sources.<modality>` |
| `<split>_dataset.<modality>.inputs` / `.prognostics` / `.diagnostics` / `.context_vars` | `features.<modality>` |
| `<split>_dataset.spatial` / `.stats_spatial` / `.anchor` / `.temporal` / `.init_hours` | `splits.<split>` |
| `_experiment`, `_model_name`, `_loss`, `_horizon`, `_data` | the run name in `name`, built by `experiment_name` |
| `model.probabilistic`, `model.scheduler_T0`, `_sigma_prognostics`, `_sigma_diagnostics` | dropped, nothing read them |

Where an old config omits `teacher_forcing_loss` or `rollout_loss` (the v13 and v14 runs), the
migration sets the term on, which is what the code of the time did. Before this layout the
code and `conf/` disagreed on one default (`detach_rollout`: true in `train.py`, false in the
model files); every saved run sets the key, so no run was affected.

---

## 2. Computed features and what a checkpoint expects

The inputs `doy_cos`/`doy_sin`, `solar_zenith` and `hod_cos`/`hod_sin` are stored in no data
store and not in the preload cache. `MultimodalDataset` computes them from
`deep_lst/utils/features.py` every time it serves a sample. A change in that module therefore
takes effect without rebuilding the cache, and it changes what an existing checkpoint is fed.

This has happened once. Since commit `e67540a`, `hod_cos`/`hod_sin` on `develop` encode the
local solar hour angle of the grid cell. Before that commit, and on the publication snapshot
`DeepLST_v1.0`, the same names hold the UTC hour of day. Checkpoints trained before the commit,
which includes every run of the paper, expect the UTC encoding: run them from `DeepLST_v1.0`,
not from `develop`.

---

## 3. Gotchas

- `FileNotFoundError: ... .tmp` at start-up: the preload-cache race (`docs/RUNBOOK.md`, section 4.6). Run the first job
  on a new selection or evaluation set alone.
- `FileNotFoundError` / missing group for `<modality>_stats_<sel>.zarr`: statistics not yet
  computed for that selection (`docs/RUNBOOK.md`, section 4.4).
- Job dies immediately with no log: `scripts/out/` does not exist.
- Two different runs share one run directory as `version_0` and `version_1`: `_spatial`,
  `window.lookback` or `seed` were not put into `tag`.
- The transformer at long lookback does not fit on one A40 at batch 512: 4 GPUs x 128.
- Inference or baseline jobs fail on memory settings: add `unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU`.
- `ModuleNotFoundError` from eodata's `lazy_load` after work in `../eodata_slim`: eodata is an
  editable install, so a branch checkout there changes what deep_lst imports. Use a separate
  worktree and environment for eodata branch work.
