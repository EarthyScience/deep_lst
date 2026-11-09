# Architecture

How a run is configured and named, what the computed input features mean for existing
checkpoints, and the pitfalls that follow from the design. The role of every package is in
`README.md`.

---

## 1. Configuration and run naming (Hydra)

`conf/train.yaml` is the composition root. Its `defaults` pick one file from each group:

| Group | Options | Default | Sets |
|---|---|---|---|
| `model` | the files in `conf/model/` | `mlp_rollout` | backbone and its hyper-parameters, loss terms, `_model_name` |
| `loss` | `conf/loss/` | `smoothl1` | `model.loss_fn`, `_loss` |
| `horizon` | `conf/horizon/` | `h24` | forecast horizon in hours for every modality, `_horizon` |
| `data` | `conf/data/` | `full` | which variables the model sees as inputs, prognostics and diagnostics, and which modalities are active; `_data` |

`python deep_lst/train.py --help` lists the options of every group.

Scalar keys you will override often (all in `conf/train.yaml`):

| Key | Meaning |
|---|---|
| `_spatial` | name of the training pixel selection, resolves to `assets/spatial/indices_<_spatial>.npz`. `stats_spatial` follows it, so the statistics for that name must exist. |
| `_spatial_val` | name of the pixel selection that training validates and tests on; the default is a spatial holdout disjoint from the default `_spatial`. Normalisation stays on the statistics of `_spatial`. |
| `_temporal_train`, `_temporal_val` | names of the temporal split files under `assets/temporal/` |
| `_lookback` | hours of input history |
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
python deep_lst/train.py model=transformer_rollout loss=smoothl1 horizon=h24 data=full \
    _spatial=<selection> _lookback=72 training.batch_size=128 training.max_epochs=40 tag=<tag>
```

The **run name** is `_experiment` from `conf/train.yaml`, extended by `tag` and `_smoke` in
`deep_lst/train.py`:

```
<_version>_<_model_name>_<_loss>_<_horizon>_<_data>[_<tag>][_smoke]
```

(`_version` is the data/selection generation set in `conf/train.yaml`; runs before v18 used
`<_model_name>_<_loss>_<_horizon>_<_data>_<_version>` and keep their directory names). The
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
  `_lookback` or `seed` were not put into `tag`.
- The transformer at long lookback does not fit on one A40 at batch 512: 4 GPUs x 128.
- Inference or baseline jobs fail on memory settings: add `unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU`.
- `ModuleNotFoundError` from eodata's `lazy_load` after work in `../eodata_slim`: eodata is an
  editable install, so a branch checkout there changes what deep_lst imports. Use a separate
  worktree and environment for eodata branch work.
