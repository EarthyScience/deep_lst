# deep_lst — orientation for Claude

Forecasting land surface temperature over Africa from SEVIRI + ERA5/ERA5-Land, PyTorch
Lightning + Hydra + W&B. **This file describes the `develop` branch.**

`README.md` on `develop` is the long-form guide (install, assets, full pipeline, SLURM
resources, newcomer workflow). Read it before asking about anything it already covers. Where
README and code disagree, the code wins — the README still says the data label is `v14`, the
configs say `v16`.

## Branches

| Branch | What it is |
|---|---|
| `develop` | the working line; everything new goes here |
| `DeepLST_v1.0` | publication snapshot (label `v14`): old `evaluation/` pruned, reviewer README, `conf/data/*.yaml` committed |

Two differences that cause real breakage, not just noise:

- **`conf/data/*.yaml` is tracked only on `DeepLST_v1.0`.** On `develop` those files are
  untracked working-tree files, so `data=full` works here but would fail in a fresh clone of
  `develop`. Check `git ls-files conf/data` before assuming a config group exists.
- **`hod_cos`/`hod_sin` changed meaning on `develop`** (commit `e67540a`): they are now the
  **local solar hour angle** computed per grid in `solar_features`, not the UTC hour-of-day
  that `time_features` returned before. The names are identical. Checkpoints trained before
  that commit (all `v14`/paper runs) expect the UTC encoding, so do **not** run paper
  inference from `develop` — use `DeepLST_v1.0` (or `features.py` at `54c65da`). Merge base of
  the two branches is `7703d97`.

## Environment

- Run everything from the repo root (Hydra pins its run dir there; all asset paths are relative).
- Use `.venv/bin/python` or `uv run`, never a bare `python`. Python 3.12.
- `eodata` is an **editable install of the sibling `../eodata_slim`**. Checking out a different
  branch in that tree silently changes what deep_lst imports (entry-point registry breaks with
  `ModuleNotFoundError` from `lazy_load`). Use a separate worktree + venv for eodata branch work
  instead of switching in place.
- Network needs `ml proxy` (git, uv, wandb).
- `ExperimentPaths.preload_cache_root` is hard-coded to `/Net/Groups/BGI/tscratch/mwessel/tmp`;
  everything else in `deep_lst/utils/io.py` is relative to the CWD.

## Layout

```
deep_lst/
  train.py / finetune.py      Hydra entry points -> experiments/<run>/version_N/
  lit_module.py               LightningModule wrapper
  dataset/                    preload into RAM, MultimodalDataset, LitDataModule, climatology
  models/                     mlp/, transformer/, static_embeddings.py
  losses/                     rollout.py (in use), crps.py, diagnostic.py, common.py
  evaluation/                 inference.py (Forecast), spatial.py, metrics, tables, plots
  evaluation_climpred/        the live verification pipeline (see below)
  baselines/                  climatology / persistence / ERA5(-Land) reference forecasts
  exploration/                one-off analyses, not part of the pipeline
  utils/                      io.py (ExperimentPaths), features.py, spatial.py, normalisation.py
conf/                         train.yaml, infer.yaml, statistics.yaml, finetune.yaml + groups
scripts/                      *.py entry points are tracked; all *.sh launchers are gitignored
```

`evaluation/` and `evaluation_climpred/` both exist. **`evaluation_climpred/` is the current
scoring pipeline.** From `evaluation/`, only `inference.py` (`Forecast`) and `spatial.py`
(`pixel_grid`) have live consumers; the rest is legacy kept for reference.

## Config and run naming

`conf/train.yaml` is the composition root; groups are `model` (`mlp_rollout`,
`transformer_rollout`, `mlp_diag`), `loss` (`smoothl1`, `mse`, `mae`), `horizon`
(`h4…h72`), `data` (`full`, `noera5`, `noera5land`, `nosevaux`).

Defaults on `develop`: `mlp_rollout` + `smoothl1` + `h24` + `full`, `_spatial: mtg_new`,
`_lookback: 24`, `modality_ids: [era5, era5_land, seviri]` (glofas plumbed but off).
Split: train ≤2022 (`train_full_2023b10`), validate days 11–20 of each month of 2023
(`val_2023b10`), test `test_period_00`.

Run name (`train.py:73`) is `<_model_name>_<_loss>_<_horizon>_<_data>_v16[_<tag>][_smoke]`.
`_spatial` and `_lookback` are **not** in the name — put them in `tag`, or two different runs
collide in one `experiments/<run>/` as `version_0`, `version_1`.

Ground truth of what actually ran is `experiments/<run>/version_N/config.yml`, not `conf/`.
An "epoch" is `training.limit_train_batches` (2500) batches, not a pass over the data.
Seed is hard-coded to 42 in `train.py`.

## Data path and features

- Training never reads zarr. `MultimodalDataset.preload_shared` loads all selected pixels into
  RAM, normalised, and caches to `preload_<spatial>_<fingerprint>.pt` (20–30 GB for 3000
  sites; jobs ask for 256 GB). The fingerprint covers anchor ids, store names, `load` lists and
  the statistics mtimes — so recomputing statistics rebuilds the cache. Two jobs racing on a
  missing cache both write the same `.tmp` and one dies; warm it with a single `smoke=true` run.
- Keyed on the training selection: `indices_<sel>.npz`, the three `*_stats_<sel>.zarr`, the
  cache. Keyed on the evaluation set: eval indices and baselines.
- `_<modality>_load` in `train.yaml` says what reaches RAM; `conf/data/<data>.yaml` says what
  the model sees (`inputs` / `prognostics` / `context_vars`). Loaded-but-unused is normal
  (e.g. SEVIRI `MLE`, `MH`, `quality_flag`; ERA5-Land `skt` is baseline-only).
- **Computed features are built per sample in `__getitem__`** (`_add_computed`), *not* stored in
  the preload cache: `doy_cos/doy_sin` (global), and per grid `solar_zenith`, `hod_cos/hod_sin`.
  So a features.py change takes effect without rebuilding the cache — and silently changes what
  an old checkpoint is fed.
- Spatial selections carry anchor-grid ids in the single key `src_indices`; `anchor: mtg` means
  global MajorTom 5 km ids, mapped to each store's rows by nearest centre in
  `utils/spatial.py::map_to_store_rows`.

## Commands

```
.venv/bin/python deep_lst/train.py model=mlp_rollout _spatial=<sel> tag=<tag>
.venv/bin/python scripts/assets/compute_statistics.py modality=seviri spatial=<sel>
.venv/bin/python scripts/inference/run_inference.py eval=<eval> experiment=<run> version=version_0
.venv/bin/python -m deep_lst.evaluation_climpred.verif_init  --experiment <run> --eval_name <eval> --windows all day night
.venv/bin/python -m deep_lst.evaluation_climpred.evaluate    --experiment <run> --eval_name <eval>
.venv/bin/pytest
```

Verification is three `verif_*` jobs (init / pixel / anom) and then `evaluate`; it is
memory-bound (`big` partition, hundreds of GB for dense regions). The SLURM launchers that
chain this — including `scripts/evaluation/run_climpred.sh` — are gitignored, so they exist only
in the working tree; read the actual script before assuming its flags.

## Working here

- `assets/`, `experiments/`, `results/`, `plots/`, `tables/`, `notebooks/`, `scripts/**/*.sh`
  and `scripts/out/` are all gitignored. Don't propose committing them.
- Temporary scripts and intermediate outputs go to tscratch, not into the repo.
- Legacy code under `evaluation/` and `exploration/` is deliberately kept — mention dead code,
  don't delete it.
