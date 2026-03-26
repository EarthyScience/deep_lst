# Deep-LST: Multimodal land surface temperature forecasting

Data-driven forecasting of land surface temperature (LST), combining satellite observations and reanalysis data. LST is estimated diagnostically and prognostically from medium-range, large-scale synoptic weather conditions and small- to medium scale surface properties.
To use and work in this repository, clone and install [wedata][https://github.com/EarthyScience/wedata] into the same directory.

Currently supports tools for manually creating samples of geospatial scenes, training of MLP (diagnostically, prognostically), a tiny ResNet (diagnostically), and running and analysing forecasts from the models. Uses WandB for logging.

## Config structure

Configs are organised by model family under `configs/`. Each family has a base config and optional override files for variants:
```
configs/
  mlp_rollout/
    config.yml           # base config for the rollout model
    overrides/
      madagaskar.yml     # region-specific overrides
      blocked.yml        # blocked-time-split variant
  mlp_diag/
    config.yml
    overrides/
      ...
```

Override files only contain the keys that differ from the base config. They are merged on top of the base at launch time using OmegaConf. The `experiment:` field in an override identifies the variant (e.g. `experiment: madagaskar`) and is used for WandB grouping.

## Sample scenes

Create a *scenes.yml* file in `configs` and store the created sampling indices under `assets/{modality}/spatial/` per split.
```
uv run scripts/tools/sample_scenes.py --src_grid seviri --split train
```

## Training

Run a base config:
```
uv run deep_lst/mlp/train.py --config mlp_rollout
```

Run with one or more overrides applied on top:
```
uv run deep_lst/mlp/train.py --config mlp_rollout --override madagaskar
uv run deep_lst/mlp/train.py --config mlp_rollout --override madagaskar blocked
```

The experiment folder name is derived automatically from the CLI arguments — no `name:` field is needed in the config files:

| Command | Experiment folder |
|---|---|
| `--config mlp_rollout` | `experiments/mlp_rollout/version_N/` |
| `--config mlp_rollout --override madagaskar` | `experiments/mlp_rollout_madagaskar/version_N/` |
| `--config mlp_rollout --override madagaskar blocked` | `experiments/mlp_rollout_madagaskar_blocked/version_N/` |

Append `--tag <label>` to add a short suffix for within-family sweeps (e.g. `--tag lr5e5`).

Each run saves a `config.yml` (the fully merged config) and a `metadata.json` (git hash, overrides, SLURM job ID, timestamp) into the version folder.

### Smoke test

Use `--smoke` to run a fast sanity check (1 epoch, 3 batches, CSV logger only, no WandB):
```
uv run deep_lst/mlp/train.py --config mlp_rollout --smoke
uv run deep_lst/mlp/train.py --config mlp_rollout --override madagaskar --smoke
```

A SLURM smoke test script covering all configs is available at `scripts/training/smoke_test.sh`.

## Pipeline walkthrough

What happens when you call `python deep_lst/mlp/train.py --config mlp_rollout --override madagaskar`:

1. `train.py` derives the run name from CLI args (`mlp_rollout_madagaskar`), loads and merges the base config with the override using OmegaConf, then applies `--tag`/`--smoke` suffixes if present.
2. CSVLogger and WandBLogger are instantiated with the derived name. The merged config and metadata are snapshotted into the version folder.
3. `LitDataModule(**config)` is instantiated and `setup()` is called, which builds `MultimodalDataset` three times (train/val/test) by unpacking the respective sub-config.
4. `MultimodalDataset.from_modalities()` calls `_load_samples()` per modality. Depending on the config, samples are filtered either via precomputed temporal indices (`temporal_split:`) or a date range (`start_time:`/`end_time:`).
5. `get_keys_from_config()` parses the dataset config to build input/output key tuples for the model.
6. `MultiModalMLP` is instantiated with those keys, wrapped in `LitModelModule`, and the loss function is dynamically imported via `importlib`.

## Analysis

Run a forecast for a completed training run with:
```
uv run scripts/analysis/run_forecast.py --config experiment_name --version version_x
```
