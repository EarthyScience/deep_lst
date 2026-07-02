# deep_lst

Data-driven forecasting of land surface temperature (LST) from satellite observations and
reanalysis. LST is estimated diagnostically and prognostically from medium-range, large-scale
synoptic weather conditions (ERA5) and small- to medium-scale surface state (SEVIRI).

Built on PyTorch Lightning, configured with Hydra, logged with Weights & Biases.

> **Status:** v0.2 — research code, single-maintainer. Interfaces and configs may change.

## Installation

deep_lst depends on [`eodata_slim`](https://github.com/EarthyScience/eodata_slim), the data-provision
framework (extraction, zarr creation, and the data reader). It is installed as an **editable local
dependency from a sibling directory**, so clone both repos side by side:

```
git clone git@github.com:MWesselkamp/deep_lst.git
git clone git@github.com:EarthyScience/eodata_slim.git
cd deep_lst
uv sync
```

This resolves `eodata` from `../eodata_slim` (see `[tool.uv.sources]` in `pyproject.toml`). Python 3.12
is pinned via `.python-version`. Training expects a CUDA GPU; the cluster entry points assume SLURM.

deep_lst consumes the **sampled zarr stores** that `eodata_slim` produces. The default location is
`/Net/Groups/BGI/tscratch/mwessel/wegen-stores/sampled_stores/` (see `ExperimentPaths.stores_root` in
`deep_lst/utils/io.py`).

## Concepts

- **Multimodal input.** Two modalities are composed per sample: `era5` (coarse synoptic drivers) and
  `seviri` (high-resolution surface state). See `modality_ids` in `conf/train.yaml`.
- **Diagnostic vs. prognostic.** `mlp_diag` estimates LST from concurrent conditions; `mlp_rollout`
  forecasts forward autoregressively over a lead-time horizon.
- **Rollout forecasting.** The prognostic model rolls its own prediction back in as state over the
  forecast horizon (`horizon/h4 … h72`, in hours).
- **Evaluation against baselines.** Forecasts are scored against climatology and persistence using
  `climpred` / `xskillscore` (skill, RMSE, ACC).

## Structure

```
deep_lst/
  dataset/       MultimodalDataset, LitDataModule, collator, climatology
  mlp/           MLP models + train / finetune entry points
  vit/  nas/     ViT and neural-architecture-search variants (experimental)
  baselines/     climatology / persistence / external baselines
  losses/        MSE, SmoothL1, increment, CRPS
  evaluation/    inference (Forecast), metrics, tables, plots
  utils/         config parsing, path resolution (ExperimentPaths), io
conf/            Hydra config groups (model, loss, horizon, size, stratification, eval)
scripts/         SLURM launchers for statistics, training, inference, evaluation
tests/           pytest suite
experiments/     per-run outputs (checkpoints, merged config, metadata)
```

### Configuration (Hydra)

`conf/train.yaml` composes config groups; override any group or key on the command line:

```
python deep_lst/mlp/train.py model=mlp_rollout loss=smoothl1 horizon=h24 size=L stratification=mtg
```

The experiment name is derived from the composed config (`_experiment`), so no `name:` field is needed.
Each run writes the fully merged config, a `metadata.json` (git hash, SLURM job ID, timestamp), and
checkpoints into `experiments/<experiment>/version_N/`.

## Reproducing results

The full pipeline runs data → statistics → training → inference → evaluation. On the cluster each stage
has a SLURM launcher under `scripts/`; the underlying commands are shown below.

1. **Data.** Produce the sampled zarr stores with `eodata_slim` (see its README).
2. **Statistics.** Compute normalization statistics (submits one SLURM job per modality × spatial
   database; edit the `SPATIALS` list inside the script first):
   ```
   bash scripts/statistics/launch_statistics.sh
   ```
3. **Train.**
   ```
   sbatch scripts/training/train_mlp.sh model=mlp_rollout loss=smoothl1 horizon=h24
   ```
   Add `smoke=true` for a fast 1-epoch sanity run (CSV logger only, no WandB).
4. **Infer.** Roll out a forecast for a trained run over an eval region:
   ```
   uv run scripts/inference/run_inference.py \
       experiment=<experiment> version=version_0 eval=gobabeb inference_horizon=72
   ```
5. **Evaluate.** Score forecasts against baselines and build tables/plots (argparse CLI):
   ```
   uv run scripts/evaluation/run_evaluation.py \
       --experiment <experiment> --version version_0 --eval gobabeb \
       --baselines era5 clim_smooth persistence
   ```

Run the test suite with `uv run pytest`.
