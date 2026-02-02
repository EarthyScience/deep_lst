# Deep-LST: Multimodal land surface temperature forecasting

Data-driven forecasting of land surface temperature (LST), combining satellite observations and reanalysis data. LST is estimated diagnostically and prognostically from medium-range, large-scale synoptic weather conditions and small- to medium scale surface properties.
To use and work in this repository, clone and install [wedata][https://github.com/EarthyScience/wedata] into the same directory.

Currently supports tools for manually creating samples of geospatial scenes, training of MLP (diagnostically, prognostically), a tiny ResNet (diagnostically), and running and analysing forecasts from the models. Uses WandB for logging. All configurations are stored in and loaded from experiment-specific config-files.

## Sample scenes

Create a *scenes.yml* file in `configs` and stores the created sampling indices under `assets/samples/` per split.
```
uv run scripts/tools/sample_scenes.py --src_grid  seviri --split train
```

## Training

Create an experiment folder in `configs` and place a config file there. For testing, use
```
uv run deep_lst/mlp/train.py --config experiment_name
```
This will create an folder in `experiments` where the training versions are saved to.

## Analysis

Run forecast for a training run with
```
uv run scripts/analysis/run_forecast.py --config experiment_name --version version_x
```
