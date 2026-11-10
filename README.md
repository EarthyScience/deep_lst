# deep_lst

[![Tests](https://github.com/MWesselkamp/deep_lst/actions/workflows/tests.yml/badge.svg)](https://github.com/MWesselkamp/deep_lst/actions/workflows/tests.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)

Multi-modal and light-weight land surface temperature forecasting

> **Status:** this is `develop`, the working branch; interfaces and configs change. The
> last publication snapshot is the tag `DeepLST_v1.0.0`; reproduce the paper from there. The latest tag DeepLST_v2.0.0 has moved forward from there.

Multi-modal machine learning for forecasting of land surface temperature (LST) over Africa from geostationary satellite observations (SEVIRI) and reanalysis (ERA5, ERA5-Land). An autoregressive MLP or transformer
rolls the hourly LST incrementes forward over a customized forecast horizon. The state is driven by the large-scale atmospheric forcing. Forecasts are verified against climatology, persistence and ERA5-Land skin temperature. The code builds on PyTorch Lightning, is configured with Hydra and logs to Weights & Biases.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Layout](#layout)
- [Documentation](#documentation)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Install

You need [uv](https://docs.astral.sh/uv/) and a conda or mamba installation. The latter
provides the native GDAL and PROJ libraries of the data library
[eodata](https://github.com/EarthyScience/eodata), which deep_lst installs in editable mode
from a sibling directory that must be named `eodata_slim`:

```sh
git clone https://github.com/MWesselkamp/deep_lst.git
git clone -b v0.3_DeepLST https://github.com/EarthyScience/eodata.git eodata_slim

cd eodata_slim
mamba env create -n eodata -f environment.yaml
mamba activate eodata    # keep it active: the GDAL bindings of deep_lst link against it
uv sync

cd ../deep_lst
uv sync                  # Python 3.12 environment with deep_lst and eodata
uv run pytest            # tests that need the data stores are skipped
```

## Usage

Training and evaluation read hourly `.eozarr` data stores and a set of index, statistics and
baseline assets. Neither is part of this repository; [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
describes them and where they come from. Training needs a CUDA GPU and about 256 GB of memory
for a selection of 3000 sites. The pipeline has five stages, run from the repository root:

```sh
# 1. normalisation statistics of a training selection (repeat for era5 and era5_land)
uv run python scripts/assets/compute_statistics.py modality=seviri spatial=<selection>
# 2. training
uv run python deep_lst/train.py model=mlp objective=rollout _spatial=<selection> tag=<tag>
# 3. inference: roll the trained run out over an evaluation set
uv run python scripts/inference/run_inference.py eval=<eval> experiment=<run>
# 4. verification against the observations and the reference forecasts
uv run python -m deep_lst.evaluation_climpred.verif_init --experiment <run> --eval_name <eval>
# 5. scores, tables and figures from the saved verification files
uv run python -m deep_lst.evaluation_climpred.evaluate --experiment <run> --eval_name <eval>
```

`<selection>` names a set of training pixels, `<eval>` an evaluation set configured in
`conf/eval/`, and `<run>` the trained run. Stage 4 has two more passes with the same
arguments, `verif_pixel` and `verif_anom`; all three compare the model with reference
forecasts that `scripts/assets/run_baselines.py` extracts once per evaluation set.

## Layout

| Path | Role |
|---|---|
| `deep_lst/` | the package; its root holds the entry points for training and fine-tuning, the Lightning module and the forecast rollout |
| `deep_lst/dataset/` | loads the selected pixels of every store into memory once, then serves normalised lookback and horizon windows as batches |
| `deep_lst/models/` | the MLP and transformer backbones, and the embedding of categorical static inputs they share |
| `deep_lst/losses/` | training objectives: the autoregressive rollout loss and the losses for diagnostic targets |
| `deep_lst/baselines/` | reference forecasts at the evaluation pixels: persistence, climatology, ERA5 and ERA5-Land |
| `deep_lst/evaluation_climpred/` | verification of the forecasts against observations and baselines; derives scores, tables and figures |
| `deep_lst/sensitivity/` | input sensitivity of a trained model: forecasts with one input perturbed at a time |
| `deep_lst/exploration/` | stand-alone analyses of the data and the site sampling; not part of the pipeline |
| `deep_lst/nas/` | hyper-parameter search |
| `deep_lst/utils/` | shared helpers: experiment paths, config handling, computed solar and time features, normalisation, grid mapping |
| `conf/` | Hydra configuration: one root file per entry point and the groups model, objective, loss, optim, horizon, data, sources and eval |
| `scripts/` | entry points outside the package (assets, inference, sensitivity) and SLURM launchers written for the MPI-BGC cluster |
| `tests/` | pytest suite; runs in CI without the data stores |

## Documentation

- [`docs/RUNBOOK.md`](docs/RUNBOOK.md): operating the pipeline on the MPI-BGC cluster, from
  prerequisites and directories to data, assets and the SLURM job of every stage.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): configuration groups, run naming, computed
  input features and known pitfalls.
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md): every configuration file, group, option
  and key.
- [`docs/ONBOARDING.md`](docs/ONBOARDING.md): the workflow for a new training selection, and a
  first project.

Where a document and the code disagree, the code wins.

## Citation

If you use this software, please cite it with the metadata in [`CITATION.cff`](CITATION.cff)
(on GitHub: "Cite this repository").

## Contributing

Questions and bug reports go to the
[issue tracker](https://github.com/MWesselkamp/deep_lst/issues). Pull requests against
`develop` are welcome; run `uv run pre-commit run --all-files` and `uv run pytest` first.

## License

BSD-3-Clause, see [`LICENSE`](LICENSE). © 2026 Marieke Wesselkamp
