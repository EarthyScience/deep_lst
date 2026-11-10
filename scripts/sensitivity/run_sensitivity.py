"""Sensitivity of a trained run to its stored inputs, one input at a time (deep_lst/sensitivity).

Data: the run's saved config.yml and best checkpoint; the eval dataset built from the same
preload cache the inference uses (every stored input of the eval sites, normalised), with the
windows, collator and rollout of scripts/inference/run_inference.py (forecast.PerturbedInference).
Perturbation: the chosen input's days are shuffled per cell in the preloaded columns
(perturb.permute_days), within calendar months (block=month: day-to-day anomalies only) or across
the whole record (block=all: seasonal cycle too); every other input stays as it is.
Output: one forecast store per input,
results/<experiment>/<version>/<eval>/sensitivity/permute_days_<block>/<modality>.<variable>.zarr,
then scores.collect compares every store with the run's own forecast.zarr on the samples where
both forecasts and the observation are valid: scores.nc (per pixel) and summary.csv (pooled per
solar window and forecast day). Settings: conf/sensitivity.yaml; the eval dataset follows
conf/infer.yaml (e.g. splits.test.init_hours=[0] thins the inits to 00 UTC).

    python scripts/sensitivity/run_sensitivity.py experiment=<exp> eval=eval_mtg sensitivity.block=month
    python scripts/sensitivity/run_sensitivity.py experiment=<exp> eval=eval_mtg \\
        sensitivity.variables='[era5_land.swvl1,era5_land.swvl2]'   # a subset; default = every stored temporal input

Launcher: scripts/sensitivity/run_sensitivity.sh (CPU partition). Figures: plot_sensitivity.py.
"""

import hydra
from omegaconf import DictConfig

from deep_lst.sensitivity.forecast import PerturbedInference, sensitivity_dir, stored_inputs
from deep_lst.sensitivity.scores import collect
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.inference import load_run_config


@hydra.main(config_path="../../conf", config_name="sensitivity", version_base="1.3")
def main(cfg: DictConfig) -> None:
    paths = ExperimentPaths()
    config = load_run_config(cfg, paths)
    settings = cfg.sensitivity
    out_dir = sensitivity_dir(paths, cfg.experiment, cfg.version, cfg.eval_name, settings.method, settings.block)

    inference = PerturbedInference(cfg, config)
    if settings.variables is None:
        variables = stored_inputs(config, inference.shared["var_names"])
    else:
        variables = [tuple(name.split(".", 1)) for name in settings.variables]
    print(f"Experiment : {cfg.experiment} / {cfg.version} on {cfg.eval_name}")
    print(f"Variables  : {', '.join(f'{m}.{v}' for m, v in variables)}")
    inference.run(out_dir, variables, settings.method, settings.block, settings.seed)
    if settings.score:
        collect(paths, cfg.experiment, cfg.version, cfg.eval_name, out_dir)


if __name__ == "__main__":
    main()
