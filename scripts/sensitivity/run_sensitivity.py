"""Sensitivity of a trained model to its stored inputs: inference with one input permuted at a
time, then the RMSE change per pixel and forecast day (conf/sensitivity.yaml).

    python scripts/sensitivity/run_sensitivity.py experiment=<exp> eval=eval_mtg \\
        sensitivity.variables='[era5_land.swvl1,era5_land.swvl2]'
    python scripts/sensitivity/run_sensitivity.py experiment=<exp> eval=eval_mtg   # every temporal input
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
