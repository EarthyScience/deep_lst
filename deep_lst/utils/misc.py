import torch
import yaml
import os
from pathlib import Path

def open_config(experiment: str, path: str = "experiments") -> dict:
    path = os.path.expanduser(path)
    with open(f"{path}/{experiment}/config.yml", 'r') as stream:
        return yaml.safe_load(stream)

def open_hparams(experiment: str, version: str, path: str = "experiments") -> dict:
    path = os.path.expanduser(path)
    with open(f"{path}/{experiment}/logs/prototype_{experiment}/{version}/hparams.yaml", 'r') as stream:
        return yaml.safe_load(stream)
    
def load_checkpoint(experiment, version, name = "prototype", path = ".experiments/"):
    checkpoints_root = Path(path+experiment + f"/logs/{name}_{experiment}/" + version + "/checkpoints/")
    print(f"Looking for checkpoints in {checkpoints_root}")
    checkpoint_path = sorted(checkpoints_root.glob("*.ckpt"))[-1]
    return checkpoint_path

def save_statistics_npz(statistics: dict, path: Path) -> None:
    """
    statistics: {var: {stat_name: scalar}}
    Writes one .npz with:
      - variables: array[str] of variable names (index)
      - each stat_name: array[float] aligned with `variables`
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    variables = np.array(list(statistics.keys()), dtype=object)

    # Collect all stat names (assumes scalars only)
    stat_names = sorted({k for stats in statistics.values() for k in stats.keys()})

    arrays: dict[str, np.ndarray] = {"variables": variables}

    for stat in stat_names:
        vals = []
        for var in variables:
            if stat not in statistics[var]:
                raise KeyError(f"Missing statistic '{stat}' for variable '{var}'.")
            vals.append(np.asarray(statistics[var][stat]).item())  # ensure scalar python value
        arrays[stat] = np.asarray(vals)

    np.savez(path, **arrays)

def load_statistics_from_npz(path: Path) -> dict[str, dict[str, float]]:
    """
    Loads statistics saved with `save_statistics_npz`.
    Returns:
        dict[var_name, dict[stat_name, value]]
    """
    path = Path(path)
    out: dict[str, dict[str, float]] = {}

    with np.load(path, allow_pickle=True) as z:
        variables = list(z["variables"])
        stat_names = [k for k in z.files if k != "variables"]

        for i, var in enumerate(variables):
            out[str(var)] = {}
            for stat in stat_names:
                out[str(var)][stat] = np.asarray(z[stat][i]).item()

    return out

def build_feature_list(config: dict) -> list:
    """
    Returns a flat list of input feature names from cfg, in order:
    era5(dynamic->static) then seviri(dynamic->static), but only
    for types listed in each group's `use_input`. Safely handles None/missing.
    """
    feature_list = []
    for group in ("era5", "seviri"):
        g = (config.get(group) or {})
        channels_input = ((g.get("channels") or {}).get("input") or {})
        use = g.get("use_input") or []
        if isinstance(use, str):
            use = [use]

        # iterate in the exact order specified by use_input
        for kind in use:
            names = channels_input.get(kind)
            if not names:
                continue  # None, [], or missing
            feature_list.extend([n for n in names if n])
            
    return feature_list