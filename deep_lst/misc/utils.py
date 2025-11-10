import torch
import yaml
import os
from pathlib import Path

def open_config(experiment: str, path: str = "deep_lst/experiments") -> dict:
    path = os.path.expanduser(path)
    with open(f"{path}/{experiment}/config.yaml", 'r') as stream:
        return yaml.safe_load(stream)

def open_hparams(experiment: str, version: str, path: str = "deep_lst/experiments") -> dict:
    path = os.path.expanduser(path)
    with open(f"{path}/{experiment}/logs/prototype_{experiment}/{version}/hparams.yaml", 'r') as stream:
        return yaml.safe_load(stream)
    
def load_checkpoint(experiment, version, name = "prototype", path = ".experiments/"):
    checkpoints_root = Path(path+experiment + f"/logs/{name}_{experiment}/" + version + "/checkpoints/")
    print(f"Looking for checkpoints in {checkpoints_root}")
    checkpoint_path = sorted(checkpoints_root.glob("*.ckpt"))[-1]
    return checkpoint_path


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