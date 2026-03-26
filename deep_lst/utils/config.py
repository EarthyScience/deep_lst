from pathlib import Path

from omegaconf import OmegaConf


def open_config(path: Path) -> dict:
    """Load a YAML config and return a plain dict with all interpolations resolved."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def merge_configs(base_path: Path, override_paths: list[Path]) -> dict:
    """Load base config, apply overrides in order, return plain dict.
    Later overrides win on conflicting keys. All interpolations resolved."""
    merged = OmegaConf.load(base_path)
    for p in override_paths:
        merged = OmegaConf.merge(merged, OmegaConf.load(p))
    return OmegaConf.to_container(merged, resolve=True)


def build_feature_list(config: dict) -> list:
    """
    Returns a flat list of input feature names from cfg, in order:
    era5(dynamic->static) then seviri(dynamic->static), but only
    for types listed in each group's `use_input`. Safely handles None/missing.
    """
    feature_list = []
    for group in ("era5", "seviri"):
        g = config.get(group) or {}
        channels_input = (g.get("channels") or {}).get("input") or {}
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


def get_keys_from_config(dataset_config):
    in_keys = []
    prognostic_keys = []
    prognostic_out_keys = []
    diagnostic_keys = []
    diagnostic_out_keys = []
    for modality, mod_cfg in dataset_config.items():
        if not isinstance(mod_cfg, dict):
            continue
        for var in mod_cfg.get("inputs", []):
            in_keys.append((modality, var))
        for var in mod_cfg.get("prognostics", []):
            prognostic_keys.append((modality, var))
            prognostic_out_keys.append((modality, f"{var}_pred"))
        for var in mod_cfg.get("diagnostics", []):
            diagnostic_keys.append((modality, var))
            diagnostic_out_keys.append((modality, f"{var}_pred"))
    td_keys = {
        "in_keys": in_keys,
        "out_keys": prognostic_out_keys + diagnostic_out_keys,
        "prognostic_keys": prognostic_keys,
        "prognostic_out_keys": prognostic_out_keys,
        "diagnostic_keys": diagnostic_keys,
        "diagnostic_out_keys": diagnostic_out_keys,
    }
    return td_keys
