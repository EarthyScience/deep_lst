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
        # Probabilistic outputs: model predicts log-sigma alongside mean.
        # These go to out_keys only — no target lookup in data.
        for var in mod_cfg.get("sigma_prognostics", []):
            prognostic_out_keys.append((modality, f"{var}_log_sigma_pred"))
        for var in mod_cfg.get("sigma_diagnostics", []):
            diagnostic_out_keys.append((modality, f"{var}_log_sigma_pred"))
    td_keys = {
        "in_keys": in_keys,
        "out_keys": prognostic_out_keys + diagnostic_out_keys,
        "prognostic_keys": prognostic_keys,
        "prognostic_out_keys": prognostic_out_keys,
        "diagnostic_keys": diagnostic_keys,
        "diagnostic_out_keys": diagnostic_out_keys,
    }
    return td_keys


def get_context_keys(dataset_config: dict, lookback: int) -> list[tuple[tuple[str, str], int]]:
    """Returns [((modality, f"{var}_ctx"), feature_size)] for all modalities."""
    if lookback <= 0:
        return []
    out = []
    for modality, mod_cfg in dataset_config.items():
        if not isinstance(mod_cfg, dict):
            continue
        for var in mod_cfg.get("context_vars", []):
            out.append(((modality, f"{var}_ctx"), 2 * lookback))
    return out


def get_context_var_pairs(dataset_config: dict) -> list[tuple[str, str]]:
    """Returns [(modality, var)] for all modality-keyed context vars."""
    return [
        (modality, var)
        for modality, mod_cfg in dataset_config.items()
        if isinstance(mod_cfg, dict)
        for var in mod_cfg.get("context_vars", [])
    ]
