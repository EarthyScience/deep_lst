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


def get_context_var_pairs(dataset_config: dict) -> list[tuple[str, str]]:
    """Returns [(modality, var)] for all modality-keyed context vars."""
    return [
        (modality, var)
        for modality, mod_cfg in dataset_config.items()
        if isinstance(mod_cfg, dict)
        for var in mod_cfg.get("context_vars", [])
    ]


def get_packing_keys(dataset_config: dict) -> dict:
    """Ordered (modality, var) key lists for the packed (dense-tensor) data path.

    The collator concatenates each group along the channel axis in exactly this order,
    and every downstream consumer (model output split, loss, inference, denormalisation)
    reads the same order. Assumes one channel per variable.

      state_keys           prognostics THEN diagnostics  -> y_prog channels (prog-first)
      input_keys           all `inputs`                  -> X_dyn channels (step forcings)
      forcing_context_keys context_vars minus prognostics -> X_dyn_ctx channels (window)
    """
    base = get_keys_from_config(dataset_config)
    prognostic_keys = base["prognostic_keys"]
    context_pairs = get_context_var_pairs(dataset_config)
    forcing_context_keys = [pair for pair in context_pairs if pair not in prognostic_keys]
    return {
        "state_keys": prognostic_keys + base["diagnostic_keys"],
        "input_keys": base["in_keys"],
        "forcing_context_keys": forcing_context_keys,
        "n_prognostic": len(prognostic_keys),
        "n_diagnostic": len(base["diagnostic_keys"]),
    }
