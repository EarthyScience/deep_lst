import importlib

import torch

from deep_lst.lit_module import LitModelModule
from deep_lst.utils.io import ExperimentPaths

MODEL_CLASSES = {
    "mlp": "MultiModalMLP",
    "transformer": "Transformer",
}


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


def build_model(config: dict, dataset_config: dict, lookback: int, paths=None):
    """Construct the configured backbone with its static-embedding spec attached.

    One place builds models for training and inference so both agree on input_size and on
    the vocabulary the embeddings were trained with.
    """
    from deep_lst.dataset.static_vocab import build_static_spec

    packing = get_packing_keys(dataset_config)
    static_spec = None
    if config["model"].get("static_encoding", "embedding") == "embedding":
        static_spec = build_static_spec(dataset_config, paths or ExperimentPaths(), packing["input_keys"])

    model_cls = get_model_class(config["model"]["name"])
    return model_cls(
        n_prognostic=packing["n_prognostic"],
        n_diagnostic=packing["n_diagnostic"],
        n_input=len(packing["input_keys"]),
        n_forcing_context=len(packing["forcing_context_keys"]),
        lookback=lookback,
        config=config["model"],
        static_spec=static_spec,
    )


def get_model_class(model_id: str):
    """Resolve the model class for a config `model.name`, mirroring LitModelModule._get_loss.

    Imported lazily via importlib because the model modules import this module.
    """
    module = importlib.import_module(f"deep_lst.models.{model_id}.model")
    return getattr(module, MODEL_CLASSES[model_id])


def load_model(
    config: dict,
    td_keys: dict,
    experiment: str,
    version: str,
    prefer_best: bool = False,
):
    lookback = config.get("train_dataset", {}).get("seviri", {}).get("lookback", 0)
    paths = ExperimentPaths()
    model = build_model(config, config["train_dataset"], lookback, paths)

    ckpt_path = (
        paths.best_checkpoint(experiment, version) if prefer_best else paths.latest_checkpoint(experiment, version)
    )

    lit_module = LitModelModule.load_from_checkpoint(ckpt_path, model=model, config=config, **td_keys)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_module.eval()
    lit_module.to(device)

    return lit_module, device
