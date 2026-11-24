import copy
import importlib

import torch

from deep_lst.lit_module import LitModelModule
from deep_lst.utils.io import ExperimentPaths, default_sources

MODEL_CLASSES = {
    "mlp": "MultiModalMLP",
    "transformer": "Transformer",
}

# Layout version of conf/train.yaml and of every config.yml written from it.
# 2: objective / optim / window / sources / features / splits. 3: sources name product stores.
CONFIG_VERSION = 3

SPLIT_NAMES = ("train", "val", "test")
# per-modality keys of the `features` section (conf/data/<data>.yaml)
ROLE_KEYS = ("inputs", "prognostics", "diagnostics", "context_vars")
# config groups whose chosen option is a token of the run name, in name order
RUN_NAME_GROUPS = ("model", "objective", "loss", "horizon", "data")
# features computed when a sample is served, not read from a store (deep_lst/utils/features.py)
COMPUTED_FEATURES = ("solar_zenith", "doy_cos", "doy_sin", "hod_cos", "hod_sin")

# --- config.yml files written before config_version 2 (lift_to_version_2) ---
# model.loss_fn -> the pointwise node of conf/loss/<option>.yaml
LEGACY_POINTWISE = {
    "smooth_l1": {"_target_": "torch.nn.functional.smooth_l1_loss", "_partial_": True, "beta": 1.0},
    "mse": {"_target_": "torch.nn.functional.mse_loss", "_partial_": True},
    "mae": {"_target_": "torch.nn.functional.l1_loss", "_partial_": True},
}
# model keys that moved to `objective` or `optim`, and the ones nothing ever read
LEGACY_MODEL_KEYS = (
    "step_loss",
    "loss_fn",
    "teacher_forcing_loss",
    "incremental_loss",
    "rollout_loss",
    "detach_rollout",
    "init_fill",
    "init_hide_fraction",
    "checkpoint_monitor",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "scheduler_T0",
    "probabilistic",
)
# top-level keys replaced by window / sources / features / splits / the run name in `name`
LEGACY_TOP_LEVEL_KEYS = (
    "train_dataset",
    "val_dataset",
    "test_dataset",
    "_lookback",
    "_experiment",
    "_model_name",
    "_loss",
    "_horizon",
    "_data",
    "_sigma_prognostics",
    "_sigma_diagnostics",
    "_seviri_store",
    "_era5_store",
    "_era5land_store",
    "_glofas_store",
    "_seviri_load",
    "_era5_load",
    "_era5land_load",
    "_glofas_load",
    "_glofas_transform",
)


def split_views(config: dict) -> dict[str, dict]:
    """The dataset dict of every split, {"train": ..., "val": ..., "test": ...}.

    Each view is what MultimodalDataset, get_keys_from_config and the baselines consume: the
    split's selection keys (spatial, stats_spatial, anchor, temporal, ...) and one block per
    modality with stores / load / transform from `sources`, lookback / horizon from `window`
    and the role lists from `features`. The modalities come in the order of `sources`, which
    fixes the channel order of the packed tensors.
    """
    window = config["window"]
    views = {}
    for split_name in SPLIT_NAMES:
        view = copy.deepcopy(config["splits"][split_name])
        for modality, source in config["sources"].items():
            modality_view = copy.deepcopy(source)
            modality_view.pop("increment_vars", None)  # statistics only (compute_statistics.py)
            modality_view["lookback"] = window["lookback"]
            modality_view["horizon"] = window["horizon"]
            roles = config["features"].get(modality, {})
            modality_view.update(copy.deepcopy(roles))
            view[modality] = modality_view
        views[split_name] = view
    return views


def migrate_config(saved: dict) -> dict:
    """Lift a saved config.yml of any earlier layout to the current one.

    A config written before config_version 2 is first brought to version 2 (lift_to_version_2),
    a version-2 config then to version 3 (lift_to_version_3). A current config passes unchanged.
    """
    config = saved
    if config.get("config_version", 1) == 1:
        config = lift_to_version_2(config)
    if config["config_version"] == 2:
        config = lift_to_version_3(config)
    return config


def lift_to_version_3(config: dict) -> dict:
    """Point a version-2 config at the product stores.

    Version 2 named one .eozarr store per modality, read through eodata. Every modality block of
    `sources` is replaced by its block in conf/sources/product.yaml; `features`, and with it the
    order of the model's input channels, stays as saved. The roots of the .eozarr stores are
    dropped from `paths`.

    Raises:
        ValueError: A modality or a variable of the run is not available from the product stores.
    """
    product_sources = default_sources()
    lifted = copy.deepcopy(config)
    sources = {}
    unavailable = []
    for modality, saved_source in config["sources"].items():
        if modality not in product_sources:
            raise ValueError(f"lift_to_version_3: no product stores for the modality {modality!r}")
        source = copy.deepcopy(product_sources[modality])
        if "quality_mask" in saved_source:
            source["quality_mask"] = copy.deepcopy(saved_source["quality_mask"])
        sources[modality] = source

        loaded = [variable for variables in source["load"].values() for variable in variables]
        roles = config["features"].get(modality, {})
        for role in ROLE_KEYS:
            for variable in roles.get(role, []):
                if variable not in loaded and variable not in COMPUTED_FEATURES:
                    unavailable.append(f"{modality}.{variable}")
    if unavailable:
        raise ValueError(f"lift_to_version_3: the product stores do not provide {sorted(set(unavailable))}")

    lifted["sources"] = sources
    if "paths" in lifted:
        lifted["paths"].pop("eozarr_root", None)
        lifted["paths"]["store_roots"] = {}
    lifted["config_version"] = 3
    return lifted


def lift_to_version_2(saved: dict) -> dict:
    """Lift a config.yml written before config_version 2 to version 2.

    The loss terms, the optimiser keys and the per-split dataset dicts move to `objective`,
    `optim`, `window`, `sources`, `features` and `splits`; split_views() of the result gives
    back the three dataset dicts of the saved config. Keys the old code filled in when a
    config omitted them get the value that code used (teacher_forcing_loss and rollout_loss on,
    init_fill none); keys nothing ever read are dropped.
    """
    legacy_model = saved["model"]
    step_loss = legacy_model.get("step_loss")
    if step_loss != "RolloutLoss":
        raise ValueError(f"lift_to_version_2: only RolloutLoss runs can be migrated, got model.step_loss={step_loss!r}")

    objective = {
        "_target_": "deep_lst.losses.RolloutLoss",
        "kind": "rollout",
        "terms": {
            "teacher_forcing": float(legacy_model.get("teacher_forcing_loss", True)),
            "rollout": float(legacy_model.get("rollout_loss", True)),
            "increment": float(legacy_model["incremental_loss"]),
        },
        "detach_rollout": legacy_model["detach_rollout"],
        "min_valid_fraction": 0.05,
        "init_fill": legacy_model.get("init_fill", "none"),
        "init_hide_fraction": legacy_model.get("init_hide_fraction", 0.5),
        "checkpoint_monitor": legacy_model["checkpoint_monitor"],
        "pointwise": copy.deepcopy(LEGACY_POINTWISE[legacy_model["loss_fn"]]),
    }
    optim = {
        "learning_rate": legacy_model["learning_rate"],
        "weight_decay": legacy_model.get("weight_decay", 1e-4),
        "warmup_steps": legacy_model.get("warmup_steps", 20000),
        "eps": 1e-8,
    }
    model = {}
    for key, value in legacy_model.items():
        if key not in LEGACY_MODEL_KEYS:
            model[key] = copy.deepcopy(value)

    legacy_views = {
        "train": saved["train_dataset"],
        "val": saved["val_dataset"],
        "test": saved["test_dataset"],
    }
    train_view = legacy_views["train"]
    window = {
        "lookback": train_view["seviri"]["lookback"],
        "horizon": train_view["seviri"]["horizon"],
    }
    sources = {}
    features = {}
    for modality, modality_view in train_view.items():
        if not isinstance(modality_view, dict):
            continue
        source = {}
        roles = {}
        for key, value in modality_view.items():
            if key in ROLE_KEYS:
                roles[key] = copy.deepcopy(value)
            elif key not in ("lookback", "horizon"):
                source[key] = copy.deepcopy(value)
        sources[modality] = source
        if roles:
            features[modality] = roles
    splits = {}
    for split_name, legacy_view in legacy_views.items():
        selection = {}
        for key, value in legacy_view.items():
            if not isinstance(value, dict):
                selection[key] = copy.deepcopy(value)
        splits[split_name] = selection

    migrated = {"config_version": 2}
    for key, value in saved.items():
        if key not in LEGACY_TOP_LEVEL_KEYS and key != "model":
            migrated[key] = copy.deepcopy(value)
    migrated["model"] = model
    migrated["objective"] = objective
    migrated["optim"] = optim
    migrated["window"] = window
    migrated["sources"] = sources
    migrated["features"] = features
    migrated["splits"] = splits

    # one window and one set of modality blocks for all splits is all the new layout can say
    if split_views(migrated) != legacy_views:
        raise ValueError(
            "lift_to_version_2: the saved train/val/test dataset dicts differ in more than their "
            "selection keys (or in lookback/horizon between modalities) and cannot be migrated"
        )
    return migrated


def experiment_name(config: dict, choices: dict) -> str:
    """Run name <_version>_<model>_<objective>_<loss>_<horizon>_<data>, without tag and smoke.

    `choices` maps a config group to its chosen option: HydraConfig.get().runtime.choices in
    a Hydra-run entry point, passed by hand under the compose API.
    """
    tokens = [config["_version"]]
    for group in RUN_NAME_GROUPS:
        tokens.append(choices[group])
    return "_".join(tokens)


def validate_config(config: dict) -> None:
    """Fail on an objective that the chosen data option cannot serve, before any data is read."""
    keys = get_keys_from_config(split_views(config)["train"])
    prognostic_keys = keys["prognostic_keys"]
    diagnostic_keys = keys["diagnostic_keys"]
    objective = config["objective"]

    if objective["kind"] == "diagnostic":
        if prognostic_keys or not diagnostic_keys:
            raise ValueError(
                "objective=diagnostic needs a data option without prognostics and with at least one "
                f"diagnostic (data=full_lstdiag); got prognostics={prognostic_keys}, diagnostics={diagnostic_keys}"
            )
    if objective["init_fill"] != "none":
        without_copy = [key for key in prognostic_keys if key not in diagnostic_keys]
        if without_copy:
            raise ValueError(
                f"objective.init_fill={objective['init_fill']} needs a diagnostic copy of every prognostic "
                f"(data=full_joint); no diagnostic copy of {without_copy}"
            )


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
    prognostic_keys = packing["state_keys"][: packing["n_prognostic"]]
    diagnostic_keys = packing["state_keys"][packing["n_prognostic"] :]
    n_diagnostic_history = len([key for key in diagnostic_keys if key not in prognostic_keys])
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
        n_diagnostic_history=n_diagnostic_history,
    )


def get_model_class(model_id: str):
    """Resolve the model class for a config `model.name`.

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
    paths = ExperimentPaths()
    model = build_model(config, split_views(config)["train"], config["window"]["lookback"], paths)

    ckpt_path = (
        paths.best_checkpoint(experiment, version) if prefer_best else paths.latest_checkpoint(experiment, version)
    )

    lit_module = LitModelModule.load_from_checkpoint(ckpt_path, model=model, config=config, **td_keys)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_module.eval()
    lit_module.to(device)

    return lit_module, device
