"""Vocabularies for the integer-coded categorical static inputs.

Raw class codes (ERA5 tvh/tvl/slt/lsm, SEVIRI LWMASK/LANDCOV) are arbitrary and
non-contiguous, so every categorical variable gets a raw code -> contiguous id mapping
built from the codes present in the training selection, plus one reserved id for codes
never seen there. Dense evaluation regions do contain classes the training pixels miss;
those map to the reserved id instead of crashing or silently aliasing onto a real class.

The mapping is part of the model contract, like the normalisation statistics: it is
persisted next to the other assets and loaded again at train and inference time.

Build:  python -m deep_lst.preprocessing.static_vocab --spatial mtg_new
"""

import argparse
import json

from deep_lst.utils.io import ExperimentPaths

# Embedding width per vocabulary size (vocabulary INCLUDES the reserved unknown id).
# These legends are tiny; the cap keeps the static block from dominating the input vector.
MAX_EMBEDDING_DIM = 8


def embedding_dim(vocab_size: int) -> int:
    if vocab_size <= 8:
        return 3
    if vocab_size <= 32:
        return 6
    return MAX_EMBEDDING_DIM


def categorical_vars(dataset_config: dict) -> dict[str, list[str]]:
    """{modality: [var, ...]} for every variable read from a data_static_categorical leaf."""
    categorical = {}
    for modality, modality_config in dataset_config.items():
        if not isinstance(modality_config, dict):
            continue
        names = modality_config.get("load", {}).get("data_static_categorical")
        if names:
            categorical[modality] = list(names)
    return categorical


def build_vocab(dataset_config: dict, modality_ids: list[str]) -> dict:
    """Collect the distinct raw codes per categorical variable over the training pixels."""
    import numpy as np

    from deep_lst.dataset.dataset import MultimodalDataset

    shared = MultimodalDataset.preload_shared(modality_ids, **dataset_config)
    columns, inverse, var_names = shared["columns"], shared["inverse"], shared["var_names"]

    vocab = {}
    for modality in modality_ids:
        names = var_names[modality].get("data_static_categorical")
        if not names:
            continue
        codes_per_pixel = columns[modality]["data_static_categorical"][inverse[modality]].numpy()
        for channel, var in enumerate(names):
            column = codes_per_pixel[:, channel]
            if not np.allclose(column, np.round(column)):
                raise ValueError(
                    f"{modality}/{var} is not integer-valued; it is a continuous variable and must "
                    "be read from data_static, not embedded as a class code."
                )
            # sorted() so two builds over the same selection are byte-identical
            codes = sorted(int(code) for code in np.unique(column))
            vocab.setdefault(modality, {})[var] = {
                "codes": codes,
                "unknown_id": len(codes),
                "dim": embedding_dim(len(codes) + 1),
            }
    return {"spatial": dataset_config["spatial"], "vocab": vocab}


def vocab_path(paths: ExperimentPaths, spatial: str):
    return paths.assets_root / "static" / f"vocab_{spatial}.json"


def save_vocab(vocab: dict, paths: ExperimentPaths) -> str:
    path = vocab_path(paths, vocab["spatial"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=2, sort_keys=True))
    return str(path)


def load_vocab(paths: ExperimentPaths, spatial: str) -> dict:
    path = vocab_path(paths, spatial)
    if not path.exists():
        raise FileNotFoundError(
            f"No static vocabulary at {path}. Build it once with "
            f"`python -m deep_lst.preprocessing.static_vocab --spatial {spatial}`."
        )
    return json.loads(path.read_text())


def build_static_spec(dataset_config: dict, paths: ExperimentPaths, input_keys: list) -> list[dict]:
    """Per-variable embedding specs aligned to the packed X_dyn channel order.

    input_keys is the packing order from get_packing_keys; the channel index a spec carries
    is the position of that (modality, var) in the packed input tensor.
    """
    stored = load_vocab(paths, dataset_config["stats_spatial"])["vocab"]
    categorical = categorical_vars(dataset_config)

    specs = []
    for channel, (modality, var) in enumerate(input_keys):
        if var not in categorical.get(modality, []):
            continue
        entry = stored[modality][var]
        specs.append(
            {
                "name": f"{modality}/{var}",
                "channel": channel,
                "codes": entry["codes"],
                "dim": entry["dim"],
            }
        )
    return specs


def main():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", default=None, help="Override the config's spatial selection.")
    parser.add_argument("--overrides", nargs="*", default=[], help="Extra hydra overrides.")
    args = parser.parse_args()

    config_dir = str((ExperimentPaths().experiment_root.parent / "conf").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=list(args.overrides))
    config = OmegaConf.to_container(cfg, resolve=True)

    dataset_config = config["train_dataset"]
    if args.spatial:
        dataset_config = dict(dataset_config, spatial=args.spatial, stats_spatial=args.spatial)

    vocab = build_vocab(dataset_config, config["modality_ids"])
    path = save_vocab(vocab, ExperimentPaths())
    print(f"Wrote {path}")
    for modality, variables in vocab["vocab"].items():
        for var, entry in variables.items():
            print(
                f"  {modality}/{var:<8} codes={entry['codes']} (+unknown id {entry['unknown_id']}) dim={entry['dim']}"
            )


if __name__ == "__main__":
    main()
