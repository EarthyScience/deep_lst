"""Regenerate the reference baselines for one eval region, off the preloaded columns.

Persistence + climatology come from the seviri LST series; the cross-modality reference comes
from a chosen modality variable (default era5_land skt). All read from the region's preload — the
same cache inference builds — so baseline coords match forecast.zarr exactly.

    uv run scripts/baselines/run_baselines.py --region passage --test_year 2024
"""

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deep_lst.baselines import ClimatologyExtractor, Era5Extractor, Era5LandExtractor, PersistenceExtractor
from deep_lst.dataset.dataset import MultimodalDataset

CONF_DIR = "/Net/Groups/BGI/people/mwessel/deep_lst/conf"

REFERENCE_EXTRACTORS = {"era5": Era5Extractor, "era5_land": Era5LandExtractor}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", type=str, required=True, help="Eval config name, e.g. passage, dahra, train_mtg.")
    p.add_argument("--test_year", type=int, required=True, help="Year the baselines forecast (train years precede it).")
    p.add_argument("--window_days", type=int, default=7, help="Circular DOY window for the smoothed climatology.")
    p.add_argument(
        "--reference_modality",
        type=str,
        default="era5_land",
        choices=sorted(REFERENCE_EXTRACTORS),
        help="Modality for the cross-modality reference.",
    )
    p.add_argument("--reference_var", type=str, default="skt", help="Variable for the cross-modality reference.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing baseline files.")
    return p.parse_args()


def build_config(region: str):
    """Full data config for a region: train.yaml's test_dataset (stores/load) with the eval
    region's spatial/anchor/temporal overlaid — the same inputs preload_shared saw at inference."""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="train")
    region_cfg = OmegaConf.load(Path(CONF_DIR) / "eval" / f"{region}.yaml")
    cfg.test_dataset = OmegaConf.merge(cfg.test_dataset, region_cfg.test_dataset)
    return OmegaConf.to_container(cfg, resolve=True)


def main():
    args = parse_args()
    config = build_config(args.region)
    store_key = config["test_dataset"]["spatial"]

    shared = MultimodalDataset.preload_shared(config["modality_ids"], **config["test_dataset"])

    persistence = PersistenceExtractor(var_name="LST", test_year=args.test_year, overwrite=args.overwrite)
    persistence.run(shared, config, store_key)

    hourly = ClimatologyExtractor(var_name="LST", test_year=args.test_year, smooth=False, overwrite=args.overwrite)
    hourly.run(shared, config, store_key)

    smoothed = ClimatologyExtractor(
        var_name="LST", test_year=args.test_year, smooth=True, window_days=args.window_days, overwrite=args.overwrite
    )
    smoothed.run(shared, config, store_key)

    reference_cls = REFERENCE_EXTRACTORS[args.reference_modality]
    reference = reference_cls(var_name=args.reference_var, test_year=args.test_year, overwrite=args.overwrite)
    reference.run(shared, config, store_key)


if __name__ == "__main__":
    main()
