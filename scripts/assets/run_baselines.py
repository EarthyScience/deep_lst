"""Regenerate the reference baselines for one eval region, off the preloaded columns.

Persistence + climatology come from the seviri LST series; the cross-modality reference comes
from a chosen modality variable (default era5_land skt). All read from the region's preload — the
same cache inference builds — so baseline coords match forecast.zarr exactly.

    uv run scripts/assets/run_baselines.py --region zambia_drought --test_year 2024
"""

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deep_lst.baselines import ClimatologyExtractor, Era5Extractor, Era5LandExtractor, PersistenceExtractor
from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.utils.config import split_views

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
    p.add_argument(
        "--force_sparse",
        action="store_true",
        help="Write (valid_time, pixel) even for dense boxes; must mirror the inference force_sparse flag.",
    )
    p.add_argument("--out_key", default=None, help="Baseline filename key; defaults to the spatial name.")
    return p.parse_args()


def build_config(region: str):
    """Full data config for a region: train.yaml's test split (stores/load) with the eval
    region's spatial/anchor/temporal overlaid — the same inputs preload_shared saw at inference.
    The test dataset dict the extractors read is attached as config["test_dataset"]."""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="train")
    region_cfg = OmegaConf.load(Path(CONF_DIR) / "eval" / f"{region}.yaml")
    cfg.splits.test = OmegaConf.merge(cfg.splits.test, region_cfg.splits.test)
    config = OmegaConf.to_container(cfg, resolve=True)
    config["test_dataset"] = split_views(config)["test"]
    return config


def main():
    args = parse_args()
    config = build_config(args.region)
    store_key = args.out_key or config["test_dataset"]["spatial"]

    shared = MultimodalDataset.preload_shared(config["modality_ids"], **config["test_dataset"])

    persistence = PersistenceExtractor(
        var_name="LST", test_year=args.test_year, overwrite=args.overwrite, force_sparse=args.force_sparse
    )
    persistence.run(shared, config, store_key)

    hourly = ClimatologyExtractor(
        var_name="LST",
        test_year=args.test_year,
        smooth=False,
        overwrite=args.overwrite,
        force_sparse=args.force_sparse,
    )
    hourly.run(shared, config, store_key)

    smoothed = ClimatologyExtractor(
        var_name="LST",
        test_year=args.test_year,
        smooth=True,
        window_days=args.window_days,
        overwrite=args.overwrite,
        force_sparse=args.force_sparse,
    )
    smoothed.run(shared, config, store_key)

    reference_cls = REFERENCE_EXTRACTORS[args.reference_modality]
    reference = reference_cls(
        var_name=args.reference_var,
        test_year=args.test_year,
        overwrite=args.overwrite,
        force_sparse=args.force_sparse,
    )
    reference.run(shared, config, store_key)


if __name__ == "__main__":
    main()
