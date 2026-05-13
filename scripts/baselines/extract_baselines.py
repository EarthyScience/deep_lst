"""
Extract baseline from specific store.
uv run scripts/benchmarks/extract_baselines.py --store eval_dahra --test_year 2024
"""

import argparse

from deep_lst.baselines.extractor import ClimatologyExtractor, Era5Extractor, PersistenceExtractor


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--test_year",
        type=int,
        required=True,
        help="Year to produce the climatology for (default: None). All hours of this year are written to the output.",
    )
    p.add_argument(
        "--store",
        type=str,
        required=True,
        help=("Override the zarr store spatial key. Default: None. Use e.g. KG_L for training-site eval regions."),
    )
    p.add_argument(
        "--window_days", type=int, default=7, help="Window size in days for smoothed climatology (default: 7)"
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return p.parse_args()


def main():
    args = parse_args()

    persistence_extractor = PersistenceExtractor(
        var_name="LST",
        test_year=args.test_year,
        overwrite=args.overwrite,
    )
    persistence_extractor.run(store=args.store, modality="seviri")

    climatology_extractor = ClimatologyExtractor(
        var_name="LST",
        test_year=args.test_year,
        window_days=args.window_days,
        overwrite=args.overwrite,
    )
    climatology_extractor.run(
        store=args.store,
        modality="seviri",
        mode="hourly",
    )
    climatology_extractor.run(
        store=args.store,
        modality="seviri",
        mode="smoothed_hourly",
    )

    era5_extractor = Era5Extractor(
        var_name="skt",
        test_year=args.test_year,
        overwrite=args.overwrite,
    )
    era5_extractor.run(store=args.store)


if __name__ == "__main__":
    main()
