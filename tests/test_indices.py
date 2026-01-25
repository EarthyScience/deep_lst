import pandas as pd

from deep_lst.misc.utils import open_config


def test_indices():
    config = open_config("global", path="experiments")

    era5_indices = pd.read_parquet(config["experiment_path"] + config["era5"]["spatial_indices"])
    seviri_indices = pd.read_parquet(config["experiment_path"] + config["seviri"]["spatial_indices"])

    assert not era5_indices.duplicated(["row_idx", "col_idx"]).any()
    assert not seviri_indices.duplicated(["row_idx", "col_idx"]).any()

    assert set(zip(seviri_indices.row_idx, seviri_indices.col_idx)) == set(
        zip(era5_indices.row_idx, era5_indices.col_idx)
    )

    dup_rate = 1 - len(era5_indices[["lat_idx", "lon_idx"]].drop_duplicates()) / len(era5_indices)
    print(f"ERA5 duplication rate: {dup_rate:.1%}")

    print(era5_indices.distance.describe())
    print(seviri_indices.distance.describe())
