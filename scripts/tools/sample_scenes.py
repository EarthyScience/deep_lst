import argparse

import numpy as np
from wedata.grids import EquiangularGrid, FilteredGridMapping, filter_mapping
from wedata.util.spatial import get_landmask_from_zarrstore

from deep_lst.utils.config import open_config
from deep_lst.utils.io import ExperimentPaths


def build_parser():
    parser = argparse.ArgumentParser(description="Create filtered spatial grid mappings between SEVIRI and ERA5 grids.")
    parser.add_argument(
        "--src_grid",
        type=str,
        required=True,
        help='specify which grid to use as source grid (e.g., "era5", "seviri")',
    )

    parser.add_argument(
        "--scenes_config",
        type=str,
        required=True,
        help='config file specifying scenes (e.g., "scenes.yml") needs training, validation, and test splits specified',
    )

    return parser


def main():
    args = build_parser().parse_args()

    paths = ExperimentPaths()
    scenes_config = open_config(paths.scenes_path(experiment=args.scenes_config))

    # change this in the future to dynamically subset with naturalearth mask
    land_mask = get_landmask_from_zarrstore(
        f"/Net/Groups/BGI/work_5/scratch/mwessel/wegen-stores/{args.src_grid}.zarr", layer_name="LWMASK", land_value=1
    )

    era5 = EquiangularGrid(
        resolution=0.25,
        latitude_bounds=(90.0, -90.0),  # descending
        longitude_bounds=(-180.0, 180.0),
        include_xmax=False,
        include_ymin=True,
    )

    seviri = EquiangularGrid(
        resolution=0.05,
        latitude_bounds=(80.0, -80.0),  # (80, -80),
        longitude_bounds=(-80.0, 80.0),  # (-80, 80),
        include_ymin=True,  # include edges
        include_xmax=True,
    )

    # create mappings
    mapping_era = FilteredGridMapping.from_nearest_neighbors(seviri, era5, precise=False, cached=False)
    mapping_sev = FilteredGridMapping.from_nearest_neighbors(
        seviri, seviri, precise=False, cached=False
    )  # identity mapping

    for split in ["train", "val", "test"]:
        scenes = scenes_config[split]
        for scene_id, _ in scenes.items():
            mapping_era5 = filter_mapping(
                mapping=mapping_era,
                land_mask=land_mask,
                bbox=scenes[scene_id]["bbox"],
            )
            mapping_seviri = filter_mapping(
                mapping=mapping_sev,
                land_mask=land_mask,
                bbox=scenes[scene_id]["bbox"],
            )

            scene = {
                "era5_indices": mapping_era5.indices,
                "seviri_indices": mapping_seviri.indices,
                "scene_bbox": scenes[scene_id]["bbox"],
                "scene_id": scene_id,
            }

            np.savez_compressed(
                f"/Net/Groups/BGI/people/mwessel/deep_lst/assets/samples/{args.scenes_config}_{split}_{scene_id}.npz",
                **scene,
            )


if __name__ == "__main__":
    main()
