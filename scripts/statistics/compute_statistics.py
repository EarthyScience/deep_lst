import argparse
import os

import numpy as np
import torch
from eodata.dataset import deep_to
from eodata.eo_zarr import EOZarrStore
from eodata.schemes.torch.base import VariableType
from eodata.statistics import StandardStatisticsMetric
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from deep_lst.dataset.dataset import UnimodalDataset
from deep_lst.utils.config import open_config
from deep_lst.utils.io import ExperimentPaths, save_statistics_npz


def _load_indices(spatial_path: str, modality: str, temporal_path: str) -> np.ndarray:
    """Load spatial and temporal indices and return a meshed (N_space * N_time, 2) array.

    Supports both key conventions for spatial .npz files:
      - '{modality}_indices'  used by scene files
      - 'mapping'             used by global coverage files (e.g. indices_1p_global.npz)

    Temporal file must contain key 'time_indices' (int64 T1H positions), as written
    by eodata.grids.temporal_filter.TemporalFilter.save().
    """
    spatial_data = np.load(spatial_path)
    spatial_idx = spatial_data.get(f"{modality}_indices", spatial_data.get("mapping"))
    if spatial_idx is None:
        raise KeyError(
            f"Spatial index file {spatial_path!r} contains neither "
            f"'{modality}_indices' nor 'mapping'. "
            f"Available keys: {list(spatial_data.keys())}"
        )

    temporal_idx = np.load(temporal_path)["time_indices"].astype(np.int64)

    space, time = np.meshgrid(spatial_idx, temporal_idx, indexing="ij")
    indices = np.stack([space.ravel(), time.ravel()], axis=1)  # (N_space * N_time, 2)
    return indices


def calculate_statistics(
    store: EOZarrStore,
    reader_cls,
    indices: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
    **reader_kwargs,
) -> dict:
    """Compute per-variable mean, std, min, max over the given (spatial, temporal) index pairs.

    reader_cls is instantiated here without calling set_normalizer(), so data is
    returned in physical units. Only CONTINUOUS variables (per the reader's torch
    schema) receive statistics; CATEGORICAL variables (lsm, LWMASK, LANDCOV) are
    skipped automatically.

    dim convention: reduced_dimensions_indices is 0-based within the variable's own
    dims; +1 shifts each index to account for the batch dimension prepended by the
    DataLoader, so dim=(0, 1, 2, 3) reduces over (batch, time, lat, lon) and leaves
    the trailing channel dim (size 1) unreduced. save_statistics_npz calls .item()
    to flatten that to a scalar.
    """
    device = torch.device(device)

    # Instantiate reader without set_normalizer() — data stays in physical units.
    reader = reader_cls(eozarr_store=store, **reader_kwargs)
    dataset = UnimodalDataset(reader=reader, indices=indices)

    if torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False, seed=0)
    else:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False, seed=0, rank=0, num_replicas=1)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        # No collate_fn: default collate handles TensorDict batches correctly.
    )

    schema = reader.get_torch_schema()
    metrics: dict[str, StandardStatisticsMetric] = {}
    for name, var in schema.variables.items():
        if var.var_type == VariableType.CONTINUOUS:
            # reduced_dimensions_indices: 0-based positions in the variable's own dims.
            # +1 shifts each index past the batch dim prepended by DataLoader.
            dim = (0,) + tuple(idx + 1 for idx in var.reduced_dimensions_indices)
            metrics[name] = StandardStatisticsMetric(dim=dim).to(device=device)

    use_tqdm = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    for batch in tqdm(dataloader, disable=not use_tqdm, smoothing=0.005):
        batch = deep_to(batch, device=device, non_blocking=True)
        for key in metrics:
            if key in batch:
                metrics[key].update(batch[key])

    all_variables: dict = {}
    for var_name, metric in metrics.items():
        stats = metric.compute()
        all_variables[var_name] = {k: v.cpu().numpy() for k, v in stats.items()}

    return all_variables


def main():
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(device_id=int(os.environ["LOCAL_RANK"]))

    parser = argparse.ArgumentParser(
        description="Compute normalisation statistics for an EOZarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--store",
        required=True,
        help="Path to the EOZarr store (.zarr directory).",
    )
    parser.add_argument(
        "-i",
        "--spatial_indices",
        required=True,
        help=(
            "Path to spatial indices (.npz). "
            "Accepted keys: '{modality}_indices' (scene files) or 'mapping' (global files)."
        ),
    )
    parser.add_argument(
        "-t",
        "--temporal_indices",
        required=True,
        help=(
            "Path to temporal block indices (.npz, key: 'time_indices', dtype int64). "
            "Use assets/temporal/time_indices_train.npz for the standard training blocks."
        ),
    )
    parser.add_argument(
        "-m",
        "--modality",
        required=True,
        choices=["era5", "seviri"],
        help="Modality to compute statistics for.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="statistics",
        help="Config name under configs/ — used to read variable lists (default: statistics).",
    )
    parser.add_argument(
        "--statistics",
        default="global",
        help="Output label; statistics are saved to assets/{modality}/statistics/statistics_{label}.npz.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available() and torch.distributed.is_initialized():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── reader class ──────────────────────────────────────────────────────────
    if args.modality == "era5":
        from wedata.data_reader.era5 import Era5Reader as ReaderCls
    else:
        from wedata.data_reader.seviri import SeviriReader as ReaderCls

    # ── reader kwargs from config ─────────────────────────────────────────────
    # Variable lists come from the training config so statistics always cover
    # exactly the variables the model uses. patch_size, lookback, and horizon
    # are overridden to point-wise single-step to avoid redundant overlap between
    # adjacent time windows and to minimise memory.
    paths = ExperimentPaths()
    config = open_config(paths.config_path(args.config))
    modality_cfg = config[args.modality]
    reader_kwargs = {
        "inputs": modality_cfg["inputs"],
        "diagnostics": modality_cfg["diagnostics"],
        "prognostics": modality_cfg["prognostics"],
        "row_extent": modality_cfg["row_extent"],
        "col_extent": modality_cfg["col_extent"],
        "patch_size": 1,  # always point-wise for statistics
        "lookback": 0,  # single timestep — no window overlap
        "horizon": 0,
    }

    # ── indices ───────────────────────────────────────────────────────────────
    store = EOZarrStore(args.store, mode="r")
    indices = _load_indices(args.spatial_indices, args.modality, args.temporal_indices)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        print(f"Modality     : {args.modality}")
        print(
            f"Spatial file : {args.spatial_indices}  ({indices[:, 0].nunique() if hasattr(indices[:, 0], 'nunique') else len(np.unique(indices[:, 0]))} unique pixels)"
        )
        print(f"Temporal file: {args.temporal_indices}  ({len(np.unique(indices[:, 1]))} unique timesteps)")
        print(f"Total samples: {len(indices):,}")

    # ── compute ───────────────────────────────────────────────────────────────
    stats = calculate_statistics(
        store,
        reader_cls=ReaderCls,
        indices=indices,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **reader_kwargs,
    )

    # ── save (rank 0 only) ────────────────────────────────────────────────────
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        for var_name, var_stats in stats.items():
            print(f"Variable: {var_name}")
            for stat_name, value in var_stats.items():
                print(f"  {stat_name}: {value}")

        out_path = paths.statistics_path(args.modality, args.statistics)
        save_statistics_npz(stats, out_path)
        print(f"Saved statistics to {out_path}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
