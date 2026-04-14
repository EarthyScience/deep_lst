"""
Compute per-variable normalisation statistics for a sampled EOZarr store.

Statistics (mean, std, min, max) are computed via Welford's online algorithm
over a spatial × temporal index mesh, then written to two places:
  1. assets/{modality}/statistics/statistics_{spatial}.npz  — for legacy loaders
  2. zarr array attributes (or root group for computed vars) — for training

Which indices are used
----------------------
--spatial   Name of the spatial index file, e.g. KGLC_L.
            Resolves to: assets/{modality}/spatial/indices_{spatial}.npz
            The spatial key read from the file is '{modality}_indices', with
            'mapping' as fallback (older files).

--temporal  Name of the temporal index file (default: train).
            Resolves to: assets/temporal/time_indices_{temporal}.npz

The statistics label stored in the zarr store equals --spatial, so training
configs that reference label='KGLC_L' will load the right statistics.

Which variables are included
-----------------------------
Controlled by configs/statistics/config.yml (--config flag).
The config lists inputs/diagnostics/prognostics per modality.
Only CONTINUOUS variables (per the reader's torch schema) receive statistics;
CATEGORICAL variables (lsm, LWMASK, LANDCOV) are loaded but skipped.
Computed variables (solar_zenith, doy_sin, doy_cos) have no backing zarr
array — their statistics are stored on the root group of the store instead.

Typical usage
-------------
Single GPU (development / small stores):
    python scripts/statistics/compute_statistics.py \\
        -s /path/to/seviri_sampled_KGLC_L.zarr \\
        -m seviri \\
        --spatial KGLC_L \\
        --temporal train \\
        --preload

Multi-GPU via torchrun (recommended for large stores):
    torchrun --nproc_per_node=4 scripts/statistics/compute_statistics.py \\
        -s /path/to/seviri_sampled_KGLC_L.zarr \\
        -m seviri \\
        --spatial KGLC_L \\
        --temporal train \\
        --preload

Nested sampling strategy
------------------------
For nested splits (S ⊂ M ⊂ L), compute statistics once on the largest set (L)
and use that label for all levels — the distribution is representative for all.
Run once per modality × spatial strategy, not once per size level.
"""

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


def _load_indices(modality: str, spatial_path, temporal_path) -> np.ndarray:
    """Load spatial and temporal indices and return a meshed (N_space * N_time, 2) array."""
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
    """Compute per-variable mean, std, min, max over the given (spatial, temporal) index pairs."""
    device = torch.device(device)

    reader = reader_cls(eozarr_store=store, **reader_kwargs)
    if reader_kwargs.get("preload", False):
        reader.preload()
    dataset = UnimodalDataset(reader=reader, indices=indices)

    if torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, seed=0)
    else:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, seed=0, rank=0, num_replicas=1)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=torch.stack,
        pin_memory=True,
    )

    schema = reader.get_torch_schema()
    metrics: dict[str, StandardStatisticsMetric] = {}
    for name, var in schema.variables.items():
        if var.var_type == VariableType.CONTINUOUS:
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
        torch.distributed.init_process_group(device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))

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
        "-m",
        "--modality",
        required=True,
        choices=["era5", "seviri"],
        help="Modality to compute statistics for.",
    )
    parser.add_argument(
        "--spatial",
        required=True,
        help="Spatial index name (e.g. KGLC_L). Resolves to assets/{modality}/spatial/indices_{name}.npz. "
        "Also used as the statistics label and output filename suffix.",
    )
    parser.add_argument(
        "--temporal",
        default="train",
        help="Temporal split name (default: train). Resolves to assets/temporal/time_indices_{name}.npz.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="statistics",
        help="Config name under configs/ — used to read variable lists (default: statistics).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--preload",
        action="store_true",
        default=False,
        help="Preload all zarr arrays into RAM before computing statistics. "
        "Recommended for sampled stores (~9 GB total).",
    )
    args = parser.parse_args()

    # Label and output name derived from the spatial index name
    label = args.spatial  # e.g. 'KGLC_L'

    # ── device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available() and torch.distributed.is_initialized():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── reader class ──────────────────────────────────────────────────────────
    if args.modality == "era5":
        from eodata.data_reader.era5 import Era5Reader as ReaderCls
    else:
        from eodata.data_reader.seviri import SeviriReader as ReaderCls

    # ── reader kwargs from config ─────────────────────────────────────────────
    paths = ExperimentPaths()
    config = open_config(paths.config_path(args.config))
    modality_cfg = config[args.modality]
    reader_kwargs = {
        "inputs": modality_cfg["inputs"],
        "diagnostics": modality_cfg["diagnostics"],
        "prognostics": modality_cfg["prognostics"],
        "row_extent": modality_cfg["row_extent"],
        "col_extent": modality_cfg["col_extent"],
        "lookback": 0,
        "horizon": 0,
        "preload": args.preload,
    }

    # ── indices ───────────────────────────────────────────────────────────────
    store = EOZarrStore(args.store, mode="r+")
    indices = _load_indices(
        args.modality,
        spatial_path=paths.spatial_indices_path(args.modality, args.spatial),
        temporal_path=paths.temporal_indices_path(args.temporal),
    )

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        print(f"Modality     : {args.modality}")
        print(f"Store        : {args.store}")
        print(f"Spatial      : {args.spatial}  ({len(np.unique(indices[:, 0]))} unique pixels)")
        print(f"Temporal     : {args.temporal}  ({len(np.unique(indices[:, 1]))} unique timesteps)")
        print(f"Total samples: {len(indices):,}")
        print(f"Label        : {label}")

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

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ── save (rank 0 only) ────────────────────────────────────────────────────
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        for var_name, var_stats in stats.items():
            print(f"  {var_name}: mean={var_stats['mean']:.4f}  std={var_stats['std']:.4f}")

        # Save to npz (for compatibility with existing training configs)
        out_path = paths.statistics_path(args.modality, label)
        save_statistics_npz(stats, out_path)
        print(f"\nSaved statistics npz → {out_path}")

        # Embed statistics in the zarr store under the experiment label.
        # Computed variables (solar_zenith, doy_sin, doy_cos) have no backing
        # zarr array — set_statistics stores them on the root group automatically.
        for var_name, var_stats in stats.items():
            store.set_statistics(args.modality, var_name, var_stats, label=label, overwrite=True)
        print(f"Embedded statistics in zarr store (label='{label}')")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
