import os

import hydra
import numpy as np
import torch
from eodata.dataset import deep_to
from eodata.eo_zarr import EOZarrStore
from eodata.schemes.torch.base import VariableType
from eodata.statistics import StandardStatisticsMetric
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from deep_lst.dataset.dataset import UnimodalDataset, load_samples
from deep_lst.utils.io import ExperimentPaths, save_statistics_npz


def calculate_all_statistics(
    store: EOZarrStore,
    reader_cls,
    indices: np.ndarray,
    increment_variables: list[str] | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
    **reader_kwargs,
) -> dict:
    """Compute per-variable mean, std, min, max, and optionally increment statistics.

    When increment_variables is non-empty, uses horizon=1 so each sample
    contains two consecutive timesteps, enabling both regular and increment stats
    (Y_{t+1} - Y_t) in a single pass with one preload. The last temporal index
    is excluded so horizon=1 never reads past the store end. Increment variables
    are moved from inputs to diagnostics to preserve NaN masking (avoiding
    nan_to_num bias on cloud-masked pixels). Increment stats are keyed as
    "{var}_increment".

    When increment_variables is empty or None, behaves as a plain statistics
    pass with horizon=0 over the full index set (the ERA5 path).
    """
    device = torch.device(device)
    increment_variables = increment_variables or []

    if increment_variables:
        # Exclude last timestep — horizon=1 needs t+1 to be valid.
        max_time = indices[:, 1].max()
        safe_indices = indices[indices[:, 1] < max_time]

        # horizon=1; move increment variables from inputs → diagnostics (NaN preservation).
        unified_kwargs = dict(reader_kwargs)
        unified_kwargs["horizon"] = 1
        unified_kwargs["inputs"] = [v for v in reader_kwargs.get("inputs", []) if v not in increment_variables]
        existing_diag = list(reader_kwargs.get("diagnostics", []))
        unified_kwargs["diagnostics"] = existing_diag + [v for v in increment_variables if v not in existing_diag]
    else:
        safe_indices = indices
        unified_kwargs = reader_kwargs

    reader = reader_cls(eozarr_store=store, **unified_kwargs)
    if unified_kwargs.get("preload", False):
        reader.preload()

    # Init process group AFTER preload so the NCCL watchdog doesn't expire
    # while ranks are busy loading large stores (e.g. ERA5's 15 GB) from NFS.
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    dataset = UnimodalDataset(reader=reader, indices=safe_indices)

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

    inc_metrics: dict[str, StandardStatisticsMetric] = {
        var: StandardStatisticsMetric(dim=(0, 1, 2, 3)).to(device=device) for var in increment_variables
    }

    use_tqdm = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    for batch in tqdm(dataloader, disable=not use_tqdm, smoothing=0.005):
        batch = deep_to(batch, device=device, non_blocking=True)
        for key in metrics:
            if key in batch:
                data = batch[key]
                # With horizon=1, time-varying variables have shape (B, 2, H, W, C).
                # Take t=0 slice to recover (B, H, W, C) for the metric dim indices.
                if data.ndim == 5:
                    data = data[:, 0, ...]
                metrics[key].update(data)
        for var in increment_variables:
            if var in batch:
                # batch[var]: (B, 2, H, W, C) — index 0 = t, index 1 = t+1
                inc_metrics[var].update(batch[var][:, 1, ...] - batch[var][:, 0, ...])

    all_stats: dict = {}
    for var_name, metric in metrics.items():
        stats = metric.compute()
        all_stats[var_name] = {k: v.cpu().numpy() for k, v in stats.items()}
    for var, metric in inc_metrics.items():
        stats = metric.compute()
        all_stats[f"{var}_increment"] = {k: v.cpu().numpy() for k, v in stats.items()}

    return all_stats


@hydra.main(config_path="../../conf", config_name="statistics", version_base="1.3")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    modality = config["modality"]
    modality_cfg = config[modality]

    label = config["spatial"]  # e.g. 'KGLC_L'

    # ── device ────────────────────────────────────────────────────────────────
    # init_process_group is deferred until after preloading (inside
    # calculate_all_statistics), so derive device/rank from env vars directly.
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── reader class ──────────────────────────────────────────────────────────
    if modality == "era5":
        from eodata.data_reader.era5 import Era5Reader as ReaderCls
    else:
        from eodata.data_reader.seviri import SeviriReader as ReaderCls

    paths = ExperimentPaths()
    reader_kwargs = {
        "inputs": modality_cfg["inputs"],
        "diagnostics": modality_cfg["diagnostics"],
        "prognostics": modality_cfg["prognostics"],
        "row_extent": modality_cfg["row_extent"],
        "col_extent": modality_cfg["col_extent"],
        "lookback": 0,
        "horizon": 0,
        "preload": config["preload"],
    }

    store = EOZarrStore(config["store_path"], mode="r+")
    indices = load_samples(
        path=paths.spatial_indices_path(modality, config["spatial"]),
        temporal_indices_path=paths.temporal_indices_path(config["temporal"]),
    )

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"Modality     : {modality}")
        print(f"Store        : {config['store_path']}")
        print(f"Spatial      : {config['spatial']}  ({len(np.unique(indices[:, 0]))} unique pixels)")
        print(f"Temporal     : {config['temporal']}  ({len(np.unique(indices[:, 1]))} unique timesteps)")
        print(f"Total samples: {len(indices):,}")
        print(f"Label        : {label}")

    increment_variables = modality_cfg.get("increment_variables", [])
    if rank == 0 and increment_variables:
        print(f"Computing regular + increment statistics for: {increment_variables}")
    all_stats = calculate_all_statistics(
        store,
        reader_cls=ReaderCls,
        indices=indices,
        increment_variables=increment_variables,
        device=device,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        **reader_kwargs,
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ── save (rank 0 only) ────────────────────────────────────────────────────
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        for var_name, var_stats in all_stats.items():
            print(f"  {var_name}: mean={var_stats['mean']:.4f}  std={var_stats['std']:.4f}")

        # Save to npz (for compatibility with existing training configs)
        out_path = paths.statistics_path(modality, label)
        save_statistics_npz(all_stats, out_path)
        print(f"\nSaved statistics npz → {out_path}")

        # Embed statistics in the zarr store under the experiment label.
        # Computed variables (solar_zenith, doy_sin, doy_cos) and increment
        # variables have no backing zarr array — set_statistics stores them
        # on the root group automatically.
        for var_name, var_stats in all_stats.items():
            store.set_statistics(modality, var_name, var_stats, label=label, overwrite=True)
        print(f"Embedded statistics in zarr store (label='{label}')")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
