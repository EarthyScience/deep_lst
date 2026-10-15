"""Perturbed inference: model, preload and dataset of an experiment are built once, then one
forecast store is written per perturbed input, next to the run's own forecast.zarr:

    results/<experiment>/<version>/<eval>/sensitivity/<method>_<block>/<modality>.<variable>.zarr
"""

from torch.utils.data import DataLoader

from deep_lst.dataset.collator import Collator
from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.evaluation.inference import Forecast
from deep_lst.utils.config import get_keys_from_config, get_packing_keys, load_model

from .perturb import STATIC_LEAVES, locate, perturbed


def stored_inputs(config, var_names):
    """(modality, variable) of every temporal input the model reads from the stores, in config
    order; computed features and static leaves are left out."""
    inputs = []
    for modality in config["modality_ids"]:
        for variable in config["test_dataset"][modality].get("inputs", []):
            try:
                leaf, _ = locate(var_names, modality, variable)
            except KeyError:
                continue  # computed per sample
            if leaf not in STATIC_LEAVES:
                inputs.append((modality, variable))
    return inputs


def sensitivity_dir(paths, experiment, version, eval_name, method, block):
    return paths.results_dir(experiment, version, eval_name) / "sensitivity" / f"{method}_{block}"


class PerturbedInference:
    """Holds the loaded model and the preloaded dataset of one (experiment, eval)."""

    def __init__(self, cfg, config):
        self.cfg = cfg  # hydra: experiment, version, best, land_water_mask
        self.config = config  # resolved run config
        self.td_keys = get_keys_from_config(config["test_dataset"])
        self.model, self.device = load_model(
            config, self.td_keys, experiment=cfg.experiment, version=cfg.version, prefer_best=cfg.best
        )
        self.shared = MultimodalDataset.preload_shared(config["modality_ids"], **config["test_dataset"])
        self.dataset = MultimodalDataset.from_modalities(config["modality_ids"], self.shared, **config["test_dataset"])

    def forecast(self, path):
        """One pass over the dataset as it is now (perturbed or not) into a forecast store.
        The loader is created here so that its workers fork after the perturbation."""
        analysis = self.config["analysis"]
        loader = DataLoader(
            self.dataset,
            batch_size=analysis["batch_size"],
            num_workers=analysis["num_workers"],
            shuffle=False,
            drop_last=False,
            collate_fn=Collator(self.config["model"]["name"], get_packing_keys(self.config["test_dataset"])),
            pin_memory=analysis.get("pin_memory", False),
            persistent_workers=analysis["num_workers"] > 0,
            prefetch_factor=analysis.get("prefetch_factor", 2),
        )
        forecast = Forecast(
            model=self.model,
            device=self.device,
            loader=loader,
            td_keys=self.td_keys,
            config=self.config["test_dataset"],
            spatial_indices=self.dataset.spatial_idx,
            pixel_lonlat=self.dataset.pixel_lonlat(),
            non_land=self.dataset.non_land_mask(),
            prediction_mode=self.config["model"]["step_loss"],
            apply_lw_mask=self.cfg.land_water_mask,
        )
        forecast.predict_and_save(path, self.dataset.time_idx)
        Forecast.rechunk_store(path)

    def run(self, out_dir, variables, method, block, seed):
        """One store per (modality, variable); existing stores are kept."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for modality, variable in variables:
            path = out_dir / f"{modality}.{variable}.zarr"
            if path.exists():
                print(f"skip {path.name}: exists")
                continue
            print(f"perturbing {modality}.{variable} ({method}, block={block}, seed={seed})")
            with perturbed(self.shared, modality, variable, method, block, seed):
                self.forecast(path)
