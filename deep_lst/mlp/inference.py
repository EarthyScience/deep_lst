import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print(os.getcwd())  # ensure we are in deep_lst/deep_lst/mlp

import xarray as xr
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from deep_lst.mlp.datamodule import MLPDataModule
from deep_lst.mlp.model import MLP
from deep_lst.mlp.datareader import wedataReader

from deep_lst.misc.filter import SpatialFilter, TemporalFilter

from deep_lst.misc.utils import open_config, open_hparams, load_checkpoint
from deep_lst.misc.normalisation import Normaliser
from torch.utils.data import DataLoader, Dataset


def inference(config,
              hparams = None):

    data_module = MLPDataModule(config=config, 
                                batch_size=config['batch_size']*4, 
                                num_workers=4)
    data_module.setup()

    hidden_sizes = config['hidden_sizes']  # Example hidden layer sizes

    model = MLP(input_size=len(data_module.feature_list), 
                hidden_size=hidden_sizes, 
                output_size=len(data_module.target_list))

    checkpoint_path = load_checkpoint(experiment="baseline", version="version_0", path="./deep_lst/experiments/")
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    test_loader = data_module.test_dataloader()

    predictions = []
    targets = []

    with torch.no_grad():
        for features, target in test_loader:
            
            output = model(features)

            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            print("first load.")

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    normaliser = Normaliser(config=config, group='seviri')

    predictions = normaliser.denormalise(predictions, var='LST')
    targets = normaliser.denormalise(targets, var='LST')

    df = pd.DataFrame({"predictions": predictions.flatten(), "targets": targets.flatten()})
    df.to_parquet(config['experiment_path'] + "logs/prototype_baseline/version_0/inference_results.parquet")
    
    return np.asarray(predictions), np.asarray(targets)


if __name__ == "__main__":
    
    config = open_config(experiment="baseline", path="./deep_lst/experiments")
    print(config)
    predictions, targets = inference(config=config, hparams=None)

    print("Predictions shape:", predictions.shape)
    print("Targets shape:", targets.shape)
    print("Predictions shape:", predictions)
    print("Targets shape:", targets)
    print("Targets shape:", np.isfinite(targets).sum())