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
from deep_lst.mlp.datareader import wedataReader, weTestDataset

from deep_lst.misc.filter import SpatialFilter, TemporalFilter

from deep_lst.misc.utils import open_config, open_hparams, load_checkpoint
from deep_lst.misc.normalisation import Normaliser
from torch.utils.data import DataLoader, Dataset


def inference(config):

    testData = weTestDataset(config=config, start_time=config['test_start'], end_time=config['test_end'])
    test_loader = DataLoader(
        testData,
        batch_size=config['batch_size']*2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=testData._collate_fn
    )

    hidden_sizes = config['hidden_sizes']  # Example hidden layer sizes

    model = MLP(input_size=len(testData.feature_list), 
                hidden_size=hidden_sizes, 
                output_size=len(testData.target_list))

    checkpoint_path = load_checkpoint(experiment=config['experiment_name'], version="version_0", path="./experiments/")
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

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
    df.to_parquet(config['experiment_path'] + f"logs/prototype_{config['experiment_name']}/version_0/inference_results.parquet")

    return np.asarray(predictions), np.asarray(targets)


if __name__ == "__main__":
    
    config = open_config(experiment="baseline_0", path="./experiments")
    print(config)
    predictions, targets = inference(config=config)

    print("Predictions shape:", predictions.shape)
    print("Targets shape:", targets.shape)
    print("Predictions shape:", predictions)
    print("Targets shape:", targets)
    print("Targets shape:", np.isfinite(targets).sum())