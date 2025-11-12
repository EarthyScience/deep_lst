import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print(os.getcwd())  # ensure we are in deep_lst/deep_lst/mlp

import numpy as np
import torch
import pandas as pd
import argparse

from deep_lst.mlp.model import MLP
from deep_lst.mlp.datareader import weTestDataset
from deep_lst.misc.utils import open_config, load_checkpoint
from deep_lst.misc.normalisation import Normaliser

from torch.utils.data import DataLoader

argparser = argparse.ArgumentParser()
argparser.add_argument("--experiment", default="", type=str, help="Name of the experiment.")
args = argparser.parse_args()

def inference(config, version, log_path):

    testData = weTestDataset(config=config, start_time=config['test_start'], end_time=config['test_end'])
    test_loader = DataLoader(
        testData,
        batch_size=config['batch_size']*4,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        collate_fn=testData._collate_fn
    )

    hidden_sizes = config['hidden_sizes']  # Example hidden layer sizes

    model = MLP(input_size=len(testData.feature_list), 
                hidden_size=hidden_sizes, 
                output_size=len(testData.target_list))

    checkpoint_path = load_checkpoint(experiment=config['experiment_name'], version=version, path="./experiments/")
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
    df.to_parquet(log_path + "inference_results.parquet")

    return np.asarray(predictions), np.asarray(targets)


if __name__ == "__main__":
    
    experiment = args.experiment
    config = open_config(experiment=experiment, path="./experiments")
    version = "version_0"   
    log_path = config["experiment_path"] + f"logs/prototype_{experiment}/{version}/"

    predictions, targets = inference(config=config, version=version, log_path=log_path)

    print("Inference completed.")