# Pipeline walkthrough when running a training

Prerequisites:
- a config.yml file in an experiment folder with a dedicated name.

What happens when you call
```
python deep_lst/mlp/train.py --config mlp_rollout
```

1. train.py loads the config, instantiates LitDataModule(**config), calls data_module.setup().
2. LitDataModule.setup() calls MultimodalDataset.from_modalities() three times — for train, val, and test — unpacking the respective sub-config dict each time.
3. MultimodalDataset.from_modalities() calls _load_samples() for each modality, which is where a branching between temporal_indices_path and start_time/end_time happens - depending of using pre-computed time indices. Will be condensed to one branch in the future.
4. _load_samples(): if temporal_indices_path is not None use that path, otherwise fall back to the date range.
5. Model build: get_keys_from_config() iterates the dataset config, skips non-dict values (split, start_time, end_time) and builds the key tuples.
6. MultiModalMLP is instantiated with those keys, wrapped in LitModelModule, and RolloutMSE is dynamically imported via importlib.
