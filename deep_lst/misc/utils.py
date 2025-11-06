import yaml

def open_config(experiment: str, path: str = "deep_lst/experiments") -> dict:
    with open(f"{path}/{experiment}/config.yaml", 'r') as stream:
        return yaml.safe_load(stream)

