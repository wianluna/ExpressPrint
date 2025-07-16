from pathlib import Path
from typing import Any, Dict

import yaml


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)


# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)


def load_config(config_path: Path) -> Dict[str, Any]:
    try:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    except Exception as exc:
        raise RuntimeError(f"Cannot load training configuration {config_path}") from exc

    return config
