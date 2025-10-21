# src/utils/config_loader.py

import os
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_PATH = "configs/params.yaml"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path (str): Path to YAML file. Defaults to 'configs/params.yaml'.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML content is invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise yaml.YAMLError("Top-level YAML content must be a mapping/dict.")

    return cfg


def get_param(cfg: Dict[str, Any], dotted_key: str, default: Optional[Any] = None) -> Any:
    """
    Retrieve a value from a nested dict using dotted notation.

    Example:
        get_param(cfg, "model.learning_rate", 0.1)

    Args:
        cfg (dict): Configuration dictionary.
        dotted_key (str): Dotted path to the key (e.g., "model.max_depth").
        default (Any): Default value if key is missing.

    Returns:
        Any: Retrieved value or default if not found.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node
