from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_detection_score_threshold() -> float:
    """Read and validate the single configured detection threshold."""
    try:
        threshold = float(load_config()["inference"]["score_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("config.inference.score_threshold must be a number.") from error

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("config.inference.score_threshold must be between 0 and 1.")
    return threshold
