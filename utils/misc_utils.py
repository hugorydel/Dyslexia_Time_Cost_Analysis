"""
General utilities for dyslexia analysis
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def make_serializable(obj: Any) -> Any:
    """
    Convert non-serializable objects to serializable format for JSON output

    Args:
        obj: Object to make serializable

    Returns:
        Serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, "__dict__"):
        return str(obj)
    else:
        return obj


def save_json_results(results: dict, filepath: Path) -> None:
    """
    Save analysis results to JSON file

    Args:
        results: Dictionary with analysis results
        filepath: Path to save JSON file
    """
    try:
        serializable_results = make_serializable(results)
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save results to {filepath}: {e}")


def create_output_directories(base_dir: Path) -> dict:
    """
    Create necessary output directories

    Args:
        base_dir: Base directory for outputs

    Returns:
        Dictionary with created directory paths
    """
    directories = {
        "results": base_dir,
    }

    for name, dir_path in directories.items():
        dir_path.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Created directory: {dir_path}")

    return directories


def setup_logging(log_file: str = "dyslexia_analysis.log") -> None:
    """
    Setup logging configuration with UTF-8 encoding
    """
    import sys

    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding="utf-8")

    # Create console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)

    # Set formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def validate_config(config) -> None:
    """
    Validate configuration object has required attributes

    Args:
        config: Configuration object to validate

    Raises:
        AttributeError: If required configuration is missing
    """
    required_attrs = ["COPCO_PATH"]

    for attr in required_attrs:
        if not hasattr(config, attr):
            raise AttributeError(f"Configuration missing required attribute: {attr}")

    # Check if data path exists
    copco_path = Path(config.COPCO_PATH)
    if not copco_path.exists():
        logger.warning(f"CopCo data path does not exist: {copco_path}")
