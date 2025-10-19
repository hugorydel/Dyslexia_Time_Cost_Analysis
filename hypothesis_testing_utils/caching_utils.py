"""
Caching utilities to avoid re-running expensive computations
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def compute_data_hash(data: pd.DataFrame) -> str:
    """Compute hash of dataset for cache invalidation"""
    # Use shape + first/last rows + column names
    hash_input = (
        f"{data.shape}_{data.columns.tolist()}_"
        f"{data.head(5).values.tobytes()}_{data.tail(5).values.tobytes()}"
    )
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def load_or_compute(
    cache_path: Path, compute_fn: Callable, *args, reuse: bool = True, **kwargs
) -> Any:
    """
    Load from cache or compute and save

    Args:
        cache_path: Path to cache file
        compute_fn: Function to call if cache miss
        reuse: If True, load from cache if available
        *args, **kwargs: Arguments to pass to compute_fn

    Returns:
        Computed or cached result
    """
    if reuse and cache_path.exists():
        logger.info(f"✓ Loading from cache: {cache_path.name}")
        return joblib.load(cache_path)

    logger.info(f"⚙ Computing (not cached): {cache_path.name}")
    result = compute_fn(*args, **kwargs)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, cache_path)
    logger.info(f"💾 Saved to cache: {cache_path.name}")

    return result
