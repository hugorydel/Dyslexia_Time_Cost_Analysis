"""
Caching utilities
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

import joblib

logger = logging.getLogger(__name__)


def compute_data_hash(data) -> str:
    """Compute hash of dataset for cache invalidation"""
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            hash_input = (
                f"{data.shape}_{data.columns.tolist()}_"
                f"{data.head(5).values.tobytes()}_{data.tail(5).values.tobytes()}"
            )
            return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    except:
        pass
    return hashlib.md5(str(data).encode()).hexdigest()[:16]


def atomic_joblib_dump(obj: Any, path: Path, compress: int = 3):
    """
    Atomically write joblib file to prevent corruption

    Writes to temporary file first, then replaces target
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(obj, tmp, compress=compress)
        tmp.replace(path)  # Atomic on POSIX systems
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise e


def load_or_recompute(
    cache_path: Path, compute_fn: Callable, reuse: bool = True, *args, **kwargs
) -> Any:
    """
    Load from cache or compute and save with atomic writes

    Args:
        cache_path: Path to cache file
        compute_fn: Function to call if cache miss
        reuse: If True, attempt to load from cache
        *args, **kwargs: Arguments to pass to compute_fn

    Returns:
        Computed or cached result
    """
    if reuse and cache_path.exists():
        try:
            logger.info(f"✓ Loading from cache: {cache_path.name}")
            return joblib.load(cache_path)
        except Exception as e:
            logger.warning(f"⚠️ Cache load failed ({e}), recomputing...")
            # Delete corrupt cache file
            try:
                cache_path.unlink()
            except:
                pass

    logger.info(f"⚙️ Computing (not cached): {cache_path.name}")
    result = compute_fn(*args, **kwargs)

    # Save to cache with atomic write
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_joblib_dump(result, cache_path)
        logger.info(f"💾 Saved to cache: {cache_path.name}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save cache: {e}")
        # Continue even if caching fails

    return result


def safe_cache_load(cache_path: Path, default=None) -> Any:
    """
    Safely load from cache with fallback

    Returns default if cache doesn't exist or is corrupt
    """
    if not cache_path.exists():
        return default

    try:
        return joblib.load(cache_path)
    except Exception as e:
        logger.warning(f"⚠️ Corrupt cache file {cache_path.name}: {e}")
        # Try to delete corrupt file
        try:
            cache_path.unlink()
        except:
            pass
        return default


def clear_cache_directory(cache_dir: Path, pattern: str = "*.pkl"):
    """
    Clear all cache files matching pattern

    Args:
        cache_dir: Directory containing cache files
        pattern: Glob pattern for files to delete
    """
    if not cache_dir.exists():
        return

    deleted = 0
    for cache_file in cache_dir.glob(pattern):
        try:
            cache_file.unlink()
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to delete {cache_file.name}: {e}")

    if deleted > 0:
        logger.info(f"Cleared {deleted} cache files from {cache_dir}")
