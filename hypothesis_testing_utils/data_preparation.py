"""
Data Preparation
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_pooled_length_bins(
    data: pd.DataFrame, n_bins: int = 7
) -> Tuple[np.ndarray, pd.Series]:
    """
    Create length bins on POOLED sample (both groups combined)

    CRITICAL: This is computed ONCE and used for all conditional analyses

    Args:
        data: Full dataset (control + dyslexic combined)
        n_bins: Number of bins (default 5)

    Returns:
        (bin_edges, bin_weights) tuple
    """
    logger.info(f"Creating pooled length bins (n={n_bins})...")

    # Use qcut to get equal-frequency bins
    _, bin_edges = pd.qcut(
        data["length"], q=n_bins, labels=False, duplicates="drop", retbins=True
    )

    # Compute bin assignments on pooled data
    data["length_bin"] = pd.cut(
        data["length"], bins=bin_edges, labels=False, include_lowest=True
    )

    # Compute pooled bin weights (proportion in each bin)
    pooled_weights = data["length_bin"].value_counts(normalize=True).sort_index()

    logger.info(f"  Bin edges: {bin_edges}")
    logger.info(f"  Bin weights: {pooled_weights.values}")

    # Check balance
    for i, weight in enumerate(pooled_weights):
        count = (data["length_bin"] == i).sum()
        logger.info(f"    Bin {i}: {count:,} obs ({weight:.1%})")

    return bin_edges, pooled_weights


def check_feature_orientations(data: pd.DataFrame, quartiles: Dict) -> Dict:
    """
    Verify that Q1 < Q3 in the correct direction for each feature

    Expected orientations:
    - length: Q1 (shorter) < Q3 (longer) ✓
    - zipf: Q1 (rarer) < Q3 (more frequent) ✓
    - surprisal: Q1 (predictable) < Q3 (surprising) ✓

    Args:
        data: Prepared data
        quartiles: Computed quartiles

    Returns:
        Dictionary with orientation checks
    """
    logger.info("\nChecking feature orientations...")

    checks = {}

    for feature in ["length", "zipf", "surprisal"]:
        q1 = quartiles[feature]["q1"]
        q3 = quartiles[feature]["q3"]

        correct = q1 < q3

        # Describe semantics
        if feature == "length":
            semantic = f"Q1={q1:.1f} (shorter) → Q3={q3:.1f} (longer)"
        elif feature == "zipf":
            semantic = f"Q1={q1:.1f} (rarer) → Q3={q3:.1f} (more frequent)"
        else:  # surprisal
            semantic = f"Q1={q1:.1f} (predictable) → Q3={q3:.1f} (surprising)"

        checks[feature] = {
            "q1": q1,
            "q3": q3,
            "correct_direction": correct,
            "semantic": semantic,
        }

        status = "✓" if correct else "✗"
        logger.info(f"  {feature}: {semantic} {status}")

        if not correct:
            logger.warning(f"  ⚠️  {feature} orientation is REVERSED!")

    return checks


def prepare_gam_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for GAM fitting (NO orthogonalization)

    Args:
        data: Raw data with linguistic features

    Returns:
        Cleaned DataFrame ready for GAM fitting
    """
    logger.info("=" * 60)
    logger.info("PREPARING DATA FOR GAM ANALYSIS")
    logger.info("MODE: Raw Zipf (natural correlation with length)")
    logger.info("=" * 60)

    # Check required columns
    required_cols = [
        "word_length",
        "word_frequency_zipf",
        "surprisal",
        "dyslexic",
        "subject_id",
        "word_text",
        "total_reading_time",
        "n_fixations",
    ]

    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create working copy
    gam_data = data.copy()

    # Rename to standard names
    gam_data = gam_data.rename(
        columns={
            "word_length": "length",
            "word_frequency_zipf": "zipf",
            "total_reading_time": "TRT",
        }
    )

    # Create derived columns
    gam_data["skip"] = (gam_data["n_fixations"] == 0).astype(int)
    gam_data["group"] = gam_data["dyslexic"].map({False: "control", True: "dyslexic"})
    gam_data["group_numeric"] = gam_data["dyslexic"].astype(int)

    # Remove missing values
    before_clean = len(gam_data)
    gam_data = gam_data.dropna(subset=["length", "zipf", "surprisal"])
    after_clean = len(gam_data)

    logger.info(f"Dropped {before_clean - after_clean:,} rows with missing features")

    # Log correlation (should be high for raw zipf)
    corr_length_zipf = gam_data["length"].corr(gam_data["zipf"])
    logger.info(f"\nCorrelation (length, zipf): {corr_length_zipf:.3f}")

    # Remove TRT outliers (fixated words only)
    fixated = gam_data[gam_data["skip"] == 0]
    if len(fixated) > 0:
        trt_mean = fixated["TRT"].mean()
        trt_std = fixated["TRT"].std()
        trt_upper = trt_mean + 3 * trt_std

        before_outlier = len(gam_data)
        gam_data = gam_data[(gam_data["skip"] == 1) | (gam_data["TRT"] <= trt_upper)]
        after_outlier = len(gam_data)

        logger.info(
            f"Removed {before_outlier - after_outlier:,} extreme TRT outliers (>3 SD)"
        )

    # Compute ERT
    gam_data["ERT"] = gam_data.apply(
        lambda row: 0 if row["skip"] == 1 else row["TRT"], axis=1
    )

    # Summary
    logger.info(f"\nPrepared data summary:")
    logger.info(f"  Total observations: {len(gam_data):,}")
    logger.info(f"  Control: {(gam_data['group']=='control').sum():,}")
    logger.info(f"  Dyslexic: {(gam_data['group']=='dyslexic').sum():,}")
    logger.info(f"  Unique subjects: {gam_data['subject_id'].nunique()}")

    # Feature ranges
    logger.info(f"\nFeature ranges:")
    for feat in ["length", "zipf", "surprisal"]:
        logger.info(
            f"  {feat}: [{gam_data[feat].min():.2f}, {gam_data[feat].max():.2f}], "
            f"mean={gam_data[feat].mean():.2f}"
        )

    return gam_data


def compute_feature_quartiles(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute Q1, Q3, IQR for each feature on POOLED data"""

    features = ["length", "zipf", "surprisal"]
    quartiles = {}

    for feat in features:
        q1 = data[feat].quantile(0.25)
        q3 = data[feat].quantile(0.75)
        iqr = q3 - q1

        quartiles[feat] = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr)}

    logger.info("\nFeature quartiles (pooled):")
    for feat, vals in quartiles.items():
        logger.info(
            f"  {feat}: Q1={vals['q1']:.2f}, Q3={vals['q3']:.2f}, IQR={vals['iqr']:.2f}"
        )

    return quartiles


def prepare_data_pipeline(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete data preparation pipeline

    Returns:
        (prepared_data, metadata) tuple where metadata includes:
        - quartiles
        - pooled_bin_edges
        - pooled_bin_weights
        - orientation_checks
    """
    # Prepare data (raw zipf, no orthogonalization)
    gam_data = prepare_gam_data(data)

    # Compute quartiles on POOLED data
    quartiles = compute_feature_quartiles(gam_data)

    # Check orientations
    orientation_checks = check_feature_orientations(gam_data, quartiles)

    # Create pooled length bins
    bin_edges, bin_weights = create_pooled_length_bins(gam_data, n_bins=7)

    # Apply bins to data
    gam_data["length_bin"] = pd.cut(
        gam_data["length"], bins=bin_edges, labels=False, include_lowest=True
    )

    # Package metadata
    metadata = {
        "quartiles": quartiles,
        "pooled_bin_edges": bin_edges.tolist(),
        "pooled_bin_weights": bin_weights.to_dict(),
        "orientation_checks": orientation_checks,
        "n_total": len(gam_data),
        "note": "Using RAW zipf with conditional evaluation within length bins",
    }

    return gam_data, metadata
