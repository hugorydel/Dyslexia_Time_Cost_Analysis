"""
Data Preparation with Optional Orthogonalized Zipf
Supports both raw zipf and residualized zipf modes
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def orthogonalize_zipf(data: pd.DataFrame) -> pd.DataFrame:
    """
    Orthogonalize zipf with respect to length

    Keep length as-is, residualize zipf to get "frequency beyond length"

    Args:
        data: DataFrame with 'length' and 'zipf' columns

    Returns:
        DataFrame with added 'zipf_resid' column and renamed 'zipf_original'
    """
    logger.info("Orthogonalizing zipf with respect to length...")

    # Store original zipf values
    data["zipf_original"] = data["zipf"].copy()

    # Remove rows with missing values for regression
    valid_mask = data[["length", "zipf"]].notna().all(axis=1)
    valid_data = data[valid_mask].copy()

    # Fit linear regression: zipf ~ length
    X = valid_data[["length"]].values
    y = valid_data["zipf"].values

    reg = LinearRegression()
    reg.fit(X, y)

    # Compute residuals for all valid rows
    y_pred = reg.predict(X)
    residuals = y - y_pred

    # Store residuals
    data.loc[valid_mask, "zipf_resid"] = residuals

    # For any invalid rows, set to NaN (will be dropped later)
    data.loc[~valid_mask, "zipf_resid"] = np.nan

    # Log diagnostics
    corr_original = data["length"].corr(data["zipf_original"])
    corr_resid = data["length"].corr(data["zipf_resid"])

    logger.info(f"  Original correlation (length, zipf): {corr_original:.3f}")
    logger.info(f"  Residual correlation (length, zipf_resid): {corr_resid:.3f}")
    logger.info(f"  R² from regression: {reg.score(X, y):.3f}")
    logger.info(
        f"  Regression coefficients: intercept={reg.intercept_:.3f}, slope={reg.coef_[0]:.3f}"
    )

    # Replace zipf with zipf_resid for modeling
    data["zipf"] = data["zipf_resid"].copy()

    logger.info("  Zipf successfully orthogonalized!")
    logger.info("  NOTE: 'zipf' now contains residualized values")
    logger.info("  NOTE: 'zipf_original' contains original values")

    return data


def prepare_gam_data(
    data: pd.DataFrame, residualize_zipf: bool = False
) -> pd.DataFrame:
    """
    Prepare data for GAM fitting

    Args:
        data: Raw data with linguistic features
        residualize_zipf: If True, orthogonalize zipf against length
                         If False, use raw zipf values

    Returns:
        Cleaned DataFrame ready for GAM fitting
    """
    logger.info("=" * 60)
    logger.info("PREPARING DATA FOR GAM ANALYSIS")
    if residualize_zipf:
        logger.info("MODE: Residualized Zipf (controlling for length)")
    else:
        logger.info("MODE: Raw Zipf (natural co-occurrence)")
    logger.info("=" * 60)

    # Check required columns exist
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

    # Rename columns to standard names
    gam_data = gam_data.rename(
        columns={
            "word_length": "length",
            "word_frequency_zipf": "zipf",
            "total_reading_time": "TRT",
        }
    )

    # Create skip indicator (0 = fixated, 1 = skipped)
    gam_data["skip"] = (gam_data["n_fixations"] == 0).astype(int)

    # Create group columns
    gam_data["group"] = gam_data["dyslexic"].map({False: "control", True: "dyslexic"})
    gam_data["group_numeric"] = gam_data["dyslexic"].astype(int)

    # Remove rows with missing values in key features (BEFORE orthogonalization)
    before_clean = len(gam_data)
    gam_data = gam_data.dropna(subset=["length", "zipf", "surprisal"])
    after_clean = len(gam_data)

    logger.info(f"Dropped {before_clean - after_clean:,} rows with missing features")

    # CONDITIONAL ORTHOGONALIZATION
    if residualize_zipf:
        logger.info("\nApplying zipf orthogonalization...")
        gam_data = orthogonalize_zipf(gam_data)

        # Drop any rows where orthogonalization created NaN
        before_ortho = len(gam_data)
        gam_data = gam_data.dropna(subset=["zipf"])
        after_ortho = len(gam_data)
        if after_ortho < before_ortho:
            logger.info(
                f"Dropped {before_ortho - after_ortho:,} rows after orthogonalization"
            )
    else:
        logger.info("\nUsing raw zipf values (no orthogonalization)")
        # Store original for reference
        gam_data["zipf_original"] = gam_data["zipf"].copy()

        # Log correlation
        corr_length_zipf = gam_data["length"].corr(gam_data["zipf"])
        logger.info(f"  Correlation (length, zipf): {corr_length_zipf:.3f}")
        if abs(corr_length_zipf) > 0.7:
            logger.warning(f"  ⚠️  High correlation detected (r={corr_length_zipf:.3f})")
            logger.warning(
                f"  GAMs can handle this, but be aware of potential collinearity"
            )

    # Remove extreme outliers in TRT (beyond 3 SD for fixated words)
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

    # Compute ERT for each observation
    gam_data["ERT"] = gam_data.apply(
        lambda row: 0 if row["skip"] == 1 else row["TRT"], axis=1
    )

    # Log summary
    logger.info(f"\nPrepared data summary:")
    logger.info(f"  Total observations: {len(gam_data):,}")
    logger.info(f"  Fixated words: {(gam_data['skip']==0).sum():,}")
    logger.info(f"  Skipped words: {(gam_data['skip']==1).sum():,}")
    logger.info(f"  Control observations: {(gam_data['group']=='control').sum():,}")
    logger.info(f"  Dyslexic observations: {(gam_data['group']=='dyslexic').sum():,}")
    logger.info(f"  Unique subjects: {gam_data['subject_id'].nunique()}")
    logger.info(f"  Unique words: {gam_data['word_text'].nunique()}")

    # Feature ranges
    logger.info(f"\nFeature ranges:")
    logger.info(
        f"  length: [{gam_data['length'].min():.2f}, {gam_data['length'].max():.2f}], "
        f"mean={gam_data['length'].mean():.2f}"
    )

    zipf_label = "zipf (residualized)" if residualize_zipf else "zipf (raw)"
    logger.info(
        f"  {zipf_label}: [{gam_data['zipf'].min():.2f}, {gam_data['zipf'].max():.2f}], "
        f"mean={gam_data['zipf'].mean():.2f}"
    )

    logger.info(
        f"  surprisal: [{gam_data['surprisal'].min():.2f}, {gam_data['surprisal'].max():.2f}], "
        f"mean={gam_data['surprisal'].mean():.2f}"
    )

    return gam_data


def prepare_data_pipeline(
    data: pd.DataFrame,
    residualize_zipf: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Full data preparation pipeline

    Args:
        data: Raw data with linguistic features
        residualize_zipf: Whether to orthogonalize zipf against length

    Returns:
        (train_data, test_data, metadata) tuple
    """
    # Prepare data (with optional orthogonalization)
    gam_data = prepare_gam_data(data, residualize_zipf=residualize_zipf)

    # Compute quartiles
    quartiles = compute_feature_quartiles(gam_data)

    # Check balance
    balance_stats = check_data_balance(gam_data)

    # Train/test split
    train_data, test_data = train_test_split_by_subject(gam_data, test_size=0.2)

    # Package metadata
    metadata = {
        "quartiles": quartiles,
        "balance_stats": balance_stats,
        "n_total": len(gam_data),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "residualized": residualize_zipf,
        "note": (
            "zipf values are RESIDUALIZED (orthogonal to length)"
            if residualize_zipf
            else "zipf values are RAW (natural co-occurrence with length)"
        ),
    }

    return train_data, test_data, metadata


def compute_feature_quartiles(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute Q1 (25th) and Q3 (75th) percentiles for each feature

    Args:
        data: Prepared data

    Returns:
        Dictionary with quartile values
    """
    features = ["length", "zipf", "surprisal"]
    quartiles = {}

    for feat in features:
        q1 = data[feat].quantile(0.25)
        q3 = data[feat].quantile(0.75)
        iqr = q3 - q1

        quartiles[feat] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
        }

    logger.info("\nFeature quartiles:")
    for feat, vals in quartiles.items():
        logger.info(
            f"  {feat}: Q1={vals['q1']:.2f}, Q3={vals['q3']:.2f}, IQR={vals['iqr']:.2f}"
        )

    return quartiles


def train_test_split_by_subject(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test by subjects

    Args:
        data: Full dataset
        test_size: Proportion of subjects in test set
        random_state: Random seed

    Returns:
        (train_data, test_data) tuple
    """
    # Get unique subjects
    subjects = data["subject_id"].unique()
    n_test = int(len(subjects) * test_size)

    # Randomly select test subjects
    np.random.seed(random_state)
    test_subjects = np.random.choice(subjects, size=n_test, replace=False)

    # Split
    train_data = data[~data["subject_id"].isin(test_subjects)].copy()
    test_data = data[data["subject_id"].isin(test_subjects)].copy()

    logger.info(f"\nTrain/test split:")
    logger.info(
        f"  Train: {len(train_data):,} observations, {train_data['subject_id'].nunique()} subjects"
    )
    logger.info(
        f"  Test: {len(test_data):,} observations, {test_data['subject_id'].nunique()} subjects"
    )

    return train_data, test_data


def check_data_balance(data: pd.DataFrame) -> Dict:
    """
    Check balance of groups and features

    Args:
        data: Prepared data

    Returns:
        Dictionary with balance statistics
    """
    stats = {}

    # Group balance
    group_counts = data["group"].value_counts()
    stats["group_balance"] = {
        "control": int(group_counts.get("control", 0)),
        "dyslexic": int(group_counts.get("dyslexic", 0)),
        "ratio": float(
            group_counts.get("dyslexic", 0) / group_counts.get("control", 1)
        ),
    }

    # Skip rate by group
    skip_by_group = data.groupby("group")["skip"].mean()
    stats["skip_rates"] = {
        "control": float(skip_by_group.get("control", 0)),
        "dyslexic": float(skip_by_group.get("dyslexic", 0)),
    }

    # Mean TRT by group (fixated words only)
    fixated = data[data["skip"] == 0]
    if len(fixated) > 0:
        trt_by_group = fixated.groupby("group")["TRT"].mean()
        stats["mean_trt"] = {
            "control": float(trt_by_group.get("control", 0)),
            "dyslexic": float(trt_by_group.get("dyslexic", 0)),
        }
    else:
        stats["mean_trt"] = {"control": 0, "dyslexic": 0}

    # Feature distributions by group
    stats["feature_means"] = {}
    for feat in ["length", "zipf", "surprisal"]:
        feat_by_group = data.groupby("group")[feat].mean()
        stats["feature_means"][feat] = {
            "control": float(feat_by_group.get("control", 0)),
            "dyslexic": float(feat_by_group.get("dyslexic", 0)),
        }

    logger.info("\nData balance check:")
    logger.info(
        f"  Group sizes: Control={stats['group_balance']['control']:,}, "
        f"Dyslexic={stats['group_balance']['dyslexic']:,}"
    )
    logger.info(
        f"  Skip rates: Control={stats['skip_rates']['control']:.3f}, "
        f"Dyslexic={stats['skip_rates']['dyslexic']:.3f}"
    )
    logger.info(
        f"  Mean TRT: Control={stats['mean_trt']['control']:.1f}ms, "
        f"Dyslexic={stats['mean_trt']['dyslexic']:.1f}ms"
    )

    return stats


def prepare_data_pipeline(
    data: pd.DataFrame,
    residualize_zipf: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Full data preparation pipeline

    Args:
        data: Raw data with linguistic features
        residualize_zipf: Whether to orthogonalize zipf against length

    Returns:
        (train_data, test_data, metadata) tuple
    """
    # Prepare data (with optional orthogonalization)
    gam_data = prepare_gam_data(data, residualize_zipf=residualize_zipf)

    # Compute quartiles
    quartiles = compute_feature_quartiles(gam_data)

    # Check balance
    balance_stats = check_data_balance(gam_data)

    # Train/test split
    train_data, test_data = train_test_split_by_subject(gam_data, test_size=0.2)

    # Package metadata
    metadata = {
        "quartiles": quartiles,
        "balance_stats": balance_stats,
        "n_total": len(gam_data),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "residualized": residualize_zipf,
        "note": (
            "zipf values are RESIDUALIZED (orthogonal to length)"
            if residualize_zipf
            else "zipf values are RAW (natural co-occurrence with length)"
        ),
    }

    return train_data, test_data, metadata
