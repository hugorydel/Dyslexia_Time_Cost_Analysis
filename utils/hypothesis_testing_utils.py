# utils/hypothesis_testing_utils.py
"""
Data preparation utilities for hypothesis testing
Handles ERT creation, scaling, quartiles, and collinearity checks
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def prepare_hypothesis_testing_data(data: pd.DataFrame) -> tuple:
    """
    Prepare data for hypothesis testing

    Returns:
        Tuple of (prepared_data, quartiles, scalers, vif_results)
    """
    logger.info("Preparing data for hypothesis testing...")

    data = data.copy()

    # 1. Create ERT (Expected Reading Time)
    data = create_ert(data)

    # 2. Center and scale predictors
    data, scalers = scale_predictors(data)

    # 3. Check collinearity
    vif_results = check_collinearity(data)

    # 4. Create residualized surprisal for sensitivity analysis
    data = create_residualized_surprisal(data)

    # 5. Define pooled quartiles
    quartiles = define_pooled_quartiles(data)

    logger.info(f"Data preparation complete: {len(data):,} words ready for analysis")

    return data, quartiles, scalers, vif_results


def create_ert(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create ERT (Expected Reading Time) variable
    ERT = 0 if skipped, else TRT
    """
    logger.info("Creating ERT variable...")

    data["ERT"] = np.where(data["was_fixated"], data["total_reading_time"], 0)

    skipped_count = (data["ERT"] == 0).sum()
    fixated_count = (data["ERT"] > 0).sum()

    logger.info(
        f"  ERT created: {skipped_count:,} skipped (0ms), {fixated_count:,} fixated"
    )

    return data


def scale_predictors(data: pd.DataFrame) -> tuple:
    """
    Center and scale predictors (length, frequency, surprisal)

    Returns:
        Tuple of (data_with_scaled, scalers_dict)
    """
    logger.info("Centering and scaling predictors...")

    features = ["word_length", "word_frequency_zipf", "surprisal"]
    scalers = {}

    for feature in features:
        if feature not in data.columns:
            logger.warning(f"Feature {feature} not found, skipping")
            continue

        # Remove NaN for fitting
        valid_mask = data[feature].notna()

        scaler = StandardScaler()
        data.loc[valid_mask, f"{feature}_scaled"] = scaler.fit_transform(
            data.loc[valid_mask, [feature]]
        )

        scalers[feature] = {
            "mean": float(scaler.mean_[0]),
            "scale": float(scaler.scale_[0]),
        }

        logger.info(
            f"  {feature}: mean={scalers[feature]['mean']:.2f}, "
            f"scale={scalers[feature]['scale']:.2f}"
        )

    return data, scalers


def check_collinearity(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check for collinearity among predictors using VIF

    Returns:
        DataFrame with VIF values
    """
    logger.info("Checking collinearity (VIF)...")

    features = ["word_length_scaled", "word_frequency_zipf_scaled", "surprisal_scaled"]

    # Only use complete cases
    vif_data = data[features].dropna()

    if len(vif_data) == 0:
        logger.warning("No complete cases for VIF calculation")
        return pd.DataFrame()

    vif_results = pd.DataFrame(
        {
            "Feature": ["Length", "Frequency", "Surprisal"],
            "VIF": [variance_inflation_factor(vif_data.values, i) for i in range(3)],
        }
    )

    logger.info("VIF Results:")
    for _, row in vif_results.iterrows():
        vif_val = row["VIF"]
        status = "OK" if vif_val < 5 else ("Moderate" if vif_val < 10 else "Severe")
        logger.info(f"  {row['Feature']}: {vif_val:.2f} ({status})")

    return vif_results


def create_residualized_surprisal(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create residualized surprisal (orthogonalized w.r.t. frequency)
    For sensitivity analysis
    """
    logger.info("Creating residualized surprisal...")

    from scipy.stats import linregress

    mask = data["surprisal_scaled"].notna() & data["word_frequency_zipf_scaled"].notna()

    if mask.sum() < 10:
        logger.warning("Insufficient data for residualization")
        data["surprisal_resid"] = data["surprisal_scaled"]
        return data

    slope, intercept, r_value, _, _ = linregress(
        data.loc[mask, "word_frequency_zipf_scaled"], data.loc[mask, "surprisal_scaled"]
    )

    data["surprisal_resid"] = data["surprisal_scaled"] - (
        slope * data["word_frequency_zipf_scaled"] + intercept
    )

    logger.info(f"  Correlation (Frequency-Surprisal): r={r_value:.3f}")
    logger.info(f"  Residualized surprisal created")

    return data


def define_pooled_quartiles(data: pd.DataFrame) -> dict:
    """
    Define quartile cutpoints on pooled data (not per-group)

    Returns:
        Dictionary with Q1 and Q3 values for each feature
    """
    logger.info("Defining pooled quartiles...")

    features = ["word_length", "word_frequency_zipf", "surprisal"]
    quartiles = {}

    for feature in features:
        if feature not in data.columns:
            continue

        q1, q3 = data[feature].quantile([0.25, 0.75])
        quartiles[feature] = {"Q1": float(q1), "Q3": float(q3)}

        # Create categorical variable
        data[f"{feature}_quartile"] = pd.cut(
            data[feature], bins=[-np.inf, q1, q3, np.inf], labels=["Q1", "Q2-Q3", "Q4"]
        )

        logger.info(f"  {feature}: Q1={q1:.2f}, Q3={q3:.2f}")

    return quartiles
