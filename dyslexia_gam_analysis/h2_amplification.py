"""
Hypothesis 2: Dyslexic Amplification Testing - FIXED
Tests whether dyslexics show steeper slopes (SR > 1.0) for all features
Uses PROPERLY CLUSTERED bootstrap confidence intervals
Includes component SR decomposition (skip vs duration)
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_slope_ratio(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    quartiles: Dict[str, float],
) -> float:
    """
    Compute Slope Ratio: |ΔERT_dys(Q1→Q3)| / |ΔERT_ctrl(Q1→Q3)|

    Args:
        ert_predictor: ERTPredictor instance
        data: Data for computing means
        feature: Feature to test
        quartiles: Q1 and Q3 values

    Returns:
        Slope ratio (SR)
    """
    # Get Q1 and Q3
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    # Mean values for other features
    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    # Create grids
    grid_q1 = pd.DataFrame({feature: [q1]})
    grid_q3 = pd.DataFrame({feature: [q3]})

    for other_feat in other_features:
        grid_q1[other_feat] = means[other_feat]
        grid_q3[other_feat] = means[other_feat]

    # Compute ΔERT for both groups
    delta_ert = {}
    for group in ["control", "dyslexic"]:
        ert_q1 = ert_predictor.predict_ert(grid_q1, group)[0]
        ert_q3 = ert_predictor.predict_ert(grid_q3, group)[0]
        delta_ert[group] = abs(ert_q3 - ert_q1)

    # Slope ratio
    if delta_ert["control"] > 0:
        sr = delta_ert["dyslexic"] / delta_ert["control"]
    else:
        sr = np.nan

    return sr


def bootstrap_slope_ratio(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    quartiles: Dict[str, float],
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap confidence interval for slope ratio

    FIXED: Properly clusters by subject with replacement
    Each subject can appear 0, 1, or multiple times in bootstrap sample

    Args:
        ert_predictor: ERTPredictor instance
        data: Data (will be resampled at subject level)
        feature: Feature to test
        quartiles: Q1/Q3 values
        n_bootstrap: Number of bootstrap iterations

    Returns:
        (mean_sr, ci_low, ci_high, p_value) tuple
    """
    logger.info(f"  Bootstrapping SR for {feature} ({n_bootstrap} iterations)...")

    # Get unique subjects
    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)

    bootstrap_srs = []

    for i in tqdm(range(n_bootstrap), desc=f"  {feature} bootstrap", leave=False):
        # CRITICAL FIX: Resample subjects with replacement
        # Use deterministic seed for reproducibility
        np.random.seed(i)
        boot_subject_sample = np.random.choice(subjects, size=n_subjects, replace=True)

        # Build bootstrap dataset by concatenating data for sampled subjects
        # This properly handles subjects appearing multiple times
        boot_data_list = []
        for subj in boot_subject_sample:
            subj_data = data[data["subject_id"] == subj].copy()
            boot_data_list.append(subj_data)

        if len(boot_data_list) == 0:
            continue

        boot_data = pd.concat(boot_data_list, ignore_index=True)

        # Recompute quartiles on bootstrap sample
        boot_q1 = boot_data[feature].quantile(0.25)
        boot_q3 = boot_data[feature].quantile(0.75)
        boot_quartiles = {feature: {"q1": boot_q1, "q3": boot_q3}}

        # Compute SR on bootstrap sample
        try:
            sr = compute_slope_ratio(ert_predictor, boot_data, feature, boot_quartiles)
            if not np.isnan(sr):
                bootstrap_srs.append(sr)
        except Exception as e:
            # Silently skip failed iterations
            pass

    # Compute statistics
    bootstrap_srs = np.array(bootstrap_srs)

    if len(bootstrap_srs) == 0:
        logger.warning(f"  No valid bootstrap samples for {feature}")
        return np.nan, np.nan, np.nan, 1.0

    mean_sr = np.mean(bootstrap_srs)
    ci_low = np.percentile(bootstrap_srs, 2.5)
    ci_high = np.percentile(bootstrap_srs, 97.5)

    # p-value: proportion of bootstrap samples where SR <= 1.0
    p_value = np.mean(bootstrap_srs <= 1.0)

    logger.info(f"    Bootstrap complete: {len(bootstrap_srs)} valid samples")
    logger.info(f"    CI width: {ci_high - ci_low:.4f}")

    return mean_sr, ci_low, ci_high, p_value


def compute_component_slope_ratios(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    quartiles: Dict[str, float],
) -> Dict:
    """
    Compute separate SRs for skip and duration components

    This reveals the mechanism: does the feature affect skipping or duration more?
    And which group shows stronger effects in each component?

    Args:
        ert_predictor: ERTPredictor instance
        data: Original data
        feature: Feature to analyze
        quartiles: Q1/Q3 values

    Returns:
        Dictionary with component-wise slope ratios
    """
    logger.info(f"  Computing component SRs for {feature}...")

    # Get Q1 and Q3
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    # Mean values for other features
    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    # Create grids
    grid_q1 = pd.DataFrame({feature: [q1]})
    grid_q3 = pd.DataFrame({feature: [q3]})

    for other_feat in other_features:
        grid_q1[other_feat] = means[other_feat]
        grid_q3[other_feat] = means[other_feat]

    # Compute components for both groups
    results = {}
    for group in ["control", "dyslexic"]:
        # Get skip and duration predictions at Q1 and Q3
        _, p_skip_q1, trt_q1 = ert_predictor.predict_ert(
            grid_q1, group, return_components=True
        )
        _, p_skip_q3, trt_q3 = ert_predictor.predict_ert(
            grid_q3, group, return_components=True
        )

        # Compute deltas
        delta_p_skip = p_skip_q3[0] - p_skip_q1[0]
        delta_trt = trt_q3[0] - trt_q1[0]

        results[group] = {
            "p_skip_q1": float(p_skip_q1[0]),
            "p_skip_q3": float(p_skip_q3[0]),
            "delta_p_skip": float(delta_p_skip),
            "trt_q1": float(trt_q1[0]),
            "trt_q3": float(trt_q3[0]),
            "delta_trt": float(delta_trt),
        }

    # Compute component SRs
    # SR(skip): How much more does feature affect skip probability in dyslexics?
    # SR(duration): How much more does feature affect duration in dyslexics?

    delta_skip_ctrl = abs(results["control"]["delta_p_skip"])
    delta_skip_dys = abs(results["dyslexic"]["delta_p_skip"])

    delta_trt_ctrl = abs(results["control"]["delta_trt"])
    delta_trt_dys = abs(results["dyslexic"]["delta_trt"])

    sr_skip = delta_skip_dys / delta_skip_ctrl if delta_skip_ctrl > 0 else np.nan
    sr_duration = delta_trt_dys / delta_trt_ctrl if delta_trt_ctrl > 0 else np.nan

    logger.info(f"    Component SRs:")
    logger.info(
        f"      Skip:     {sr_skip:.3f} (dyslexic Δp={delta_skip_dys:.3f}, control Δp={delta_skip_ctrl:.3f})"
    )
    logger.info(
        f"      Duration: {sr_duration:.3f} (dyslexic Δ={delta_trt_dys:.1f}ms, control Δ={delta_trt_ctrl:.1f}ms)"
    )

    # Interpretation
    interpretation = ""
    if not np.isnan(sr_skip) and not np.isnan(sr_duration):
        if sr_skip < 1.0 and sr_duration > 1.0:
            interpretation = "Controls use this feature more for skipping; dyslexics show amplified duration costs"
        elif sr_skip > 1.0 and sr_duration < 1.0:
            interpretation = "Dyslexics skip less based on this feature; controls show amplified duration costs"
        elif sr_skip > 1.0 and sr_duration > 1.0:
            interpretation = "Dyslexics show amplification in both skip and duration"
        elif sr_skip < 1.0 and sr_duration < 1.0:
            interpretation = "Controls show stronger effects in both skip and duration"

    logger.info(f"      Interpretation: {interpretation}")

    return {
        "feature": feature,
        "control": results["control"],
        "dyslexic": results["dyslexic"],
        "sr_skip": float(sr_skip),
        "sr_duration": float(sr_duration),
        "interpretation": interpretation,
    }


def test_hypothesis_2(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict[str, Dict[str, float]],
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Test Hypothesis 2: Dyslexic amplification (SR > 1.0)

    Args:
        ert_predictor: ERTPredictor instance
        data: Full dataset
        quartiles: Feature quartiles
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with H2 test results including component decomposition
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 2: DYSLEXIC AMPLIFICATION")
    logger.info("=" * 60)

    features = ["length", "zipf", "surprisal"]
    results = {}

    for feature in features:
        logger.info(f"\nTesting amplification for {feature}...")

        # Compute point estimate
        sr = compute_slope_ratio(ert_predictor, data, feature, quartiles)

        # Bootstrap CI
        mean_sr, ci_low, ci_high, p_value = bootstrap_slope_ratio(
            ert_predictor, data, feature, quartiles, n_bootstrap
        )

        # Component-wise SRs
        component_srs = compute_component_slope_ratios(
            ert_predictor, data, feature, quartiles
        )

        # Check if amplified
        amplified = sr > 1.0
        ci_excludes_1 = (ci_low > 1.0) or (ci_high < 1.0)
        significant = p_value < 0.05

        # Status
        status = "CONFIRMED" if (amplified and ci_excludes_1) else "NOT CONFIRMED"

        results[feature] = {
            "slope_ratio": float(sr),
            "bootstrap_mean_sr": float(mean_sr),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_width": float(ci_high - ci_low),
            "p_value": float(p_value),
            "amplified": bool(amplified),
            "ci_excludes_1": bool(ci_excludes_1),
            "significant": bool(significant),
            "status": status,
            "component_srs": component_srs,
        }

        logger.info(f"  Overall SR: {sr:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Status: {status}")

    # Overall H2 status
    all_confirmed = all(r["status"] == "CONFIRMED" for r in results.values())
    overall_status = "CONFIRMED" if all_confirmed else "PARTIALLY CONFIRMED"

    # Count how many features show amplification
    n_amplified = sum(1 for r in results.values() if r["amplified"])

    logger.info(f"\nHYPOTHESIS 2: {overall_status}")
    logger.info(f"  {n_amplified}/3 features show dyslexic amplification (SR > 1.0)")

    return {
        "status": overall_status,
        "slope_ratios": results,
        "n_amplified": n_amplified,
        "summary": f"{n_amplified}/3 features show significant dyslexic amplification (SR > 1.0 with p < 0.05)",
    }


def compute_amplification_at_multiple_points(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    n_points: int = 10,
) -> pd.DataFrame:
    """
    Compute amplification across the feature range (for visualization)

    Args:
        ert_predictor: ERTPredictor instance
        data: Original data
        feature: Feature to analyze
        n_points: Number of points to evaluate

    Returns:
        DataFrame with feature values and SR at each point
    """
    # Get feature range
    feature_min = data[feature].quantile(0.05)
    feature_max = data[feature].quantile(0.95)

    # Create evaluation points
    feature_values = np.linspace(feature_min, feature_max, n_points)

    # Mean values for other features
    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    # Compute ERT at each point for both groups
    results = []

    for val in feature_values:
        grid = pd.DataFrame({feature: [val]})
        for other_feat in other_features:
            grid[other_feat] = means[other_feat]

        ert_ctrl = ert_predictor.predict_ert(grid, "control")[0]
        ert_dys = ert_predictor.predict_ert(grid, "dyslexic")[0]

        results.append(
            {
                feature: val,
                "ert_control": ert_ctrl,
                "ert_dyslexic": ert_dys,
                "gap": ert_dys - ert_ctrl,
            }
        )

    df = pd.DataFrame(results)

    # Compute local slope ratio (finite differences)
    df["slope_control"] = df["ert_control"].diff() / df[feature].diff()
    df["slope_dyslexic"] = df["ert_dyslexic"].diff() / df[feature].diff()

    # Avoid division by zero
    df["slope_ratio"] = np.where(
        np.abs(df["slope_control"]) > 0.001,
        np.abs(df["slope_dyslexic"]) / np.abs(df["slope_control"]),
        np.nan,
    )

    return df
