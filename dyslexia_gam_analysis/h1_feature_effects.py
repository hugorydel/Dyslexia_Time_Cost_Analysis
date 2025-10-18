"""
Hypothesis 1: Feature Effects Testing
Tests whether length, frequency, and surprisal affect reading time
Uses AMIE, quartile contrasts, and permutation importance
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_amie(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    group: str,
    quartiles: Dict[str, float],
) -> Dict:
    """
    Compute Average Marginal Importance Effect (AMIE)
    Q1→Q3 shift in ERT with other features at mean

    Args:
        ert_predictor: ERTPredictor instance
        data: Original data (for computing means)
        feature: Feature to test ('length', 'zipf', 'surprisal')
        group: "control" or "dyslexic"
        quartiles: Dictionary with Q1 and Q3 values for each feature

    Returns:
        Dictionary with AMIE results
    """
    logger.info(f"  Computing AMIE for {feature} ({group})...")

    # Get Q1 and Q3 for this feature
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    # Get mean values for other features
    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    # Create prediction grids
    grid_q1 = pd.DataFrame({feature: [q1]})
    grid_q3 = pd.DataFrame({feature: [q3]})

    for other_feat in other_features:
        grid_q1[other_feat] = means[other_feat]
        grid_q3[other_feat] = means[other_feat]

    # Predict ERT at Q1 and Q3
    ert_q1 = ert_predictor.predict_ert(grid_q1, group)[0]
    ert_q3 = ert_predictor.predict_ert(grid_q3, group)[0]

    # AMIE = Q3 - Q1
    amie = ert_q3 - ert_q1

    return {
        "feature": feature,
        "group": group,
        "q1": float(q1),
        "q3": float(q3),
        "ert_q1": float(ert_q1),
        "ert_q3": float(ert_q3),
        "amie_ms": float(amie),
    }


def compute_quartile_contrasts(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    quartiles: Dict[str, float],
) -> Dict:
    """
    Compute discrete ΔERT for Control vs Dyslexic at Q1 and Q3

    Args:
        ert_predictor: ERTPredictor instance
        data: Original data
        feature: Feature to test
        quartiles: Quartile values

    Returns:
        Dictionary with quartile contrasts
    """
    logger.info(f"  Computing quartile contrasts for {feature}...")

    # Get Q1 and Q3
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    # Mean values for other features
    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    # Prediction grids
    grid_q1 = pd.DataFrame({feature: [q1]})
    grid_q3 = pd.DataFrame({feature: [q3]})

    for other_feat in other_features:
        grid_q1[other_feat] = means[other_feat]
        grid_q3[other_feat] = means[other_feat]

    # Predictions for both groups
    results = {}
    for group in ["control", "dyslexic"]:
        ert_q1 = ert_predictor.predict_ert(grid_q1, group)[0]
        ert_q3 = ert_predictor.predict_ert(grid_q3, group)[0]
        delta_ert = ert_q3 - ert_q1

        results[group] = {
            "ert_q1": float(ert_q1),
            "ert_q3": float(ert_q3),
            "delta_ert": float(delta_ert),
        }

    # Group difference in effect
    diff_in_diffs = results["dyslexic"]["delta_ert"] - results["control"]["delta_ert"]

    return {
        "feature": feature,
        "control": results["control"],
        "dyslexic": results["dyslexic"],
        "difference_in_differences": float(diff_in_diffs),
    }


def compute_permutation_importance(
    ert_predictor,
    test_data: pd.DataFrame,
    feature: str,
    n_permutations: int = 100,
) -> Dict:
    """
    Compute ΔQ² (permutation importance)
    Permute feature and measure drop in prediction accuracy

    Args:
        ert_predictor: ERTPredictor instance
        test_data: Test set with observed ERT values
        feature: Feature to permute
        n_permutations: Number of permutations

    Returns:
        Dictionary with permutation importance
    """
    logger.info(f"  Computing permutation importance for {feature}...")

    # Compute baseline Q²
    baseline_q2 = compute_q2(ert_predictor, test_data)

    # Permute feature and recompute Q²
    q2_permuted = []
    for i in range(n_permutations):
        # Permute the feature
        test_permuted = test_data.copy()
        test_permuted[feature] = np.random.permutation(test_permuted[feature].values)

        # Compute Q² on permuted data
        q2_perm = compute_q2(ert_predictor, test_permuted)
        q2_permuted.append(q2_perm)

    # ΔQ² = baseline - mean(permuted)
    mean_q2_permuted = np.mean(q2_permuted)
    delta_q2 = baseline_q2 - mean_q2_permuted

    return {
        "feature": feature,
        "baseline_q2": float(baseline_q2),
        "mean_q2_permuted": float(mean_q2_permuted),
        "delta_q2": float(delta_q2),
        "importance_pct": (
            float(delta_q2 / baseline_q2 * 100) if baseline_q2 > 0 else 0.0
        ),
    }


def compute_q2(ert_predictor, data: pd.DataFrame) -> float:
    """
    Compute Q² (proportion of variance explained)

    Args:
        ert_predictor: ERTPredictor instance
        data: Data with observed ERT values and group column

    Returns:
        Q² value (0-1)
    """
    if "ERT" not in data.columns or "group" not in data.columns:
        logger.warning("Missing ERT or group column, returning 0")
        return 0.0

    # Predict ERT for each observation
    predictions = []
    observed = []

    for group_name in ["control", "dyslexic"]:
        group_data = data[data["group"] == group_name]

        if len(group_data) == 0:
            continue

        try:
            pred = ert_predictor.predict_ert(group_data, group_name)
            predictions.extend(pred)
            observed.extend(group_data["ERT"].values)
        except Exception as e:
            logger.warning(f"Prediction failed for {group_name}: {e}")
            continue

    if len(predictions) == 0:
        return 0.0

    predictions = np.array(predictions)
    observed = np.array(observed)

    # Q² = 1 - SS_res / SS_tot
    ss_res = np.sum((observed - predictions) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)

    if ss_tot == 0:
        return 0.0

    q2 = 1 - (ss_res / ss_tot)

    return float(q2)


def test_hypothesis_1(
    ert_predictor,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    quartiles: Dict[str, Dict[str, float]],
) -> Dict:
    """
    Test Hypothesis 1: Feature effects on reading time

    Args:
        ert_predictor: ERTPredictor instance
        train_data: Training data (for computing means)
        test_data: Test data (for permutation importance)
        quartiles: Dictionary of quartile values for each feature

    Returns:
        Dictionary with H1 test results
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 1: FEATURE EFFECTS")
    logger.info("=" * 60)

    features = ["length", "zipf", "surprisal"]
    results = {}

    for feature in features:
        logger.info(f"\nTesting {feature}...")

        # 1. AMIE for both groups
        amie_control = compute_amie(
            ert_predictor, train_data, feature, "control", quartiles
        )
        amie_dyslexic = compute_amie(
            ert_predictor, train_data, feature, "dyslexic", quartiles
        )

        # 2. Quartile contrasts
        contrasts = compute_quartile_contrasts(
            ert_predictor, train_data, feature, quartiles
        )

        # 3. Permutation importance
        perm_importance = compute_permutation_importance(
            ert_predictor, test_data, feature, n_permutations=100
        )

        # Expected directions
        expected_directions = {
            "length": "positive",  # Longer words → more time
            "zipf": "negative",  # Higher frequency → less time
            "surprisal": "positive",  # Higher surprisal → more time
        }

        # Check if effects are in expected direction
        expected = expected_directions[feature]
        control_correct = (
            amie_control["amie_ms"] > 0
            if expected == "positive"
            else amie_control["amie_ms"] < 0
        )
        dyslexic_correct = (
            amie_dyslexic["amie_ms"] > 0
            if expected == "positive"
            else amie_dyslexic["amie_ms"] < 0
        )

        # Status
        status = (
            "CONFIRMED" if (control_correct and dyslexic_correct) else "NOT CONFIRMED"
        )

        results[feature] = {
            "amie_control": amie_control,
            "amie_dyslexic": amie_dyslexic,
            "quartile_contrasts": contrasts,
            "permutation_importance": perm_importance,
            "expected_direction": expected,
            "correct_direction": control_correct and dyslexic_correct,
            "status": status,
        }

        logger.info(f"  AMIE Control: {amie_control['amie_ms']:.2f} ms")
        logger.info(f"  AMIE Dyslexic: {amie_dyslexic['amie_ms']:.2f} ms")
        logger.info(f"  ΔQ²: {perm_importance['delta_q2']:.4f}")
        logger.info(f"  Status: {status}")

    # Overall H1 status
    all_confirmed = all(r["status"] == "CONFIRMED" for r in results.values())
    overall_status = "CONFIRMED" if all_confirmed else "PARTIALLY CONFIRMED"

    logger.info(f"\nHYPOTHESIS 1: {overall_status}")

    return {
        "status": overall_status,
        "features": results,
        "summary": f"{'All' if all_confirmed else 'Some'} features show expected effects on reading time",
    }
