"""
Hypothesis 1: Feature Effects with CONDITIONAL Zipf Evaluation
Key changes:
- Added conditional AMIE for zipf (within length bins)
- Added pathway decomposition (skip, duration, ERT) for ALL features
- Added Cohen's h for skip pathway
- Uses pooled bins and weights
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cohens_h(p1: float, p2: float) -> float:
    """
    Compute Cohen's h effect size for two proportions

    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def compute_amie_standard(
    ert_predictor, data: pd.DataFrame, feature: str, group: str, quartiles: Dict
) -> Dict:
    """
    Standard AMIE for length and surprisal
    Q1→Q3 shift with other features at mean
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    # Mean values for other features
    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    # Grids
    grid_q1 = pd.DataFrame({feature: [q1], **means})
    grid_q3 = pd.DataFrame({feature: [q3], **means})

    # Predict
    ert_q1 = ert_predictor.predict_ert(grid_q1, group)[0]
    ert_q3 = ert_predictor.predict_ert(grid_q3, group)[0]

    amie = ert_q3 - ert_q1

    return {
        "feature": feature,
        "group": group,
        "q1": float(q1),
        "q3": float(q3),
        "ert_q1": float(ert_q1),
        "ert_q3": float(ert_q3),
        "amie_ms": float(amie),
        "method": "standard",
    }


def compute_amie_conditional_zipf(
    ert_predictor,
    data: pd.DataFrame,
    group: str,
    quartiles: Dict,
    bin_edges: np.ndarray,
    bin_weights: pd.Series,
) -> Dict:
    """
    Conditional AMIE for zipf (within length bins)

    CRITICAL: Uses pooled bins and weights to avoid composition confounds

    Steps:
    1. For each length bin, compute zipf Q1→Q3 effect
    2. Weight by POOLED bin distribution
    3. Average across bins
    """
    logger.info(f"    Computing conditional zipf AMIE for {group}...")

    group_data = data[data["group"] == group].copy()

    # Ensure bins are assigned
    if "length_bin" not in group_data.columns:
        group_data["length_bin"] = pd.cut(
            group_data["length"], bins=bin_edges, labels=False, include_lowest=True
        )

    bin_effects = []
    bin_indices_used = []

    n_bins = len(bin_weights)

    for bin_idx in range(n_bins):
        bin_data = group_data[group_data["length_bin"] == bin_idx]

        if len(bin_data) < 10:  # Skip bins with too few observations
            logger.warning(f"      Bin {bin_idx}: only {len(bin_data)} obs, skipping")
            continue

        # Compute zipf Q1→Q3 WITHIN this bin
        zipf_q1_bin = bin_data["zipf"].quantile(0.25)
        zipf_q3_bin = bin_data["zipf"].quantile(0.75)

        # Fix other features at bin means
        length_mean = bin_data["length"].mean()
        surprisal_mean = bin_data["surprisal"].mean()

        # Predict at Q1 and Q3 within bin
        grid_q1 = pd.DataFrame(
            {
                "length": [length_mean],
                "zipf": [zipf_q1_bin],
                "surprisal": [surprisal_mean],
            }
        )

        grid_q3 = pd.DataFrame(
            {
                "length": [length_mean],
                "zipf": [zipf_q3_bin],
                "surprisal": [surprisal_mean],
            }
        )

        ert_q1 = ert_predictor.predict_ert(grid_q1, group)[0]
        ert_q3 = ert_predictor.predict_ert(grid_q3, group)[0]

        effect = ert_q3 - ert_q1
        bin_effects.append(effect)
        bin_indices_used.append(bin_idx)

        logger.info(
            f"      Bin {bin_idx}: zipf Q1={zipf_q1_bin:.2f}→Q3={zipf_q3_bin:.2f}, "
            f"effect={effect:.2f}ms (n={len(bin_data)})"
        )

    if len(bin_effects) == 0:
        logger.warning(f"    No valid bins for {group}")
        return {"error": "No valid bins"}

    # Weight by POOLED distribution
    weights_used = bin_weights.iloc[bin_indices_used].values
    weights_normalized = weights_used / weights_used.sum()

    amie_conditional = np.average(bin_effects, weights=weights_normalized)

    logger.info(
        f"    Conditional AMIE ({group}): {amie_conditional:.2f}ms "
        f"(averaged over {len(bin_effects)} bins)"
    )

    return {
        "feature": "zipf",
        "group": group,
        "amie_ms": float(amie_conditional),
        "method": "conditional_within_length_bins",
        "n_bins_used": len(bin_effects),
        "bin_effects": [float(e) for e in bin_effects],
    }


def compute_pathway_effects(
    ert_predictor, data: pd.DataFrame, feature: str, group: str, quartiles: Dict
) -> Dict:
    """
    Compute effects for all three pathways: skip, duration, ERT

    Returns Q1 and Q3 values for each pathway plus deltas
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    grid_q1 = pd.DataFrame({feature: [q1], **means})
    grid_q3 = pd.DataFrame({feature: [q3], **means})

    # Get components
    ert_q1, p_skip_q1, trt_q1 = ert_predictor.predict_ert(
        grid_q1, group, return_components=True
    )
    ert_q3, p_skip_q3, trt_q3 = ert_predictor.predict_ert(
        grid_q3, group, return_components=True
    )

    # Cohen's h for skip
    h = cohens_h(p_skip_q1[0], p_skip_q3[0])

    return {
        "feature": feature,
        "group": group,
        "p_skip_q1": float(p_skip_q1[0]),
        "p_skip_q3": float(p_skip_q3[0]),
        "delta_p_skip": float(p_skip_q3[0] - p_skip_q1[0]),
        "cohens_h": float(h),
        "trt_q1": float(trt_q1[0]),
        "trt_q3": float(trt_q3[0]),
        "delta_trt_ms": float(trt_q3[0] - trt_q1[0]),
        "ert_q1": float(ert_q1[0]),
        "ert_q3": float(ert_q3[0]),
        "delta_ert_ms": float(ert_q3[0] - ert_q1[0]),
    }


def test_hypothesis_1(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict,
    bin_edges: np.ndarray,
    bin_weights: pd.Series,
) -> Dict:
    """
    Test Hypothesis 1: Feature Effects

    For each feature:
    - Compute AMIE (conditional for zipf)
    - Compute pathway decomposition (skip, duration, ERT)
    - Check expected directions
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 1: FEATURE EFFECTS")
    logger.info("=" * 60)

    features = ["length", "zipf", "surprisal"]
    results = {}

    # Expected directions
    expected_directions = {
        "length": "+",  # Longer → more time
        "zipf": "-",  # More frequent → less time
        "surprisal": "+",  # More surprising → more time
    }

    for feature in features:
        logger.info(f"\nTesting {feature}...")
        logger.info(f"  Expected direction: {expected_directions[feature]}")

        # === 1. AMIE ===
        if feature == "zipf":
            # Conditional evaluation within length bins
            amie_control = compute_amie_conditional_zipf(
                ert_predictor, data, "control", quartiles, bin_edges, bin_weights
            )
            amie_dyslexic = compute_amie_conditional_zipf(
                ert_predictor, data, "dyslexic", quartiles, bin_edges, bin_weights
            )
        else:
            # Standard evaluation
            amie_control = compute_amie_standard(
                ert_predictor, data, feature, "control", quartiles
            )
            amie_dyslexic = compute_amie_standard(
                ert_predictor, data, feature, "dyslexic", quartiles
            )

        # === 2. PATHWAY DECOMPOSITION ===
        pathway_control = compute_pathway_effects(
            ert_predictor, data, feature, "control", quartiles
        )
        pathway_dyslexic = compute_pathway_effects(
            ert_predictor, data, feature, "dyslexic", quartiles
        )

        # === 3. CHECK DIRECTION ===
        expected = expected_directions[feature]

        ctrl_amie = amie_control.get("amie_ms", np.nan)
        dys_amie = amie_dyslexic.get("amie_ms", np.nan)

        ctrl_correct = (ctrl_amie > 0) if expected == "+" else (ctrl_amie < 0)
        dys_correct = (dys_amie > 0) if expected == "+" else (dys_amie < 0)

        status = "CONFIRMED" if (ctrl_correct and dys_correct) else "NOT CONFIRMED"

        # === 4. SPECIAL NOTE FOR ZIPF ===
        note = ""
        if feature == "zipf":
            note = (
                "Zipf uses conditional evaluation within length bins to "
                "respect length-frequency coupling. Check duration pathway "
                "for amplification (may differ from combined ERT due to "
                "opposing skip effects)."
            )

        results[feature] = {
            "amie_control": amie_control,
            "amie_dyslexic": amie_dyslexic,
            "pathway_control": pathway_control,
            "pathway_dyslexic": pathway_dyslexic,
            "expected_direction": expected,
            "correct_direction": ctrl_correct and dys_correct,
            "status": status,
            "note": note,
        }

        logger.info(f"  AMIE Control: {ctrl_amie:.2f} ms")
        logger.info(f"  AMIE Dyslexic: {dys_amie:.2f} ms")
        logger.info(f"  Status: {status}")

    # Overall status
    all_confirmed = all(r["status"] == "CONFIRMED" for r in results.values())
    overall_status = "CONFIRMED" if all_confirmed else "PARTIALLY CONFIRMED"

    logger.info(f"\nHYPOTHESIS 1: {overall_status}")

    return {
        "status": overall_status,
        "features": results,
        "summary": f"{'All' if all_confirmed else 'Some'} features show expected effects",
    }
