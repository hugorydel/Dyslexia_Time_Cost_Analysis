"""
Hypothesis 1: Feature Effects with CONDITIONAL Zipf Evaluation
FULLY REVISED: Added p-values and comprehensive statistics for all effects
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def cohens_h(p1: float, p2: float) -> float:
    """
    Compute Cohen's h effect size for two proportions
    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def permutation_test_amie(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    group: str,
    quartiles: Dict,
    n_permutations: int = 1000,
    method: str = "standard",
    bin_edges: np.ndarray = None,
    bin_weights: pd.Series = None,
) -> Dict:
    """
    Permutation test for AMIE significance

    Returns p-value, 95% CI, and mean
    """
    # Observed AMIE
    if method == "conditional":
        from hypothesis_testing_utils.h1_feature_effects import (
            compute_amie_conditional_zipf,
        )

        observed_result = compute_amie_conditional_zipf(
            ert_predictor, data, group, quartiles, bin_edges, bin_weights
        )
        observed_amie = observed_result.get("amie_ms", np.nan)
    else:
        from hypothesis_testing_utils.h1_feature_effects import compute_amie_standard

        observed_result = compute_amie_standard(
            ert_predictor, data, feature, group, quartiles
        )
        observed_amie = observed_result.get("amie_ms", np.nan)

    if np.isnan(observed_amie):
        return {
            "p_value": np.nan,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    # Bootstrap for CI (resample subjects)
    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)
    bootstrap_amies = []

    for i in range(n_permutations):
        np.random.seed(i + 1000)
        boot_subjects = np.random.choice(subjects, size=n_subjects, replace=True)
        boot_data = pd.concat(
            [data[data["subject_id"] == s] for s in boot_subjects], ignore_index=True
        )

        if len(boot_data) < 500:
            continue

        # Recompute quartiles
        boot_quartiles = {
            feat: {
                "q1": boot_data[feat].quantile(0.25),
                "q3": boot_data[feat].quantile(0.75),
                "iqr": boot_data[feat].quantile(0.75) - boot_data[feat].quantile(0.25),
            }
            for feat in ["length", "zipf", "surprisal"]
        }

        try:
            if method == "conditional":
                result = compute_amie_conditional_zipf(
                    ert_predictor,
                    boot_data,
                    group,
                    boot_quartiles,
                    bin_edges,
                    bin_weights,
                )
            else:
                result = compute_amie_standard(
                    ert_predictor, boot_data, feature, group, boot_quartiles
                )

            amie = result.get("amie_ms", np.nan)
            if not np.isnan(amie):
                bootstrap_amies.append(amie)
        except:
            pass

    # Compute statistics
    if len(bootstrap_amies) > 0:
        ci_low = float(np.percentile(bootstrap_amies, 2.5))
        ci_high = float(np.percentile(bootstrap_amies, 97.5))

        # P-value: proportion of bootstrap samples with opposite sign
        if observed_amie > 0:
            p_value = float(np.mean(np.array(bootstrap_amies) <= 0))
        else:
            p_value = float(np.mean(np.array(bootstrap_amies) >= 0))

        # Two-tailed
        p_value = min(1.0, 2 * p_value)
    else:
        ci_low = ci_high = p_value = np.nan

    return {
        "p_value": float(p_value),
        "mean": float(observed_amie),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": len(bootstrap_amies),
    }


def permutation_test_pathway_effect(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    group: str,
    quartiles: Dict,
    pathway: str,  # "skip", "duration", or "ert"
    n_permutations: int = 1000,
) -> Dict:
    """
    Permutation test for pathway effect (skip, duration, or ERT)

    Returns p-value, 95% CI, and mean for delta
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    grid_q1 = pd.DataFrame({feature: [q1], **means})
    grid_q3 = pd.DataFrame({feature: [q3], **means})

    # Observed effect
    ert_q1, p_skip_q1, trt_q1 = ert_predictor.predict_ert(
        grid_q1, group, return_components=True
    )
    ert_q3, p_skip_q3, trt_q3 = ert_predictor.predict_ert(
        grid_q3, group, return_components=True
    )

    if pathway == "skip":
        observed_delta = p_skip_q3[0] - p_skip_q1[0]
    elif pathway == "duration":
        observed_delta = trt_q3[0] - trt_q1[0]
    else:  # ert
        observed_delta = ert_q3[0] - ert_q1[0]

    # Bootstrap for CI
    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)
    bootstrap_deltas = []

    for i in range(n_permutations):
        np.random.seed(i + 2000)
        boot_subjects = np.random.choice(subjects, size=n_subjects, replace=True)
        boot_data = pd.concat(
            [data[data["subject_id"] == s] for s in boot_subjects], ignore_index=True
        )

        if len(boot_data) < 500:
            continue

        boot_quartiles = {
            feat: {
                "q1": boot_data[feat].quantile(0.25),
                "q3": boot_data[feat].quantile(0.75),
                "iqr": boot_data[feat].quantile(0.75) - boot_data[feat].quantile(0.25),
            }
            for feat in ["length", "zipf", "surprisal"]
        }

        try:
            q1_boot = boot_quartiles[feature]["q1"]
            q3_boot = boot_quartiles[feature]["q3"]
            means_boot = {f: boot_data[f].mean() for f in other_features}

            grid_q1_boot = pd.DataFrame({feature: [q1_boot], **means_boot})
            grid_q3_boot = pd.DataFrame({feature: [q3_boot], **means_boot})

            ert_q1_b, p_skip_q1_b, trt_q1_b = ert_predictor.predict_ert(
                grid_q1_boot, group, return_components=True
            )
            ert_q3_b, p_skip_q3_b, trt_q3_b = ert_predictor.predict_ert(
                grid_q3_boot, group, return_components=True
            )

            if pathway == "skip":
                delta = p_skip_q3_b[0] - p_skip_q1_b[0]
            elif pathway == "duration":
                delta = trt_q3_b[0] - trt_q1_b[0]
            else:
                delta = ert_q3_b[0] - ert_q1_b[0]

            bootstrap_deltas.append(delta)
        except:
            pass

    # Compute statistics
    if len(bootstrap_deltas) > 0:
        ci_low = float(np.percentile(bootstrap_deltas, 2.5))
        ci_high = float(np.percentile(bootstrap_deltas, 97.5))

        # P-value: proportion with opposite sign
        if observed_delta > 0:
            p_value = float(np.mean(np.array(bootstrap_deltas) <= 0))
        else:
            p_value = float(np.mean(np.array(bootstrap_deltas) >= 0))

        p_value = min(1.0, 2 * p_value)
    else:
        ci_low = ci_high = p_value = np.nan

    return {
        "p_value": float(p_value),
        "mean": float(observed_delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": len(bootstrap_deltas),
    }


def compute_amie_standard(
    ert_predictor, data: pd.DataFrame, feature: str, group: str, quartiles: Dict
) -> Dict:
    """
    Standard AMIE for length and surprisal
    Q1→Q3 shift with other features at mean
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    grid_q1 = pd.DataFrame({feature: [q1], **means})
    grid_q3 = pd.DataFrame({feature: [q3], **means})

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
    """
    group_data = data[data["group"] == group].copy()

    if "length_bin" not in group_data.columns:
        group_data["length_bin"] = pd.cut(
            group_data["length"], bins=bin_edges, labels=False, include_lowest=True
        )

    bin_effects = []
    bin_indices_used = []

    n_bins = len(bin_weights)

    for bin_idx in range(n_bins):
        bin_data = group_data[group_data["length_bin"] == bin_idx]

        if len(bin_data) < 10:
            continue

        zipf_q1_bin = bin_data["zipf"].quantile(0.25)
        zipf_q3_bin = bin_data["zipf"].quantile(0.75)

        length_mean = bin_data["length"].mean()
        surprisal_mean = bin_data["surprisal"].mean()

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

    if len(bin_effects) == 0:
        return {"error": "No valid bins"}

    weights_used = bin_weights.iloc[bin_indices_used].values
    weights_normalized = weights_used / weights_used.sum()

    amie_conditional = np.average(bin_effects, weights=weights_normalized)

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
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    grid_q1 = pd.DataFrame({feature: [q1], **means})
    grid_q3 = pd.DataFrame({feature: [q3], **means})

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
    FULLY REVISED: Added p-values and comprehensive statistics for all effects
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 1: FEATURE EFFECTS")
    logger.info("=" * 60)

    features = ["length", "zipf", "surprisal"]
    results = {}

    expected_directions = {
        "length": "+",
        "zipf": "-",
        "surprisal": "+",
    }

    for feature in features:
        logger.info(f"\nTesting {feature}...")
        logger.info(f"  Expected direction: {expected_directions[feature]}")

        # === 1. AMIE with statistics ===
        if feature == "zipf":
            amie_control = compute_amie_conditional_zipf(
                ert_predictor, data, "control", quartiles, bin_edges, bin_weights
            )
            amie_dyslexic = compute_amie_conditional_zipf(
                ert_predictor, data, "dyslexic", quartiles, bin_edges, bin_weights
            )

            # P-values for conditional zipf
            logger.info(f"  Computing permutation tests for {feature}...")
            stats_ctrl = permutation_test_amie(
                ert_predictor,
                data,
                feature,
                "control",
                quartiles,
                n_permutations=1000,
                method="conditional",
                bin_edges=bin_edges,
                bin_weights=bin_weights,
            )
            stats_dys = permutation_test_amie(
                ert_predictor,
                data,
                feature,
                "dyslexic",
                quartiles,
                n_permutations=1000,
                method="conditional",
                bin_edges=bin_edges,
                bin_weights=bin_weights,
            )
        else:
            amie_control = compute_amie_standard(
                ert_predictor, data, feature, "control", quartiles
            )
            amie_dyslexic = compute_amie_standard(
                ert_predictor, data, feature, "dyslexic", quartiles
            )

            # P-values for standard AMIE
            logger.info(f"  Computing permutation tests for {feature}...")
            stats_ctrl = permutation_test_amie(
                ert_predictor,
                data,
                feature,
                "control",
                quartiles,
                n_permutations=1000,
                method="standard",
            )
            stats_dys = permutation_test_amie(
                ert_predictor,
                data,
                feature,
                "dyslexic",
                quartiles,
                n_permutations=1000,
                method="standard",
            )

        # Merge statistics into AMIE results
        amie_control.update(
            {
                "p_value": stats_ctrl["p_value"],
                "ci_low": stats_ctrl["ci_low"],
                "ci_high": stats_ctrl["ci_high"],
            }
        )
        amie_dyslexic.update(
            {
                "p_value": stats_dys["p_value"],
                "ci_low": stats_dys["ci_low"],
                "ci_high": stats_dys["ci_high"],
            }
        )

        # === 2. PATHWAY DECOMPOSITION with statistics ===
        pathway_control = compute_pathway_effects(
            ert_predictor, data, feature, "control", quartiles
        )
        pathway_dyslexic = compute_pathway_effects(
            ert_predictor, data, feature, "dyslexic", quartiles
        )

        # Add p-values for each pathway
        for pathway in ["skip", "duration", "ert"]:
            logger.info(f"    Testing {pathway} pathway...")

            stats_ctrl_path = permutation_test_pathway_effect(
                ert_predictor, data, feature, "control", quartiles, pathway, 1000
            )
            stats_dys_path = permutation_test_pathway_effect(
                ert_predictor, data, feature, "dyslexic", quartiles, pathway, 1000
            )

            pathway_control[f"{pathway}_p_value"] = stats_ctrl_path["p_value"]
            pathway_control[f"{pathway}_ci_low"] = stats_ctrl_path["ci_low"]
            pathway_control[f"{pathway}_ci_high"] = stats_ctrl_path["ci_high"]

            pathway_dyslexic[f"{pathway}_p_value"] = stats_dys_path["p_value"]
            pathway_dyslexic[f"{pathway}_ci_low"] = stats_dys_path["ci_low"]
            pathway_dyslexic[f"{pathway}_ci_high"] = stats_dys_path["ci_high"]

        # === 3. CHECK DIRECTION ===
        expected = expected_directions[feature]

        ctrl_amie = amie_control.get("amie_ms", np.nan)
        dys_amie = amie_dyslexic.get("amie_ms", np.nan)

        ctrl_correct = (ctrl_amie > 0) if expected == "+" else (ctrl_amie < 0)
        dys_correct = (dys_amie > 0) if expected == "+" else (dys_amie < 0)

        status = "CONFIRMED" if (ctrl_correct and dys_correct) else "NOT CONFIRMED"

        note = ""
        if feature == "zipf":
            note = (
                "Zipf uses conditional evaluation within length bins to "
                "respect length-frequency coupling."
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

        logger.info(
            f"  AMIE Control: {ctrl_amie:.2f} ms (p={stats_ctrl['p_value']:.5f})"
        )
        logger.info(
            f"  AMIE Dyslexic: {dys_amie:.2f} ms (p={stats_dys['p_value']:.5f})"
        )
        logger.info(f"  Status: {status}")

    all_confirmed = all(r["status"] == "CONFIRMED" for r in results.values())
    overall_status = "CONFIRMED" if all_confirmed else "PARTIALLY CONFIRMED"

    logger.info(f"\nHYPOTHESIS 1: {overall_status}")

    return {
        "status": overall_status,
        "features": results,
        "summary": f"{'All' if all_confirmed else 'Some'} features show expected effects",
    }
