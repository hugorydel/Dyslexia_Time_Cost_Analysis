"""
Hypothesis 2: Amplification with Comprehensive Bootstrap
REVISED: Fixed SR classification logic to properly detect significant effects
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for two proportions"""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def _classify_sr(ci_low: float, ci_high: float, eps: float = 1e-12) -> str:
    """
    Classify SR based on 95% CI position relative to 1.0

    Returns:
        "amplified" if CI > 1.0 (dyslexics more sensitive)
        "reduced" if CI < 1.0 (dyslexics less sensitive)
        "ns" if CI includes 1.0 (not significant)
        "undetermined" if CI is invalid/missing
    """
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "undetermined"
    if ci_low > 1.0 + eps:
        return "amplified"
    if ci_high < 1.0 - eps:
        return "reduced"
    return "ns"


def _feature_status_from_pathways(ci_dict_for_feature: Dict) -> Tuple[str, str]:
    """
    Determine feature status by checking pathways in priority order: ERT > duration > skip

    Args:
        ci_dict_for_feature: Dict with pathway keys, each containing (ci_low, ci_high) tuple

    Returns:
        (chosen_pathway, status) tuple
    """
    for pathway in ["ert", "duration", "skip"]:
        if pathway in ci_dict_for_feature:
            ci_low, ci_high = ci_dict_for_feature[pathway]
            status = _classify_sr(ci_low, ci_high)
            if status != "undetermined":
                return pathway, status
    return None, "undetermined"


def compute_slope_ratio_standard(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    quartiles: Dict,
    pathway: str = "ert",
) -> Tuple[float, Dict, Dict]:
    """
    Standard SR for length and surprisal
    SR = |Δ_dys| / |Δ_ctrl|
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    grid_q1 = pd.DataFrame({feature: [q1], **means})
    grid_q3 = pd.DataFrame({feature: [q3], **means})

    effects = {}
    for group in ["control", "dyslexic"]:
        ert_q1, p_skip_q1, trt_q1 = ert_predictor.predict_ert(
            grid_q1, group, return_components=True
        )
        ert_q3, p_skip_q3, trt_q3 = ert_predictor.predict_ert(
            grid_q3, group, return_components=True
        )

        effects[group] = {
            "delta_p_skip": p_skip_q3[0] - p_skip_q1[0],
            "delta_trt": trt_q3[0] - trt_q1[0],
            "delta_ert": ert_q3[0] - ert_q1[0],
        }

    # Select pathway
    if pathway == "skip":
        delta_ctrl = abs(effects["control"]["delta_p_skip"])
        delta_dys = abs(effects["dyslexic"]["delta_p_skip"])
    elif pathway == "duration":
        delta_ctrl = abs(effects["control"]["delta_trt"])
        delta_dys = abs(effects["dyslexic"]["delta_trt"])
    else:  # ert
        delta_ctrl = abs(effects["control"]["delta_ert"])
        delta_dys = abs(effects["dyslexic"]["delta_ert"])

    # Check for unstable denominators
    if pathway == "skip" and delta_ctrl < 0.01:
        sr = np.nan
    elif pathway == "duration" and delta_ctrl < 5:
        sr = np.nan
    else:
        sr = delta_dys / delta_ctrl if delta_ctrl > 0 else np.nan

    return sr, effects["control"], effects["dyslexic"]


def compute_slope_ratio_conditional_zipf(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict,
    bin_edges: np.ndarray,
    bin_weights: pd.Series,
    pathway: str = "ert",
) -> Tuple[float, Dict, Dict]:
    """Conditional SR for zipf (within length bins)"""
    bin_srs = []
    bin_indices_used = []

    n_bins = len(bin_weights)

    for bin_idx in range(n_bins):
        ctrl_bin = data[(data["group"] == "control") & (data["length_bin"] == bin_idx)]
        dys_bin = data[(data["group"] == "dyslexic") & (data["length_bin"] == bin_idx)]

        if len(ctrl_bin) < 10 or len(dys_bin) < 10:
            continue

        effects = {}
        for group, bin_data in [("control", ctrl_bin), ("dyslexic", dys_bin)]:
            zipf_q1 = bin_data["zipf"].quantile(0.25)
            zipf_q3 = bin_data["zipf"].quantile(0.75)
            length_mean = bin_data["length"].mean()
            surprisal_mean = bin_data["surprisal"].mean()

            grid_q1 = pd.DataFrame(
                {
                    "length": [length_mean],
                    "zipf": [zipf_q1],
                    "surprisal": [surprisal_mean],
                }
            )
            grid_q3 = pd.DataFrame(
                {
                    "length": [length_mean],
                    "zipf": [zipf_q3],
                    "surprisal": [surprisal_mean],
                }
            )

            ert_q1, p_skip_q1, trt_q1 = ert_predictor.predict_ert(
                grid_q1, group, return_components=True
            )
            ert_q3, p_skip_q3, trt_q3 = ert_predictor.predict_ert(
                grid_q3, group, return_components=True
            )

            effects[group] = {
                "delta_p_skip": p_skip_q3[0] - p_skip_q1[0],
                "delta_trt": trt_q3[0] - trt_q1[0],
                "delta_ert": ert_q3[0] - ert_q1[0],
            }

        if pathway == "skip":
            delta_ctrl = abs(effects["control"]["delta_p_skip"])
            delta_dys = abs(effects["dyslexic"]["delta_p_skip"])
        elif pathway == "duration":
            delta_ctrl = abs(effects["control"]["delta_trt"])
            delta_dys = abs(effects["dyslexic"]["delta_trt"])
        else:
            delta_ctrl = abs(effects["control"]["delta_ert"])
            delta_dys = abs(effects["dyslexic"]["delta_ert"])

        if delta_ctrl > 0:
            sr_bin = delta_dys / delta_ctrl
            bin_srs.append(sr_bin)
            bin_indices_used.append(bin_idx)

    if len(bin_srs) == 0:
        return np.nan, {}, {}

    weights_used = bin_weights.iloc[bin_indices_used].values
    weights_normalized = weights_used / weights_used.sum()

    sr_avg = np.average(bin_srs, weights=weights_normalized)

    return sr_avg, {}, {}


def bootstrap_all_metrics(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict,
    bin_edges: np.ndarray,
    bin_weights: pd.Series,
    n_bootstrap: int = 1000,
) -> Tuple[Dict, Dict]:
    """
    COMPREHENSIVE BOOTSTRAP: Resample subjects and recompute ALL metrics
    FIXED: Proper nested structure for storing results
    """
    logger.info(f"\nRunning comprehensive bootstrap ({n_bootstrap} iterations)...")

    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)

    # Proper nested structure
    bootstrap_results = {
        "slope_ratios": {
            "sr_skip": {f: [] for f in ["length", "zipf", "surprisal"]},
            "sr_duration": {f: [] for f in ["length", "zipf", "surprisal"]},
            "sr_ert": {f: [] for f in ["length", "zipf", "surprisal"]},
        },
    }

    # Suppress per-iteration logging
    for i in tqdm(range(n_bootstrap), desc="Bootstrap", leave=True):
        if (i + 1) % 100 == 0:
            logger.info(f"  Completed {i+1}/{n_bootstrap} bootstrap samples")

        np.random.seed(i)
        boot_subjects = np.random.choice(subjects, size=n_subjects, replace=True)
        boot_data = pd.concat(
            [data[data["subject_id"] == s] for s in boot_subjects], ignore_index=True
        )

        if len(boot_data) < 1000:
            continue

        # Recompute quartiles on bootstrap sample
        boot_quartiles = {
            feat: {
                "q1": boot_data[feat].quantile(0.25),
                "q3": boot_data[feat].quantile(0.75),
                "iqr": boot_data[feat].quantile(0.75) - boot_data[feat].quantile(0.25),
            }
            for feat in ["length", "zipf", "surprisal"]
        }

        # Compute SRs for all features × pathways
        for feature in ["length", "zipf", "surprisal"]:
            for pathway in ["skip", "duration", "ert"]:
                try:
                    if feature == "zipf":
                        sr, _, _ = compute_slope_ratio_conditional_zipf(
                            ert_predictor,
                            boot_data,
                            boot_quartiles,
                            bin_edges,
                            bin_weights,
                            pathway,
                        )
                    else:
                        sr, _, _ = compute_slope_ratio_standard(
                            ert_predictor, boot_data, feature, boot_quartiles, pathway
                        )

                    if not np.isnan(sr):
                        bootstrap_results["slope_ratios"][f"sr_{pathway}"][
                            feature
                        ].append(sr)
                except:
                    pass

    # Compute confidence intervals
    ci_results = {}

    for feature in ["length", "zipf", "surprisal"]:
        ci_results[feature] = {}
        for pathway in ["skip", "duration", "ert"]:
            samples = bootstrap_results["slope_ratios"][f"sr_{pathway}"][feature]
            if len(samples) > 0:
                ci_results[feature][pathway] = {
                    "mean": float(np.mean(samples)),
                    "ci_low": float(np.percentile(samples, 2.5)),
                    "ci_high": float(np.percentile(samples, 97.5)),
                    "n_samples": len(samples),
                }

    logger.info("Bootstrap complete!")
    logger.info(
        f"  SR samples: {sum(len(bootstrap_results['slope_ratios'][p][f]) for p in bootstrap_results['slope_ratios'] for f in bootstrap_results['slope_ratios'][p])}"
    )

    return bootstrap_results, ci_results


def test_hypothesis_2(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict,
    bin_edges: np.ndarray,
    bin_weights: pd.Series,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Test Hypothesis 2: Dyslexic Amplification
    REVISED: Fixed classification logic to properly detect significant effects
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 2: DYSLEXIC AMPLIFICATION")
    logger.info("=" * 60)

    features = ["length", "zipf", "surprisal"]
    results = {}

    # === 1. POINT ESTIMATES ===
    logger.info("\nComputing point estimates...")

    for feature in features:
        logger.info(f"\n{feature}:")

        feature_results = {}

        for pathway in ["skip", "duration", "ert"]:
            if feature == "zipf":
                sr, _, _ = compute_slope_ratio_conditional_zipf(
                    ert_predictor, data, quartiles, bin_edges, bin_weights, pathway
                )
            else:
                sr, _, _ = compute_slope_ratio_standard(
                    ert_predictor, data, feature, quartiles, pathway
                )

            feature_results[f"sr_{pathway}"] = float(sr)
            logger.info(f"  SR({pathway}): {sr:.3f}")

        results[feature] = feature_results

    # === 2. BOOTSTRAP CONFIDENCE INTERVALS ===
    bootstrap_samples, ci_results = bootstrap_all_metrics(
        ert_predictor, data, quartiles, bin_edges, bin_weights, n_bootstrap
    )

    # === 3. MERGE CIs INTO RESULTS ===
    for feature in features:
        for pathway in ["skip", "duration", "ert"]:
            if feature in ci_results and pathway in ci_results[feature]:
                ci_data = ci_results[feature][pathway]
                results[feature][f"sr_{pathway}_ci_low"] = ci_data["ci_low"]
                results[feature][f"sr_{pathway}_ci_high"] = ci_data["ci_high"]
                results[feature][f"sr_{pathway}_mean"] = ci_data["mean"]

    # === 4. FIXED DECISION LOGIC ===
    amplified_features = []
    reduced_features = []
    feature_status = {}

    for feature in features:
        # Pack CIs per pathway for this feature
        packed = {}
        for pathway in ["ert", "duration", "skip"]:
            if feature in ci_results and pathway in ci_results[feature]:
                ci_data = ci_results[feature][pathway]
                packed[pathway] = (ci_data["ci_low"], ci_data["ci_high"])

        # Determine status using priority order
        chosen_pathway, feat_status = _feature_status_from_pathways(packed)

        feature_status[feature] = {"pathway": chosen_pathway, "status": feat_status}

        if feat_status == "amplified":
            amplified_features.append(feature)
        elif feat_status == "reduced":
            reduced_features.append(feature)

    n_significant = len(amplified_features) + len(reduced_features)

    # Determine overall status
    if n_significant >= 2:
        status = "CONFIRMED"
    elif n_significant == 1:
        status = "PARTIALLY CONFIRMED"
    else:
        status = "NOT CONFIRMED"

    logger.info(f"\n{'='*60}")
    logger.info(f"HYPOTHESIS 2: {status}")
    logger.info(f"  {n_significant}/3 features show significant differential effects")
    logger.info(f"  Amplified: {amplified_features}")
    logger.info(f"  Reduced: {reduced_features}")

    # Log feature-specific details
    for feat, fstatus in feature_status.items():
        logger.info(f"  {feat}: {fstatus['status']} (via {fstatus['pathway']} pathway)")

    logger.info(f"{'='*60}")

    return {
        "status": status,
        "feature_status": feature_status,
        "slope_ratios": results,
        "n_significant": n_significant,
        "amplified_features": sorted(amplified_features),
        "reduced_features": sorted(reduced_features),
        "bootstrap_samples": bootstrap_samples,
        "confidence_intervals": ci_results,
        "summary": f"{n_significant}/3 features show significant differential effects (95% CI excludes 1.0)",
    }
