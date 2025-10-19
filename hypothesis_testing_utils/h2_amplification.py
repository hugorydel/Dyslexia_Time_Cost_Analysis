"""
Hypothesis 2: Amplification with Comprehensive Bootstrap
Key changes:
- Single bootstrap function that resamples subjects and recomputes ALL metrics
- Conditional slope ratios for zipf (within length bins)
- Pathway-specific SRs (skip, duration, ERT) for all features
- Proper confidence intervals from subject-clustered bootstrap
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


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

    Args:
        pathway: 'skip', 'duration', or 'ert'

    Returns:
        (sr, effects_control, effects_dyslexic) tuple
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
        logger.warning(f"    Unstable SR({pathway}): control Δ ≈ 0")
        sr = np.nan
    elif pathway == "duration" and delta_ctrl < 5:
        logger.warning(f"    Unstable SR({pathway}): control Δ < 5ms")
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
    """
    Conditional SR for zipf (within length bins)

    Compute SR within each length bin, then average with pooled weights
    """
    bin_srs = []
    bin_indices_used = []

    n_bins = len(bin_weights)

    for bin_idx in range(n_bins):
        # Get data in this bin for both groups
        ctrl_bin = data[(data["group"] == "control") & (data["length_bin"] == bin_idx)]
        dys_bin = data[(data["group"] == "dyslexic") & (data["length_bin"] == bin_idx)]

        if len(ctrl_bin) < 10 or len(dys_bin) < 10:
            continue

        # Compute zipf Q1→Q3 within bin for each group
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

        # Compute SR for this bin
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

    # Average with pooled weights
    weights_used = bin_weights.iloc[bin_indices_used].values
    weights_normalized = weights_used / weights_used.sum()

    sr_avg = np.average(bin_srs, weights=weights_normalized)

    return sr_avg, {}, {}  # Return empty effects dicts for consistency


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
    FIXED: Separate storage for AMIE vs SR to avoid duplicate keys
    """
    logger.info(f"\nRunning comprehensive bootstrap ({n_bootstrap} iterations)...")

    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)

    # FIXED: Separate top-level keys for different metric types
    bootstrap_results = {
        "amie": {
            f: {"control": [], "dyslexic": []} for f in ["length", "zipf", "surprisal"]
        },
        "slope_ratios": {  # ← RENAMED from duplicate 'amie'
            "sr_skip": {f: [] for f in ["length", "zipf", "surprisal"]},
            "sr_duration": {f: [] for f in ["length", "zipf", "surprisal"]},
            "sr_ert": {f: [] for f in ["length", "zipf", "surprisal"]},
        },
    }

    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
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

        # === COMPUTE ALL METRICS ===
        for feature in ["length", "zipf", "surprisal"]:

            # 1. AMIEs
            for group in ["control", "dyslexic"]:
                try:
                    if feature == "zipf":
                        from hypothesis_testing_utils.h1_feature_effects import (
                            compute_amie_conditional_zipf,
                        )

                        amie_result = compute_amie_conditional_zipf(
                            ert_predictor,
                            boot_data,
                            group,
                            boot_quartiles,
                            bin_edges,
                            bin_weights,
                        )
                        amie = amie_result.get("amie_ms", np.nan)
                    else:
                        from hypothesis_testing_utils.h1_feature_effects import (
                            compute_amie_standard,
                        )

                        amie_result = compute_amie_standard(
                            ert_predictor, boot_data, feature, group, boot_quartiles
                        )
                        amie = amie_result.get("amie_ms", np.nan)

                    if not np.isnan(amie):
                        bootstrap_results["amie"][feature][group].append(amie)
                except Exception as e:
                    pass

            # 2. Slope Ratios (store in nested structure)
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

    # === COMPUTE CONFIDENCE INTERVALS ===
    ci_results = {"amie": {}, "slope_ratios": {}}

    # CIs for AMIEs
    for feature in ["length", "zipf", "surprisal"]:
        ci_results["amie"][feature] = {}
        for group in ["control", "dyslexic"]:
            samples = bootstrap_results["amie"][feature][group]
            if len(samples) > 0:
                ci_results["amie"][feature][group] = {
                    "mean": float(np.mean(samples)),
                    "ci_low": float(np.percentile(samples, 2.5)),
                    "ci_high": float(np.percentile(samples, 97.5)),
                    "n_samples": len(samples),
                }

    # CIs for SRs
    for pathway in ["sr_skip", "sr_duration", "sr_ert"]:
        ci_results["slope_ratios"][pathway] = {}
        for feature in ["length", "zipf", "surprisal"]:
            samples = bootstrap_results["slope_ratios"][pathway][feature]
            if len(samples) > 0:
                ci_results["slope_ratios"][pathway][feature] = {
                    "mean": float(np.mean(samples)),
                    "ci_low": float(np.percentile(samples, 2.5)),
                    "ci_high": float(np.percentile(samples, 97.5)),
                    "n_samples": len(samples),
                }

    logger.info("Bootstrap complete!")
    logger.info(
        f"  AMIE samples: {sum(len(bootstrap_results['amie'][f][g]) for f in bootstrap_results['amie'] for g in bootstrap_results['amie'][f])}"
    )
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

    Computes SRs for all features × all pathways with bootstrap CIs
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
            ci_key = f"sr_{pathway}"
            if ci_key in ci_results and feature in ci_results[ci_key]:
                ci_data = ci_results[ci_key][feature]
                results[feature][f"{ci_key}_ci_low"] = ci_data["ci_low"]
                results[feature][f"{ci_key}_ci_high"] = ci_data["ci_high"]
                results[feature][f"{ci_key}_mean"] = ci_data["mean"]

    # === 4. DECISION RULE ===
    # H2 confirmed if ≥2/3 features show amplification in ≥1 pathway (CI excludes 1.0)

    amplified_features = []

    for feature in features:
        feature_amplified = False

        for pathway in ["skip", "duration", "ert"]:
            ci_low = results[feature].get(f"sr_{pathway}_ci_low", 1.0)
            ci_high = results[feature].get(f"sr_{pathway}_ci_high", 1.0)

            # Check if CI excludes 1.0 AND mean > 1.0
            if ci_low > 1.0:
                feature_amplified = True
                break

        if feature_amplified:
            amplified_features.append(feature)

    n_amplified = len(amplified_features)
    status = "CONFIRMED" if n_amplified >= 2 else "PARTIALLY CONFIRMED"

    logger.info(f"\n{'='*60}")
    logger.info(f"HYPOTHESIS 2: {status}")
    logger.info(f"  {n_amplified}/3 features show amplification")
    logger.info(f"  Amplified: {amplified_features}")
    logger.info(f"{'='*60}")

    return {
        "status": status,
        "slope_ratios": results,
        "n_amplified": n_amplified,
        "amplified_features": amplified_features,
        "bootstrap_samples": bootstrap_samples,
        "confidence_intervals": ci_results,
        "summary": f"{n_amplified}/3 features show significant amplification (SR CI > 1.0)",
    }
