"""
Hypothesis 2: Amplification with Comprehensive Bootstrap
FULLY REVISED: Added p-values for slope ratios
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


def cohens_d(
    mean1: float, mean2: float, std1: float, std2: float, n1: int, n2: int
) -> float:
    """
    Compute Cohen's d effect size for two means
    Uses pooled standard deviation
    """
    if n1 <= 0 or n2 <= 0:
        return np.nan

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (mean2 - mean1) / pooled_std


def _classify_sr(ci_low: float, ci_high: float, eps: float = 1e-12) -> str:
    """Classify SR based on 95% CI position relative to 1.0"""
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "undetermined"
    if ci_low > 1.0 + eps:
        return "amplified"
    if ci_high < 1.0 - eps:
        return "reduced"
    return "ns"


def _feature_status_from_pathways(ci_dict_for_feature: Dict) -> Tuple[str, str]:
    """Determine feature status by checking pathways in priority order: ERT > duration > skip"""
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
    """Standard SR for length and surprisal, now with Cohen's d for duration and ERT"""
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

        # Calculate Cohen's d for duration and ERT pathways
        # Get actual data near Q1 and Q3 (within 10% range)
        q1_range = (q1 * 0.9, q1 * 1.1)
        q3_range = (q3 * 0.9, q3 * 1.1)

        group_data = data[data["group"] == group]

        q1_data = group_data[
            (group_data[feature] >= q1_range[0]) & (group_data[feature] <= q1_range[1])
        ]
        q3_data = group_data[
            (group_data[feature] >= q3_range[0]) & (group_data[feature] <= q3_range[1])
        ]

        # Cohen's d for duration (TRT)
        if len(q1_data) > 0 and len(q3_data) > 0:
            # For TRT: use observed durations when not skipped
            trt_q1_vals = q1_data[q1_data["skip"] == 0]["TRT"]
            trt_q3_vals = q3_data[q3_data["skip"] == 0]["TRT"]

            if len(trt_q1_vals) > 1 and len(trt_q3_vals) > 1:
                effects[group]["cohens_d_duration"] = cohens_d(
                    trt_q1_vals.mean(),
                    trt_q3_vals.mean(),
                    trt_q1_vals.std(),
                    trt_q3_vals.std(),
                    len(trt_q1_vals),
                    len(trt_q3_vals),
                )
            else:
                effects[group]["cohens_d_duration"] = np.nan

            # Cohen's d for ERT (total reading time)
            ert_q1_vals = q1_data["ERT"]
            ert_q3_vals = q3_data["ERT"]

            if len(ert_q1_vals) > 1 and len(ert_q3_vals) > 1:
                effects[group]["cohens_d_ert"] = cohens_d(
                    ert_q1_vals.mean(),
                    ert_q3_vals.mean(),
                    ert_q1_vals.std(),
                    ert_q3_vals.std(),
                    len(ert_q1_vals),
                    len(ert_q3_vals),
                )
            else:
                effects[group]["cohens_d_ert"] = np.nan
        else:
            effects[group]["cohens_d_duration"] = np.nan
            effects[group]["cohens_d_ert"] = np.nan

    if pathway == "skip":
        delta_ctrl = abs(effects["control"]["delta_p_skip"])
        delta_dys = abs(effects["dyslexic"]["delta_p_skip"])
    elif pathway == "duration":
        delta_ctrl = abs(effects["control"]["delta_trt"])
        delta_dys = abs(effects["dyslexic"]["delta_trt"])
    else:
        delta_ctrl = abs(effects["control"]["delta_ert"])
        delta_dys = abs(effects["dyslexic"]["delta_ert"])

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
    """Conditional SR for zipf (within length bins), now with Cohen's d for duration and ERT"""
    bin_srs = []
    bin_indices_used = []

    # Also collect Cohen's d values across bins
    cohens_d_duration_ctrl = []
    cohens_d_duration_dys = []
    cohens_d_ert_ctrl = []
    cohens_d_ert_dys = []

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

            # Calculate Cohen's d for this bin
            # Get data near Q1 and Q3
            q1_range = (zipf_q1 * 0.9, zipf_q1 * 1.1)
            q3_range = (zipf_q3 * 0.9, zipf_q3 * 1.1)

            q1_data = bin_data[
                (bin_data["zipf"] >= q1_range[0]) & (bin_data["zipf"] <= q1_range[1])
            ]
            q3_data = bin_data[
                (bin_data["zipf"] >= q3_range[0]) & (bin_data["zipf"] <= q3_range[1])
            ]

            if len(q1_data) > 0 and len(q3_data) > 0:
                # Cohen's d for duration (TRT)
                trt_q1_vals = q1_data[q1_data["skip"] == 0]["TRT"]
                trt_q3_vals = q3_data[q3_data["skip"] == 0]["TRT"]

                if len(trt_q1_vals) > 1 and len(trt_q3_vals) > 1:
                    cd_dur = cohens_d(
                        trt_q1_vals.mean(),
                        trt_q3_vals.mean(),
                        trt_q1_vals.std(),
                        trt_q3_vals.std(),
                        len(trt_q1_vals),
                        len(trt_q3_vals),
                    )
                    if group == "control":
                        cohens_d_duration_ctrl.append(cd_dur)
                    else:
                        cohens_d_duration_dys.append(cd_dur)

                # Cohen's d for ERT
                ert_q1_vals = q1_data["ERT"]
                ert_q3_vals = q3_data["ERT"]

                if len(ert_q1_vals) > 1 and len(ert_q3_vals) > 1:
                    cd_ert = cohens_d(
                        ert_q1_vals.mean(),
                        ert_q3_vals.mean(),
                        ert_q1_vals.std(),
                        ert_q3_vals.std(),
                        len(ert_q1_vals),
                        len(ert_q3_vals),
                    )
                    if group == "control":
                        cohens_d_ert_ctrl.append(cd_ert)
                    else:
                        cohens_d_ert_dys.append(cd_ert)

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

    # Average Cohen's d values
    effect_sizes_ctrl = {
        "cohens_d_duration": (
            np.mean(cohens_d_duration_ctrl) if cohens_d_duration_ctrl else np.nan
        ),
        "cohens_d_ert": np.mean(cohens_d_ert_ctrl) if cohens_d_ert_ctrl else np.nan,
    }

    effect_sizes_dys = {
        "cohens_d_duration": (
            np.mean(cohens_d_duration_dys) if cohens_d_duration_dys else np.nan
        ),
        "cohens_d_ert": np.mean(cohens_d_ert_dys) if cohens_d_ert_dys else np.nan,
    }

    return sr_avg, effect_sizes_ctrl, effect_sizes_dys


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
    """
    logger.info(f"\nRunning comprehensive bootstrap ({n_bootstrap} iterations)...")

    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)

    bootstrap_results = {
        "amie": {
            f: {"control": [], "dyslexic": []} for f in ["length", "zipf", "surprisal"]
        },
        "slope_ratios": {
            "sr_skip": {f: [] for f in ["length", "zipf", "surprisal"]},
            "sr_duration": {f: [] for f in ["length", "zipf", "surprisal"]},
            "sr_ert": {f: [] for f in ["length", "zipf", "surprisal"]},
        },
    }

    for i in tqdm(range(n_bootstrap), desc="Bootstrap", leave=True):
        if (i + 1) % 100 == 0:
            logger.info(f"  Completed {i+1}/{n_bootstrap} bootstrap samples")

        rng = np.random.RandomState(i)
        boot_subjects = rng.choice(subjects, size=n_subjects, replace=True)
        boot_data = pd.concat(
            [data[data["subject_id"] == s] for s in boot_subjects], ignore_index=True
        )

        if len(boot_data) < 1000:
            continue

        boot_quartiles = {
            feat: {
                "q1": boot_data[feat].quantile(0.25),
                "q3": boot_data[feat].quantile(0.75),
                "iqr": boot_data[feat].quantile(0.75) - boot_data[feat].quantile(0.25),
            }
            for feat in ["length", "zipf", "surprisal"]
        }

        for feature in ["length", "zipf", "surprisal"]:
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
                except Exception:
                    pass

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

    # === COMPUTE CONFIDENCE INTERVALS AND P-VALUES ===
    ci_results = {}

    for feature in ["length", "zipf", "surprisal"]:
        ci_results[feature] = {}

        # CIs for AMIEs
        for group in ["control", "dyslexic"]:
            samples = bootstrap_results["amie"][feature][group]
            if len(samples) > 0:
                ci_results[feature][f"amie_{group}"] = {
                    "mean": float(np.mean(samples)),
                    "ci_low": float(np.percentile(samples, 2.5)),
                    "ci_high": float(np.percentile(samples, 97.5)),
                    "n_samples": len(samples),
                }

        # CIs and p-values for SRs
        for pathway in ["skip", "duration", "ert"]:
            samples = bootstrap_results["slope_ratios"][f"sr_{pathway}"][feature]
            if len(samples) > 0:
                mean_sr = float(np.mean(samples))
                ci_low = float(np.percentile(samples, 2.5))
                ci_high = float(np.percentile(samples, 97.5))

                # P-value: proportion of SR <= 1.0 or SR >= 1.0
                if mean_sr > 1.0:
                    p_value = float(np.mean(np.array(samples) <= 1.0))
                else:
                    p_value = float(np.mean(np.array(samples) >= 1.0))

                # Two-tailed
                p_value = min(1.0, 2 * p_value)

                ci_results[feature][pathway] = {
                    "mean": mean_sr,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "n_samples": len(samples),
                }

    logger.info("Bootstrap complete!")

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
    FULLY REVISED: Added p-values for all slope ratios
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
                sr, ctrl_effects, dys_effects = compute_slope_ratio_conditional_zipf(
                    ert_predictor, data, quartiles, bin_edges, bin_weights, pathway
                )
            else:
                sr, ctrl_effects, dys_effects = compute_slope_ratio_standard(
                    ert_predictor, data, feature, quartiles, pathway
                )

            feature_results[f"sr_{pathway}"] = float(sr)
            logger.info(f"  SR({pathway}): {sr:.3f}")

        # Capture Cohen's d values (computed differently for zipf vs other features)
        # - length/surprisal: from actual data at Q1/Q3
        # - zipf: weighted average across length bins
        feature_results["cohens_d_duration_control"] = float(
            ctrl_effects.get("cohens_d_duration", np.nan)
        )
        feature_results["cohens_d_duration_dyslexic"] = float(
            dys_effects.get("cohens_d_duration", np.nan)
        )
        feature_results["cohens_d_ert_control"] = float(
            ctrl_effects.get("cohens_d_ert", np.nan)
        )
        feature_results["cohens_d_ert_dyslexic"] = float(
            dys_effects.get("cohens_d_ert", np.nan)
        )

        logger.info(
            f"  Cohen's d (duration): Control={feature_results['cohens_d_duration_control']:.3f}, Dyslexic={feature_results['cohens_d_duration_dyslexic']:.3f}"
        )
        logger.info(
            f"  Cohen's d (ERT): Control={feature_results['cohens_d_ert_control']:.3f}, Dyslexic={feature_results['cohens_d_ert_dyslexic']:.3f}"
        )

        results[feature] = feature_results

    # === 2. BOOTSTRAP CIs AND P-VALUES ===
    bootstrap_samples, ci_results = bootstrap_all_metrics(
        ert_predictor, data, quartiles, bin_edges, bin_weights, n_bootstrap
    )

    # === 3. MERGE CIs AND P-VALUES INTO RESULTS ===
    for feature in features:
        for group in ["control", "dyslexic"]:
            ci_key = f"amie_{group}"
            if ci_key in ci_results.get(feature, {}):
                ci_data = ci_results[feature][ci_key]
                results[feature][f"amie_{group}_mean"] = ci_data["mean"]
                results[feature][f"amie_{group}_ci_low"] = ci_data["ci_low"]
                results[feature][f"amie_{group}_ci_high"] = ci_data["ci_high"]

        for pathway in ["skip", "duration", "ert"]:
            if feature in ci_results and pathway in ci_results[feature]:
                ci_data = ci_results[feature][pathway]
                results[feature][f"sr_{pathway}_ci_low"] = ci_data["ci_low"]
                results[feature][f"sr_{pathway}_ci_high"] = ci_data["ci_high"]
                results[feature][f"sr_{pathway}_mean"] = ci_data["mean"]
                results[feature][f"sr_{pathway}_p_value"] = ci_data["p_value"]

    # === 4. DECISION LOGIC ===
    amplified_features = []
    reduced_features = []
    feature_status = {}

    for feature in features:
        packed = {}
        for pathway in ["ert", "duration", "skip"]:
            if feature in ci_results and pathway in ci_results[feature]:
                ci_data = ci_results[feature][pathway]
                packed[pathway] = (ci_data["ci_low"], ci_data["ci_high"])

        chosen_pathway, feat_status = _feature_status_from_pathways(packed)

        feature_status[feature] = {"pathway": chosen_pathway, "status": feat_status}

        if feat_status == "amplified":
            amplified_features.append(feature)
        elif feat_status == "reduced":
            reduced_features.append(feature)

    n_significant = len(amplified_features) + len(reduced_features)

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

    for feat, fstatus in feature_status.items():
        p_val = results[feat].get(f"sr_{fstatus['pathway']}_p_value", np.nan)
        logger.info(
            f"  {feat}: {fstatus['status']} (via {fstatus['pathway']} pathway, "
            f"p={p_val:.5f})"
        )

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
        "summary": f"{n_significant}/3 features show significant differential effects",
    }
