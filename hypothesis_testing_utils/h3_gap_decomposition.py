"""
Hypothesis 3: Gap Decomposition
Shapley decomposition, equal-ease counterfactual, and per-feature equalization
All analyses use CONSISTENT sample for reproducible baselines
Per-feature equalization uses percentile matching instead of mean replacement
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Seed for all H3 analyses
ANALYSIS_SEED = 42


def compute_total_gap(data: pd.DataFrame) -> float:
    """
    Compute total observed gap: mean(ERT_dys) - mean(ERT_ctrl)

    Args:
        ert_predictor: ERTPredictor instance (not used, gap from data)
        data: Full dataset with ERT column

    Returns:
        Gap in milliseconds
    """
    # Use observed ERT values by group
    ctrl_ert = data[data["group"] == "control"]["ERT"].mean()
    dys_ert = data[data["group"] == "dyslexic"]["ERT"].mean()

    gap = dys_ert - ctrl_ert

    return float(gap)


def shapley_decomposition(ert_predictor, data: pd.DataFrame) -> Dict:
    """
    Shapley decomposition: Skip vs Duration contributions

    Path 1: Equalize skip first
    Path 2: Equalize duration first
    Average the two paths

    Args:
        ert_predictor: ERTPredictor instance
        data: Sample data (already sampled by caller - DON'T resample here)

    Returns:
        Dictionary with decomposition results
    """
    logger.info("  Computing Shapley decomposition...")

    # DON'T resample - use data as passed
    sample_data = data.copy()

    # Get predictions for dyslexic observations
    dys_data = sample_data[sample_data["group"] == "dyslexic"]

    if len(dys_data) == 0:
        logger.warning("No dyslexic observations in sample")
        return {"error": "No dyslexic data"}

    # Baseline: Full dyslexic predictions
    ert_dys, p_skip_dys, trt_dys = [], [], []
    for idx, row in dys_data.iterrows():
        features = row[["length", "zipf", "surprisal"]].to_frame().T
        try:
            ert, p_skip, trt = ert_predictor.predict_ert(
                features, "dyslexic", return_components=True
            )
            ert_dys.append(ert[0])
            p_skip_dys.append(p_skip[0])
            trt_dys.append(trt[0])
        except:
            continue

    if len(ert_dys) == 0:
        return {"error": "Prediction failed"}

    mean_ert_dys = np.mean(ert_dys)

    # Control predictions on same observations
    ert_ctrl, p_skip_ctrl, trt_ctrl = [], [], []
    for idx, row in dys_data.iterrows():
        features = row[["length", "zipf", "surprisal"]].to_frame().T
        try:
            ert, p_skip, trt = ert_predictor.predict_ert(
                features, "control", return_components=True
            )
            ert_ctrl.append(ert[0])
            p_skip_ctrl.append(p_skip[0])
            trt_ctrl.append(trt[0])
        except:
            continue

    mean_ert_ctrl = np.mean(ert_ctrl)

    # Path 1: Equalize skip first
    # ERT_1 = [1 - P(skip)_ctrl] × E[TRT|fix]_dys
    ert_1 = [(1 - p_s_c) * t_d for p_s_c, t_d in zip(p_skip_ctrl, trt_dys)]
    mean_ert_1 = np.mean(ert_1)

    delta_skip_path1 = mean_ert_dys - mean_ert_1
    delta_duration_path1 = mean_ert_1 - mean_ert_ctrl

    # Path 2: Equalize duration first
    # ERT_3 = [1 - P(skip)_dys] × E[TRT|fix]_ctrl
    ert_3 = [(1 - p_s_d) * t_c for p_s_d, t_c in zip(p_skip_dys, trt_ctrl)]
    mean_ert_3 = np.mean(ert_3)

    delta_duration_path2 = mean_ert_dys - mean_ert_3
    delta_skip_path2 = mean_ert_3 - mean_ert_ctrl

    # Average
    delta_skip = (delta_skip_path1 + delta_skip_path2) / 2
    delta_duration = (delta_duration_path1 + delta_duration_path2) / 2

    total_gap = mean_ert_dys - mean_ert_ctrl

    # Percentages
    pct_skip = (delta_skip / total_gap * 100) if total_gap != 0 else 0
    pct_duration = (delta_duration / total_gap * 100) if total_gap != 0 else 0

    logger.info(f"    Total gap: {total_gap:.2f} ms")
    logger.info(f"    Skip contribution: {delta_skip:.2f} ms ({pct_skip:.1f}%)")
    logger.info(
        f"    Duration contribution: {delta_duration:.2f} ms ({pct_duration:.1f}%)"
    )

    return {
        "total_gap": float(total_gap),
        "skip_contribution": float(delta_skip),
        "duration_contribution": float(delta_duration),
        "skip_pct": float(pct_skip),
        "duration_pct": float(pct_duration),
    }


def equal_ease_counterfactual(
    ert_predictor, data: pd.DataFrame, quartiles: Dict
) -> Dict:
    """
    Equal-ease counterfactual: Apply standardized shifts to make text easier

    Shifts:
    - Length: -IQR (shorter words)
    - Zipf: +IQR (more frequent words)
    - Surprisal: -IQR (more predictable words)

    Args:
        ert_predictor: ERTPredictor instance
        data: Sample data (already sampled by caller - DON'T resample here)
        quartiles: Feature quartiles (for IQR)

    Returns:
        Dictionary with counterfactual results
    """
    logger.info("  Computing equal-ease counterfactual...")

    # DON'T resample - use data as passed
    sample_data = data.copy()

    # Get IQR for each feature
    iqr = {feat: vals["iqr"] for feat, vals in quartiles.items()}

    # Baseline predictions
    baseline_ert = {"control": [], "dyslexic": []}

    for group_name in ["control", "dyslexic"]:
        group_data = sample_data[sample_data["group"] == group_name]

        for idx, row in group_data.iterrows():
            features = row[["length", "zipf", "surprisal"]].to_frame().T
            try:
                ert = ert_predictor.predict_ert(features, group_name)[0]
                baseline_ert[group_name].append(ert)
            except:
                continue

    if len(baseline_ert["control"]) == 0 or len(baseline_ert["dyslexic"]) == 0:
        logger.warning("Insufficient data for counterfactual")
        return {"error": "Insufficient data"}

    baseline_gap = np.mean(baseline_ert["dyslexic"]) - np.mean(baseline_ert["control"])

    # Apply shifts
    shifted_data = sample_data.copy()
    shifted_data["length"] = shifted_data["length"] - iqr["length"]
    shifted_data["zipf"] = shifted_data["zipf"] + iqr["zipf"]
    shifted_data["surprisal"] = shifted_data["surprisal"] - iqr["surprisal"]

    # Clip to reasonable ranges
    shifted_data["length"] = shifted_data["length"].clip(lower=1)
    shifted_data["zipf"] = shifted_data["zipf"].clip(lower=1, upper=7)
    shifted_data["surprisal"] = shifted_data["surprisal"].clip(lower=0)

    # Counterfactual predictions
    counterfactual_ert = {"control": [], "dyslexic": []}

    for group_name in ["control", "dyslexic"]:
        group_data = shifted_data[shifted_data["group"] == group_name]

        for idx, row in group_data.iterrows():
            features = row[["length", "zipf", "surprisal"]].to_frame().T
            try:
                ert = ert_predictor.predict_ert(features, group_name)[0]
                counterfactual_ert[group_name].append(ert)
            except:
                continue

    counterfactual_gap = np.mean(counterfactual_ert["dyslexic"]) - np.mean(
        counterfactual_ert["control"]
    )

    # Compute savings
    dys_saved = np.mean(baseline_ert["dyslexic"]) - np.mean(
        counterfactual_ert["dyslexic"]
    )
    ctrl_saved = np.mean(baseline_ert["control"]) - np.mean(
        counterfactual_ert["control"]
    )
    gap_shrink = baseline_gap - counterfactual_gap
    gap_shrink_pct = (gap_shrink / baseline_gap * 100) if baseline_gap != 0 else 0

    logger.info(f"    Baseline gap: {baseline_gap:.2f} ms")
    logger.info(f"    Counterfactual gap: {counterfactual_gap:.2f} ms")
    logger.info(f"    Dyslexic saved: {dys_saved:.2f} ms")
    logger.info(f"    Control saved: {ctrl_saved:.2f} ms")
    logger.info(f"    Gap shrink: {gap_shrink:.2f} ms ({gap_shrink_pct:.1f}%)")

    return {
        "baseline_gap": float(baseline_gap),
        "counterfactual_gap": float(counterfactual_gap),
        "dyslexic_saved": float(dys_saved),
        "control_saved": float(ctrl_saved),
        "gap_shrink_ms": float(gap_shrink),
        "gap_shrink_pct": float(gap_shrink_pct),
    }


def per_feature_equalization(ert_predictor, data: pd.DataFrame, feature: str) -> Dict:
    """
    Per-feature equalization: Estimate gap if dyslexics' feature distribution matched controls

    FIXED APPROACH: Use percentile matching to equalize feature distributions
    - For each dyslexic observation, find its percentile in dyslexic distribution
    - Map to same percentile in control distribution
    - This preserves the shape of the distribution while equalizing location/scale

    This approximates "what if dyslexics encountered the same feature distribution as controls"

    Args:
        ert_predictor: ERTPredictor instance
        data: Sample data (already sampled by caller)
        feature: Feature to equalize (for labeling)

    Returns:
        Dictionary with equalization results
    """
    logger.info(f"  Computing per-feature equalization for {feature}...")

    # Use data as passed (already sampled with fixed seed)
    sample_data = data.copy()

    # === BASELINE GAP ===
    ert_observed = []
    groups_list = []
    valid_indices = []

    for idx, row in sample_data.iterrows():
        group = row["group"]
        features = row[["length", "zipf", "surprisal"]].to_frame().T
        try:
            ert = ert_predictor.predict_ert(features, group)[0]
            ert_observed.append(ert)
            groups_list.append(group)
            valid_indices.append(idx)
        except:
            continue

    if len(ert_observed) == 0:
        return {"error": "No valid predictions"}

    sample_data_clean = sample_data.loc[valid_indices].copy()
    sample_data_clean["ERT_observed"] = ert_observed
    sample_data_clean["group"] = groups_list

    baseline_gap = (
        sample_data_clean[sample_data_clean["group"] == "dyslexic"][
            "ERT_observed"
        ].mean()
        - sample_data_clean[sample_data_clean["group"] == "control"][
            "ERT_observed"
        ].mean()
    )

    # === COUNTERFACTUAL: Percentile Matching ===
    # Equalize the feature distribution using percentile mapping

    control_data = sample_data_clean[sample_data_clean["group"] == "control"]
    dyslexic_data = sample_data_clean[sample_data_clean["group"] == "dyslexic"]

    ert_counterfactual = []

    for idx, row in sample_data_clean.iterrows():
        group = row["group"]
        features_cf = row[["length", "zipf", "surprisal"]].copy().to_frame().T

        if group == "dyslexic":
            # PERCENTILE MATCHING:
            # Find this observation's percentile in dyslexic distribution
            feature_value = row[feature]
            percentile = (dyslexic_data[feature] <= feature_value).mean()

            # Map to same percentile in control distribution
            # This equalizes the distribution while preserving within-group ordering
            control_value = control_data[feature].quantile(percentile)

            # Replace feature value
            features_cf[feature] = control_value

        # Predict with ORIGINAL group model (important!)
        try:
            ert = ert_predictor.predict_ert(features_cf, group)[0]
            ert_counterfactual.append(ert)
        except:
            ert_counterfactual.append(np.nan)

    sample_data_clean["ERT_counterfactual"] = ert_counterfactual
    sample_data_clean = sample_data_clean.dropna(subset=["ERT_counterfactual"])

    if len(sample_data_clean) == 0:
        return {"error": "No valid counterfactual predictions"}

    counterfactual_gap = (
        sample_data_clean[sample_data_clean["group"] == "dyslexic"][
            "ERT_counterfactual"
        ].mean()
        - sample_data_clean[sample_data_clean["group"] == "control"][
            "ERT_counterfactual"
        ].mean()
    )

    # === COMPUTE GAP EXPLAINED ===
    gap_explained = baseline_gap - counterfactual_gap
    pct_explained = (gap_explained / baseline_gap * 100) if baseline_gap != 0 else 0

    logger.info(f"    Baseline gap: {baseline_gap:.2f} ms")
    logger.info(
        f"    Counterfactual gap (with {feature} equalized): {counterfactual_gap:.2f} ms"
    )
    logger.info(
        f"    Gap explained by {feature}: {gap_explained:.2f} ms ({pct_explained:.1f}%)"
    )

    return {
        "feature": feature,
        "baseline_gap": float(baseline_gap),
        "counterfactual_gap": float(counterfactual_gap),
        "gap_explained": float(gap_explained),
        "pct_explained": float(pct_explained),
        "method": "percentile_matching",
    }


def test_hypothesis_3(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict[str, Dict[str, float]],
) -> Dict:
    """
    Test Hypothesis 3: Gap decomposition

    FIXED: All gap computations use the SAME sample for consistency

    Args:
        ert_predictor: ERTPredictor instance
        data: Full dataset
        quartiles: Feature quartiles

    Returns:
        Dictionary with H3 test results
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 3: GAP DECOMPOSITION")
    logger.info("=" * 60)

    # === USE CONSISTENT SAMPLE FOR ALL ANALYSES ===
    # This fixes the "inconsistent gap baseline" bug
    np.random.seed(ANALYSIS_SEED)
    sample_size = min(5000, len(data))
    analysis_sample = data.sample(sample_size, random_state=ANALYSIS_SEED)

    logger.info(
        f"\nUsing consistent sample of {len(analysis_sample):,} observations for all H3 analyses"
    )
    logger.info(f"Random seed: {ANALYSIS_SEED}")

    # === 1. TOTAL GAP (on consistent sample) ===
    total_gap_observed = compute_total_gap(analysis_sample)
    logger.info(f"\nTotal observed gap: {total_gap_observed:.2f} ms")

    # === 2. SHAPLEY DECOMPOSITION (on same sample) ===
    shapley = shapley_decomposition(ert_predictor, analysis_sample)

    # Verify Shapley gap matches total gap (should be close)
    if "total_gap" in shapley and "error" not in shapley:
        gap_diff = abs(shapley["total_gap"] - total_gap_observed)
        if gap_diff > 5:  # More than 5ms difference is suspicious
            logger.warning(
                f"⚠️  Shapley gap ({shapley['total_gap']:.2f}) differs from total gap "
                f"({total_gap_observed:.2f}) by {gap_diff:.2f}ms"
            )
        else:
            logger.info(
                f"✓ Shapley gap consistent with total gap (diff: {gap_diff:.2f}ms)"
            )

    # === 3. EQUAL-EASE COUNTERFACTUAL (on same sample) ===
    equal_ease = equal_ease_counterfactual(ert_predictor, analysis_sample, quartiles)

    # Verify equal-ease baseline matches
    if "baseline_gap" in equal_ease and "error" not in equal_ease:
        gap_diff = abs(equal_ease["baseline_gap"] - total_gap_observed)
        if gap_diff > 5:
            logger.warning(
                f"⚠️  Equal-ease baseline ({equal_ease['baseline_gap']:.2f}) differs from total gap "
                f"({total_gap_observed:.2f}) by {gap_diff:.2f}ms"
            )
        else:
            logger.info(
                f"✓ Equal-ease baseline consistent with total gap (diff: {gap_diff:.2f}ms)"
            )

    # === 4. PER-FEATURE EQUALIZATION (on same sample) ===
    per_feature = {}
    for feature in ["length", "zipf", "surprisal"]:
        per_feature[feature] = per_feature_equalization(
            ert_predictor, analysis_sample, feature
        )

        # Verify baseline consistency
        if (
            "baseline_gap" in per_feature[feature]
            and "error" not in per_feature[feature]
        ):
            gap_diff = abs(per_feature[feature]["baseline_gap"] - total_gap_observed)
            if gap_diff > 5:
                logger.warning(
                    f"⚠️  {feature} equalization baseline ({per_feature[feature]['baseline_gap']:.2f}) "
                    f"differs from total gap ({total_gap_observed:.2f}) by {gap_diff:.2f}ms"
                )

    # === 5. IMPROVED DECISION LOGIC ===
    # Check both positive and negative explanations
    any_substantial_effect = any(
        abs(res.get("pct_explained", 0)) >= 25 for res in per_feature.values()
    )

    text_difficulty_explained = equal_ease.get("gap_shrink_pct", 0) >= 20

    if any_substantial_effect and text_difficulty_explained:
        status = "STRONGLY CONFIRMED"
        summary_detail = "both differential sensitivity and text difficulty"
    elif any_substantial_effect:
        # Check if effects are positive or negative
        positive_effects = [
            res["feature"]
            for res in per_feature.values()
            if res.get("pct_explained", 0) > 25
        ]
        negative_effects = [
            res["feature"]
            for res in per_feature.values()
            if res.get("pct_explained", 0) < -25
        ]

        if positive_effects:
            status = "CONFIRMED"
            summary_detail = f"differential sensitivity ({', '.join(positive_effects)})"
        elif negative_effects:
            status = "CONFIRMED"
            summary_detail = (
                f"compositional differences (dyslexics encounter easier "
                f"{', '.join(negative_effects)} but still show larger gaps)"
            )
        else:
            status = "CONFIRMED"
            summary_detail = "differential sensitivity to word features"
    elif text_difficulty_explained:
        status = "CONFIRMED"
        summary_detail = "text difficulty (simplification reduces gap)"
    else:
        status = "NOT CONFIRMED"
        summary_detail = "features explain minimal portion of gap"

    # Generate summary
    summary = f"Features explain gap through {summary_detail}"

    # Add interpretation note about negative values
    interpretation_note = ""
    if any(res.get("pct_explained", 0) < 0 for res in per_feature.values()):
        interpretation_note = (
            "Note: Negative per-feature explanations indicate that dyslexics "
            "encounter easier text in those dimensions, suggesting the gap exists "
            "despite compositional advantages rather than because of compositional disadvantages."
        )

    logger.info(f"\nHYPOTHESIS 3: {status}")
    logger.info(f"  {summary}")
    if interpretation_note:
        logger.info(f"  {interpretation_note}")

    # === 6. CONSISTENCY CHECK SUMMARY ===
    gaps_to_check = {
        "total_gap": total_gap_observed,
        "shapley_gap": shapley.get("total_gap", None),
        "equal_ease_baseline": equal_ease.get("baseline_gap", None),
    }

    valid_gaps = [g for g in gaps_to_check.values() if g is not None]
    if len(valid_gaps) > 1:
        max_difference = max(valid_gaps) - min(valid_gaps)
        logger.info(
            f"\n✓ Gap consistency check: max difference = {max_difference:.2f}ms"
        )
        if max_difference < 1.0:
            logger.info(f"  Excellent consistency (< 1ms)")
        elif max_difference < 5.0:
            logger.info(f"  Good consistency (< 5ms)")
        else:
            logger.warning(f"  ⚠️  Poor consistency (> 5ms)")

    return {
        "status": status,
        "total_gap": total_gap_observed,
        "sample_size": len(analysis_sample),
        "random_seed": ANALYSIS_SEED,
        "shapley_decomposition": shapley,
        "equal_ease_counterfactual": equal_ease,
        "per_feature_equalization": per_feature,
        "summary": summary,
        "interpretation_note": interpretation_note,
        "gap_consistency_check": {
            "total_gap": total_gap_observed,
            "shapley_gap": shapley.get("total_gap", None),
            "equal_ease_baseline": equal_ease.get("baseline_gap", None),
            "max_difference": (
                max(valid_gaps) - min(valid_gaps) if len(valid_gaps) > 1 else 0
            ),
        },
    }
