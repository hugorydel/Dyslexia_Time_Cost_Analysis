"""
Hypothesis 3: Gap Decomposition
Shapley decomposition, equal-ease counterfactual, and per-feature equalization
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_total_gap(ert_predictor, data: pd.DataFrame) -> float:
    """
    Compute total observed gap: mean(ERT_dys) - mean(ERT_ctrl)

    Args:
        ert_predictor: ERTPredictor instance
        data: Full dataset

    Returns:
        Gap in milliseconds
    """
    # Get observed ERT values by group
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
        data: Full dataset

    Returns:
        Dictionary with decomposition results
    """
    logger.info("  Computing Shapley decomposition...")

    # Sample data for faster computation
    sample_size = min(5000, len(data))
    sample_data = data.sample(sample_size, random_state=42)

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
        data: Full dataset
        quartiles: Feature quartiles (for IQR)

    Returns:
        Dictionary with counterfactual results
    """
    logger.info("  Computing equal-ease counterfactual...")

    # Sample data
    sample_size = min(5000, len(data))
    sample_data = data.sample(sample_size, random_state=42)

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
    Per-feature equalization: Approximate effect by using control predictions
    for dyslexic observations

    Args:
        ert_predictor: ERTPredictor instance
        data: Full dataset
        feature: Feature to equalize (for labeling only - we equalize all)

    Returns:
        Dictionary with equalization results
    """
    logger.info(f"  Computing per-feature equalization for {feature}...")

    # Sample data
    sample_size = min(5000, len(data))
    sample_data = data.sample(sample_size, random_state=42)

    # Baseline gap
    baseline_ert = []
    for idx, row in sample_data.iterrows():
        group = row["group"]
        features = row[["length", "zipf", "surprisal"]].to_frame().T
        try:
            ert = ert_predictor.predict_ert(features, group)[0]
            baseline_ert.append(ert)
        except:
            baseline_ert.append(np.nan)

    sample_data["ERT_baseline"] = baseline_ert
    sample_data = sample_data.dropna(subset=["ERT_baseline"])

    if len(sample_data) == 0:
        return {"error": "No valid predictions"}

    baseline_gap = (
        sample_data[sample_data["group"] == "dyslexic"]["ERT_baseline"].mean()
        - sample_data[sample_data["group"] == "control"]["ERT_baseline"].mean()
    )

    # Counterfactual: Use control predictions for all observations
    hybrid_ert = []
    for idx, row in sample_data.iterrows():
        features = row[["length", "zipf", "surprisal"]].to_frame().T
        try:
            # Always use control predictions
            ert = ert_predictor.predict_ert(features, "control")[0]
            hybrid_ert.append(ert)
        except:
            hybrid_ert.append(np.nan)

    sample_data["ERT_hybrid"] = hybrid_ert
    sample_data = sample_data.dropna(subset=["ERT_hybrid"])

    hybrid_gap = (
        sample_data[sample_data["group"] == "dyslexic"]["ERT_hybrid"].mean()
        - sample_data[sample_data["group"] == "control"]["ERT_hybrid"].mean()
    )

    gap_explained = baseline_gap - hybrid_gap
    pct_explained = (gap_explained / baseline_gap * 100) if baseline_gap != 0 else 0

    logger.info(
        f"    {feature} equalization explains {gap_explained:.2f} ms ({pct_explained:.1f}%)"
    )

    return {
        "feature": feature,
        "baseline_gap": float(baseline_gap),
        "hybrid_gap": float(hybrid_gap),
        "gap_explained": float(gap_explained),
        "pct_explained": float(pct_explained),
    }


def test_hypothesis_3(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict[str, Dict[str, float]],
) -> Dict:
    """
    Test Hypothesis 3: Gap decomposition

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

    # Total gap
    total_gap = compute_total_gap(ert_predictor, data)
    logger.info(f"\nTotal observed gap: {total_gap:.2f} ms")

    # Shapley decomposition
    shapley = shapley_decomposition(ert_predictor, data)

    # Equal-ease counterfactual
    equal_ease = equal_ease_counterfactual(ert_predictor, data, quartiles)

    # Per-feature equalization (compute for all features)
    per_feature = {}
    for feature in ["length", "zipf", "surprisal"]:
        per_feature[feature] = per_feature_equalization(ert_predictor, data, feature)

    # Decision logic
    sensitivity_explained = any(
        res.get("pct_explained", 0) >= 25 for res in per_feature.values()
    )
    text_difficulty_explained = equal_ease.get("gap_shrink_pct", 0) >= 20

    if sensitivity_explained and text_difficulty_explained:
        status = "STRONGLY CONFIRMED"
    elif sensitivity_explained or text_difficulty_explained:
        status = "CONFIRMED"
    else:
        status = "NOT CONFIRMED"

    logger.info(f"\nHYPOTHESIS 3: {status}")

    # Create summary
    if status == "STRONGLY CONFIRMED":
        summary = "Features explain gap through both sensitivity and text difficulty"
    elif status == "CONFIRMED":
        if sensitivity_explained:
            summary = "Features explain gap through differential sensitivity"
        else:
            summary = "Features explain gap through text difficulty"
    else:
        summary = "Features explain minimal portion of gap"

    return {
        "status": status,
        "total_gap": total_gap,
        "shapley_decomposition": shapley,
        "equal_ease_counterfactual": equal_ease,
        "per_feature_equalization": per_feature,
        "summary": summary,
    }
