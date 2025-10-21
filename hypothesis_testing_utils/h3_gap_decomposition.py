"""
Hypothesis 3: Gap Decomposition - REVISED
Key changes:
1. Consistent gap baseline across ALL H3 analyses (computed once on shared sample)
2. Added equal-ease feature contributions (Shapley decomposition)
3. All gap measurements reference the same canonical baseline
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANALYSIS_SEED = 42


def compute_canonical_gap(data: pd.DataFrame) -> float:
    """
    Compute canonical gap on the analysis sample
    This is THE gap value used as baseline for all H3 analyses
    """
    ctrl_ert = data[data["group"] == "control"]["ERT"].mean()
    dys_ert = data[data["group"] == "dyslexic"]["ERT"].mean()
    return float(dys_ert - ctrl_ert)


def shapley_decomposition(
    ert_predictor, data: pd.DataFrame, canonical_gap: float
) -> Dict:
    """
    Shapley decomposition: Skip vs Duration contributions
    Uses canonical gap as baseline
    """
    logger.info("  Computing Shapley decomposition...")

    sample_data = data.copy()
    dys_data = sample_data[sample_data["group"] == "dyslexic"]

    if len(dys_data) == 0:
        return {"error": "No dyslexic data"}

    # Predictions for dyslexic observations
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
    ert_1 = [(1 - p_s_c) * t_d for p_s_c, t_d in zip(p_skip_ctrl, trt_dys)]
    mean_ert_1 = np.mean(ert_1)

    delta_skip_path1 = mean_ert_dys - mean_ert_1
    delta_duration_path1 = mean_ert_1 - mean_ert_ctrl

    # Path 2: Equalize duration first
    ert_3 = [(1 - p_s_d) * t_c for p_s_d, t_c in zip(p_skip_dys, trt_ctrl)]
    mean_ert_3 = np.mean(ert_3)

    delta_duration_path2 = mean_ert_dys - mean_ert_3
    delta_skip_path2 = mean_ert_3 - mean_ert_ctrl

    # Average
    delta_skip = (delta_skip_path1 + delta_skip_path2) / 2
    delta_duration = (delta_duration_path1 + delta_duration_path2) / 2

    # Use canonical gap for percentages
    pct_skip = (delta_skip / canonical_gap * 100) if canonical_gap != 0 else 0
    pct_duration = (delta_duration / canonical_gap * 100) if canonical_gap != 0 else 0

    logger.info(f"    Canonical gap: {canonical_gap:.2f} ms")
    logger.info(f"    Skip contribution: {delta_skip:.2f} ms ({pct_skip:.1f}%)")
    logger.info(
        f"    Duration contribution: {delta_duration:.2f} ms ({pct_duration:.1f}%)"
    )

    return {
        "total_gap": float(canonical_gap),
        "skip_contribution": float(delta_skip),
        "duration_contribution": float(delta_duration),
        "skip_pct": float(pct_skip),
        "duration_pct": float(pct_duration),
    }


def equal_ease_counterfactual(
    ert_predictor, data: pd.DataFrame, quartiles: Dict, canonical_gap: float
) -> Dict:
    """
    Equal-ease counterfactual with canonical gap baseline
    """
    logger.info("  Computing equal-ease counterfactual...")

    sample_data = data.copy()

    # Get IQR for each feature
    iqr = {feat: vals["iqr"] for feat, vals in quartiles.items()}

    # Baseline predictions (should match canonical gap)
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
    gap_shrink = canonical_gap - counterfactual_gap  # Use canonical gap
    gap_shrink_pct = (gap_shrink / canonical_gap * 100) if canonical_gap != 0 else 0

    logger.info(f"    Canonical gap: {canonical_gap:.2f} ms")
    logger.info(f"    Counterfactual gap: {counterfactual_gap:.2f} ms")
    logger.info(f"    Dyslexic saved: {dys_saved:.2f} ms")
    logger.info(f"    Control saved: {ctrl_saved:.2f} ms")
    logger.info(f"    Gap shrink: {gap_shrink:.2f} ms ({gap_shrink_pct:.1f}%)")

    return {
        "baseline_gap": float(canonical_gap),
        "counterfactual_gap": float(counterfactual_gap),
        "dyslexic_saved": float(dys_saved),
        "control_saved": float(ctrl_saved),
        "gap_shrink_ms": float(gap_shrink),
        "gap_shrink_pct": float(gap_shrink_pct),
    }


def equal_ease_feature_contributions(
    ert_predictor, data: pd.DataFrame, quartiles: Dict, n_permutations: int = 64
) -> Dict:
    """
    Shapley decomposition of equal-ease feature contributions

    Returns how much each feature (length, zipf, surprisal) contributes
    to closing the gap when applying equal-ease shifts
    """
    logger.info(f"  Computing feature contributions ({n_permutations} permutations)...")

    # Get IQR shifts
    shift_vector = {
        "length": -quartiles["length"]["iqr"],
        "zipf": quartiles["zipf"]["iqr"],
        "surprisal": -quartiles["surprisal"]["iqr"],
    }

    features = ["length", "zipf", "surprisal"]
    contributions = {f: [] for f in features}

    # Sample observations for efficiency
    sample_size = min(1000, len(data))
    sample_data = data.sample(sample_size, random_state=42)

    def measure_gap(df):
        """Measure gap on current dataframe"""
        ert_by_group = {"control": [], "dyslexic": []}
        for group_name in ["control", "dyslexic"]:
            group_data = df[df["group"] == group_name]
            for idx, row in group_data.iterrows():
                features_row = row[["length", "zipf", "surprisal"]].to_frame().T
                try:
                    ert = ert_predictor.predict_ert(features_row, group_name)[0]
                    ert_by_group[group_name].append(ert)
                except:
                    pass

        if len(ert_by_group["control"]) > 0 and len(ert_by_group["dyslexic"]) > 0:
            return np.mean(ert_by_group["dyslexic"]) - np.mean(ert_by_group["control"])
        return np.nan

    # Run permutation test
    for perm_idx in range(n_permutations):
        if (perm_idx + 1) % 16 == 0:
            logger.info(f"    Permutation {perm_idx + 1}/{n_permutations}")

        order = np.random.permutation(features)

        current_data = sample_data.copy()
        gap_prev = measure_gap(current_data)

        if np.isnan(gap_prev):
            continue

        for feat in order:
            # Apply shift for this feature
            current_data[feat] = current_data[feat] + shift_vector[feat]

            # Clip to valid ranges
            if feat == "length":
                current_data[feat] = current_data[feat].clip(lower=1)
            elif feat == "zipf":
                current_data[feat] = current_data[feat].clip(lower=1, upper=7)
            else:  # surprisal
                current_data[feat] = current_data[feat].clip(lower=0)

            gap_new = measure_gap(current_data)

            if not np.isnan(gap_new):
                marginal_reduction = gap_prev - gap_new
                contributions[feat].append(marginal_reduction)
                gap_prev = gap_new

    # Average contributions
    feature_contributions = {}
    for feat in features:
        if len(contributions[feat]) > 0:
            feature_contributions[feat] = float(np.mean(contributions[feat]))
        else:
            feature_contributions[feat] = 0.0

    total_contribution = sum(feature_contributions.values())

    logger.info(f"    Feature contributions to gap reduction:")
    for feat, contrib in feature_contributions.items():
        logger.info(
            f"      {feat}: {contrib:.2f} ms ({contrib/total_contribution*100:.1f}%)"
        )

    return {
        "feature_contributions_ms": feature_contributions,
        "total_ms": float(total_contribution),
        "n_permutations": n_permutations,
    }


def per_feature_equalization(ert_predictor, data: pd.DataFrame, feature: str) -> Dict:
    """
    Per-feature equalization using percentile matching
    """
    logger.info(f"  Computing per-feature equalization for {feature}...")

    sample_data = data.copy()

    # Baseline gap
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

    # Counterfactual with percentile matching
    control_data = sample_data_clean[sample_data_clean["group"] == "control"]
    dyslexic_data = sample_data_clean[sample_data_clean["group"] == "dyslexic"]

    ert_counterfactual = []

    for idx, row in sample_data_clean.iterrows():
        group = row["group"]
        features_cf = row[["length", "zipf", "surprisal"]].copy().to_frame().T

        if group == "dyslexic":
            # Percentile matching
            feature_value = row[feature]
            percentile = (dyslexic_data[feature] <= feature_value).mean()
            control_value = control_data[feature].quantile(percentile)
            features_cf[feature] = control_value

        try:
            ert = ert_predictor.predict_ert(features_cf, group)[0]
            ert_counterfactual.append(ert)
        except:
            ert_counterfactual.append(np.nan)

    sample_data_clean["ERT_counterfactual"] = ert_counterfactual
    sample_data_clean = sample_data_clean.dropna(subset=["ERT_counterfactual"])

    counterfactual_gap = (
        sample_data_clean[sample_data_clean["group"] == "dyslexic"][
            "ERT_counterfactual"
        ].mean()
        - sample_data_clean[sample_data_clean["group"] == "control"][
            "ERT_counterfactual"
        ].mean()
    )

    gap_explained = baseline_gap - counterfactual_gap
    pct_explained = (gap_explained / baseline_gap * 100) if baseline_gap != 0 else 0

    logger.info(f"    Baseline gap: {baseline_gap:.2f} ms")
    logger.info(f"    Counterfactual gap: {counterfactual_gap:.2f} ms")
    logger.info(f"    Gap explained: {gap_explained:.2f} ms ({pct_explained:.1f}%)")

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
    REVISED: Uses consistent canonical gap across all analyses
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 3: GAP DECOMPOSITION")
    logger.info("=" * 60)

    # === USE CONSISTENT SAMPLE AND CANONICAL GAP ===
    np.random.seed(ANALYSIS_SEED)
    sample_size = min(5000, len(data))
    analysis_sample = data.sample(sample_size, random_state=ANALYSIS_SEED)

    logger.info(f"\nUsing consistent sample: {len(analysis_sample):,} observations")
    logger.info(f"Random seed: {ANALYSIS_SEED}")

    # === COMPUTE CANONICAL GAP (used as baseline for all analyses) ===
    canonical_gap = compute_canonical_gap(analysis_sample)
    logger.info(f"\nCanonical gap: {canonical_gap:.2f} ms")

    # === SHAPLEY DECOMPOSITION ===
    shapley = shapley_decomposition(ert_predictor, analysis_sample, canonical_gap)

    # === EQUAL-EASE COUNTERFACTUAL ===
    equal_ease = equal_ease_counterfactual(
        ert_predictor, analysis_sample, quartiles, canonical_gap
    )

    # === EQUAL-EASE FEATURE CONTRIBUTIONS ===
    feature_contributions = equal_ease_feature_contributions(
        ert_predictor, analysis_sample, quartiles, n_permutations=64
    )

    # === PER-FEATURE EQUALIZATION ===
    per_feature = {}
    for feature in ["length", "zipf", "surprisal"]:
        per_feature[feature] = per_feature_equalization(
            ert_predictor, analysis_sample, feature
        )

    # === DECISION LOGIC ===
    any_substantial_effect = any(
        abs(res.get("pct_explained", 0)) >= 25 for res in per_feature.values()
    )
    text_difficulty_explained = equal_ease.get("gap_shrink_pct", 0) >= 20

    if any_substantial_effect and text_difficulty_explained:
        status = "STRONGLY CONFIRMED"
        summary = "both differential sensitivity and text difficulty"
    elif any_substantial_effect:
        status = "CONFIRMED"
        summary = "differential sensitivity to word features"
    elif text_difficulty_explained:
        status = "CONFIRMED"
        summary = "text difficulty (simplification reduces gap)"
    else:
        status = "NOT CONFIRMED"
        summary = "features explain minimal portion of gap"

    logger.info(f"\nHYPOTHESIS 3: {status}")
    logger.info(f"  {summary}")

    return {
        "status": status,
        "total_gap": canonical_gap,
        "sample_size": len(analysis_sample),
        "random_seed": ANALYSIS_SEED,
        "shapley_decomposition": shapley,
        "equal_ease_counterfactual": equal_ease,
        "equal_ease_feature_contributions": feature_contributions,
        "per_feature_equalization": per_feature,
        "summary": summary,
    }
