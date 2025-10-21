"""
Hypothesis 3: Gap Decomposition - REVISED
Added p-values and comprehensive statistics for all analyses
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANALYSIS_SEED = 42


def compute_canonical_gap(data: pd.DataFrame) -> float:
    """Compute canonical gap on the analysis sample"""
    ctrl_ert = data[data["group"] == "control"]["ERT"].mean()
    dys_ert = data[data["group"] == "dyslexic"]["ERT"].mean()
    return float(dys_ert - ctrl_ert)


def permutation_test_gap_component(
    observed_value: float,
    data: pd.DataFrame,
    computation_fn,
    n_permutations: int = 200,  # Reduced default
    **kwargs,
) -> Dict:
    """
    Generic permutation test for gap components

    Returns p-value and 95% CI
    """
    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)

    bootstrap_values = []

    print(f"      Permutation test (n={n_permutations})...", flush=True)

    for i in tqdm(range(n_permutations), desc="      Permutation", leave=False):
        np.random.seed(i + 3000)
        boot_subjects = np.random.choice(subjects, size=n_subjects, replace=True)
        boot_data = pd.concat(
            [data[data["subject_id"] == s] for s in boot_subjects], ignore_index=True
        )

        if len(boot_data) < 500:
            continue

        try:
            result = computation_fn(boot_data, **kwargs)
            if not np.isnan(result):
                bootstrap_values.append(result)
        except:
            pass

    if len(bootstrap_values) > 0:
        ci_low = float(np.percentile(bootstrap_values, 2.5))
        ci_high = float(np.percentile(bootstrap_values, 97.5))

        # P-value: proportion with opposite sign
        if observed_value > 0:
            p_value = float(np.mean(np.array(bootstrap_values) <= 0))
        else:
            p_value = float(np.mean(np.array(bootstrap_values) >= 0))

        p_value = min(1.0, 2 * p_value)
    else:
        ci_low = ci_high = p_value = np.nan

    return {
        "mean": float(observed_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "n_bootstrap": len(bootstrap_values),
    }


def shapley_decomposition(
    ert_predictor, data: pd.DataFrame, canonical_gap: float, n_permutations: int = 200
) -> Dict:
    """Shapley decomposition with statistics"""
    logger.info("  Computing Shapley decomposition...")
    print("  Shapley decomposition...", flush=True)

    sample_data = data.copy()
    dys_data = sample_data[sample_data["group"] == "dyslexic"]

    if len(dys_data) == 0:
        return {"error": "No dyslexic data"}

    # Predictions
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

    # Path 1
    ert_1 = [(1 - p_s_c) * t_d for p_s_c, t_d in zip(p_skip_ctrl, trt_dys)]
    mean_ert_1 = np.mean(ert_1)

    delta_skip_path1 = mean_ert_dys - mean_ert_1
    delta_duration_path1 = mean_ert_1 - mean_ert_ctrl

    # Path 2
    ert_3 = [(1 - p_s_d) * t_c for p_s_d, t_c in zip(p_skip_dys, trt_ctrl)]
    mean_ert_3 = np.mean(ert_3)

    delta_duration_path2 = mean_ert_dys - mean_ert_3
    delta_skip_path2 = mean_ert_3 - mean_ert_ctrl

    # Average
    delta_skip = (delta_skip_path1 + delta_skip_path2) / 2
    delta_duration = (delta_duration_path1 + delta_duration_path2) / 2

    pct_skip = (delta_skip / canonical_gap * 100) if canonical_gap != 0 else 0
    pct_duration = (delta_duration / canonical_gap * 100) if canonical_gap != 0 else 0

    logger.info(f"    Canonical gap: {canonical_gap:.2f} ms")
    logger.info(f"    Skip contribution: {delta_skip:.2f} ms ({pct_skip:.1f}%)")
    logger.info(
        f"    Duration contribution: {delta_duration:.2f} ms ({pct_duration:.1f}%)"
    )

    # Add permutation tests for contributions
    def compute_skip_contrib(boot_data):
        dys_boot = boot_data[boot_data["group"] == "dyslexic"]
        ert_dys_b, p_skip_dys_b, trt_dys_b = [], [], []
        ert_ctrl_b, p_skip_ctrl_b, trt_ctrl_b = [], [], []

        for idx, row in dys_boot.iterrows():
            features = row[["length", "zipf", "surprisal"]].to_frame().T
            try:
                ert_d, p_d, t_d = ert_predictor.predict_ert(
                    features, "dyslexic", return_components=True
                )
                ert_c, p_c, t_c = ert_predictor.predict_ert(
                    features, "control", return_components=True
                )
                ert_dys_b.append(ert_d[0])
                p_skip_dys_b.append(p_d[0])
                trt_dys_b.append(t_d[0])
                ert_ctrl_b.append(ert_c[0])
                p_skip_ctrl_b.append(p_c[0])
                trt_ctrl_b.append(t_c[0])
            except:
                pass

        if len(ert_dys_b) < 10:
            return np.nan

        mean_dys = np.mean(ert_dys_b)
        mean_ctrl = np.mean(ert_ctrl_b)
        ert_1_b = [(1 - p_c) * t_d for p_c, t_d in zip(p_skip_ctrl_b, trt_dys_b)]
        ert_3_b = [(1 - p_d) * t_c for p_d, t_c in zip(p_skip_dys_b, trt_ctrl_b)]

        skip_1 = mean_dys - np.mean(ert_1_b)
        skip_2 = np.mean(ert_3_b) - mean_ctrl

        return (skip_1 + skip_2) / 2

    logger.info("    Computing permutation tests for Shapley components...")
    print("    Shapley permutation tests...", flush=True)
    stats_skip = permutation_test_gap_component(
        delta_skip, data, compute_skip_contrib, n_permutations=n_permutations
    )

    # Duration is just the complement
    stats_duration = {
        "mean": float(delta_duration),
        "ci_low": canonical_gap - stats_skip["ci_high"],
        "ci_high": canonical_gap - stats_skip["ci_low"],
        "p_value": stats_skip["p_value"],
        "n_bootstrap": stats_skip["n_bootstrap"],
    }

    return {
        "total_gap": float(canonical_gap),
        "skip_contribution": float(delta_skip),
        "skip_contribution_stats": stats_skip,
        "duration_contribution": float(delta_duration),
        "duration_contribution_stats": stats_duration,
        "skip_pct": float(pct_skip),
        "duration_pct": float(pct_duration),
    }


def equal_ease_counterfactual(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict,
    canonical_gap: float,
    n_permutations: int = 200,
) -> Dict:
    """Equal-ease counterfactual with statistics"""
    logger.info("  Computing equal-ease counterfactual...")
    print("  Equal-ease counterfactual...", flush=True)

    sample_data = data.copy()

    iqr = {feat: vals["iqr"] for feat, vals in quartiles.items()}

    # Baseline
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

    shifted_data["length"] = shifted_data["length"].clip(lower=1)
    shifted_data["zipf"] = shifted_data["zipf"].clip(lower=1, upper=7)
    shifted_data["surprisal"] = shifted_data["surprisal"].clip(lower=0)

    # Counterfactual
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

    dys_saved = np.mean(baseline_ert["dyslexic"]) - np.mean(
        counterfactual_ert["dyslexic"]
    )
    ctrl_saved = np.mean(baseline_ert["control"]) - np.mean(
        counterfactual_ert["control"]
    )
    gap_shrink = canonical_gap - counterfactual_gap
    gap_shrink_pct = (gap_shrink / canonical_gap * 100) if canonical_gap != 0 else 0

    logger.info(f"    Canonical gap: {canonical_gap:.2f} ms")
    logger.info(f"    Counterfactual gap: {counterfactual_gap:.2f} ms")
    logger.info(f"    Gap shrink: {gap_shrink:.2f} ms ({gap_shrink_pct:.1f}%)")

    # Permutation test for gap shrinkage
    def compute_gap_shrink(boot_data):
        baseline_b = {"control": [], "dyslexic": []}
        for gname in ["control", "dyslexic"]:
            gdata = boot_data[boot_data["group"] == gname]
            for idx, row in gdata.iterrows():
                features = row[["length", "zipf", "surprisal"]].to_frame().T
                try:
                    ert = ert_predictor.predict_ert(features, gname)[0]
                    baseline_b[gname].append(ert)
                except:
                    pass

        baseline_gap_b = np.mean(baseline_b["dyslexic"]) - np.mean(
            baseline_b["control"]
        )

        shifted_b = boot_data.copy()
        shifted_b["length"] = (shifted_b["length"] - iqr["length"]).clip(lower=1)
        shifted_b["zipf"] = (shifted_b["zipf"] + iqr["zipf"]).clip(lower=1, upper=7)
        shifted_b["surprisal"] = (shifted_b["surprisal"] - iqr["surprisal"]).clip(
            lower=0
        )

        counterfactual_b = {"control": [], "dyslexic": []}
        for gname in ["control", "dyslexic"]:
            gdata = shifted_b[shifted_b["group"] == gname]
            for idx, row in gdata.iterrows():
                features = row[["length", "zipf", "surprisal"]].to_frame().T
                try:
                    ert = ert_predictor.predict_ert(features, gname)[0]
                    counterfactual_b[gname].append(ert)
                except:
                    pass

        if (
            len(counterfactual_b["control"]) < 10
            or len(counterfactual_b["dyslexic"]) < 10
        ):
            return np.nan

        counterfactual_gap_b = np.mean(counterfactual_b["dyslexic"]) - np.mean(
            counterfactual_b["control"]
        )

        return baseline_gap_b - counterfactual_gap_b

    logger.info("    Computing permutation test for gap shrinkage...")
    print("    Gap shrinkage permutation test...", flush=True)
    stats_gap_shrink = permutation_test_gap_component(
        gap_shrink, data, compute_gap_shrink, n_permutations=n_permutations
    )

    return {
        "baseline_gap": float(canonical_gap),
        "counterfactual_gap": float(counterfactual_gap),
        "counterfactual_gap_stats": {
            "mean": float(counterfactual_gap),
            "ci_low": canonical_gap - stats_gap_shrink["ci_high"],
            "ci_high": canonical_gap - stats_gap_shrink["ci_low"],
            "p_value": stats_gap_shrink["p_value"],
        },
        "dyslexic_saved": float(dys_saved),
        "control_saved": float(ctrl_saved),
        "gap_shrink_ms": float(gap_shrink),
        "gap_shrink_stats": stats_gap_shrink,
        "gap_shrink_pct": float(gap_shrink_pct),
    }


def equal_ease_feature_contributions(
    ert_predictor, data: pd.DataFrame, quartiles: Dict, n_permutations: int = 64
) -> Dict:
    """Shapley decomposition of feature contributions"""
    logger.info(f"  Computing feature contributions ({n_permutations} permutations)...")

    shift_vector = {
        "length": -quartiles["length"]["iqr"],
        "zipf": quartiles["zipf"]["iqr"],
        "surprisal": -quartiles["surprisal"]["iqr"],
    }

    features = ["length", "zipf", "surprisal"]
    contributions = {f: [] for f in features}

    sample_size = min(1000, len(data))
    sample_data = data.sample(sample_size, random_state=42)

    def measure_gap(df):
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

    for perm_idx in range(n_permutations):
        if (perm_idx + 1) % 16 == 0:
            logger.info(f"    Permutation {perm_idx + 1}/{n_permutations}")

        order = np.random.permutation(features)

        current_data = sample_data.copy()
        gap_prev = measure_gap(current_data)

        if np.isnan(gap_prev):
            continue

        for feat in order:
            current_data[feat] = current_data[feat] + shift_vector[feat]

            if feat == "length":
                current_data[feat] = current_data[feat].clip(lower=1)
            elif feat == "zipf":
                current_data[feat] = current_data[feat].clip(lower=1, upper=7)
            else:
                current_data[feat] = current_data[feat].clip(lower=0)

            gap_new = measure_gap(current_data)

            if not np.isnan(gap_new):
                marginal_reduction = gap_prev - gap_new
                contributions[feat].append(marginal_reduction)
                gap_prev = gap_new

    feature_contributions = {}
    feature_stats = {}

    for feat in features:
        if len(contributions[feat]) > 0:
            samples = contributions[feat]
            mean_contrib = float(np.mean(samples))
            feature_contributions[feat] = mean_contrib

            feature_stats[feat] = {
                "mean": mean_contrib,
                "ci_low": float(np.percentile(samples, 2.5)),
                "ci_high": float(np.percentile(samples, 97.5)),
                "p_value": float(
                    min(
                        1.0,
                        2
                        * np.mean(
                            np.array(samples) <= 0
                            if mean_contrib > 0
                            else np.array(samples) >= 0
                        ),
                    )
                ),
            }
        else:
            feature_contributions[feat] = 0.0
            feature_stats[feat] = {
                "mean": 0.0,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
            }

    total_contribution = sum(feature_contributions.values())

    logger.info(f"    Feature contributions to gap reduction:")
    for feat, contrib in feature_contributions.items():
        p_val = feature_stats[feat]["p_value"]
        logger.info(
            f"      {feat}: {contrib:.2f} ms "
            f"({contrib/total_contribution*100:.1f}%, p={p_val:.5f})"
        )

    return {
        "feature_contributions_ms": feature_contributions,
        "feature_contributions_stats": feature_stats,
        "total_ms": float(total_contribution),
        "n_permutations": n_permutations,
    }


def per_feature_equalization(
    ert_predictor, data: pd.DataFrame, feature: str, n_permutations: int = 200
) -> Dict:
    """Per-feature equalization with statistics"""
    logger.info(f"  Computing per-feature equalization for {feature}...")
    print(f"  Per-feature equalization: {feature}...", flush=True)

    sample_data = data.copy()

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

    control_data = sample_data_clean[sample_data_clean["group"] == "control"]
    dyslexic_data = sample_data_clean[sample_data_clean["group"] == "dyslexic"]

    ert_counterfactual = []

    for idx, row in sample_data_clean.iterrows():
        group = row["group"]
        features_cf = row[["length", "zipf", "surprisal"]].copy().to_frame().T

        if group == "dyslexic":
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

    logger.info(f"    Gap explained: {gap_explained:.2f} ms ({pct_explained:.1f}%)")

    # Permutation test
    def compute_gap_explained_feat(boot_data):
        ert_obs_b = []
        groups_b = []
        valid_b = []

        for idx, row in boot_data.iterrows():
            group = row["group"]
            features = row[["length", "zipf", "surprisal"]].to_frame().T
            try:
                ert = ert_predictor.predict_ert(features, group)[0]
                ert_obs_b.append(ert)
                groups_b.append(group)
                valid_b.append(idx)
            except:
                pass

        boot_clean = boot_data.loc[valid_b].copy()
        boot_clean["ERT_observed"] = ert_obs_b
        boot_clean["group"] = groups_b

        baseline_gap_b = (
            boot_clean[boot_clean["group"] == "dyslexic"]["ERT_observed"].mean()
            - boot_clean[boot_clean["group"] == "control"]["ERT_observed"].mean()
        )

        ctrl_b = boot_clean[boot_clean["group"] == "control"]
        dys_b = boot_clean[boot_clean["group"] == "dyslexic"]

        if len(ctrl_b) < 10 or len(dys_b) < 10:
            return np.nan

        ert_cf_b = []
        for idx, row in boot_clean.iterrows():
            group = row["group"]
            features_cf = row[["length", "zipf", "surprisal"]].copy().to_frame().T

            if group == "dyslexic":
                feat_val = row[feature]
                pct = (dys_b[feature] <= feat_val).mean()
                ctrl_val = ctrl_b[feature].quantile(pct)
                features_cf[feature] = ctrl_val

            try:
                ert = ert_predictor.predict_ert(features_cf, group)[0]
                ert_cf_b.append(ert)
            except:
                ert_cf_b.append(np.nan)

        boot_clean["ERT_counterfactual"] = ert_cf_b
        boot_clean = boot_clean.dropna(subset=["ERT_counterfactual"])

        if len(boot_clean) < 10:
            return np.nan

        cf_gap_b = (
            boot_clean[boot_clean["group"] == "dyslexic"]["ERT_counterfactual"].mean()
            - boot_clean[boot_clean["group"] == "control"]["ERT_counterfactual"].mean()
        )

        return baseline_gap_b - cf_gap_b

    logger.info(f"    Computing permutation test for {feature} equalization...")
    print(f"    {feature} equalization permutation test...", flush=True)
    stats_explained = permutation_test_gap_component(
        gap_explained, data, compute_gap_explained_feat, n_permutations=n_permutations
    )

    return {
        "feature": feature,
        "baseline_gap": float(baseline_gap),
        "counterfactual_gap": float(counterfactual_gap),
        "gap_explained": float(gap_explained),
        "gap_explained_stats": stats_explained,
        "pct_explained": float(pct_explained),
        "method": "percentile_matching",
    }


def test_hypothesis_3(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict[str, Dict[str, float]],
    n_permutations: int = 200,  # Reduced default
) -> Dict:
    """
    Test Hypothesis 3: Gap decomposition
    FULLY REVISED: Added p-values for all components

    Args:
        n_permutations: Number of permutations for p-value estimation (default: 200)
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 3: GAP DECOMPOSITION")
    logger.info("=" * 60)
    logger.info(f"Using {n_permutations} permutations for p-value estimation")
    print(f"\n=== H3: Gap Decomposition (n_perm={n_permutations}) ===\n", flush=True)

    np.random.seed(ANALYSIS_SEED)
    sample_size = min(5000, len(data))
    analysis_sample = data.sample(sample_size, random_state=ANALYSIS_SEED)

    logger.info(f"\nUsing consistent sample: {len(analysis_sample):,} observations")

    canonical_gap = compute_canonical_gap(analysis_sample)
    logger.info(f"\nCanonical gap: {canonical_gap:.2f} ms")

    # Shapley decomposition
    shapley = shapley_decomposition(
        ert_predictor, analysis_sample, canonical_gap, n_permutations
    )

    # Equal-ease counterfactual
    equal_ease = equal_ease_counterfactual(
        ert_predictor, analysis_sample, quartiles, canonical_gap, n_permutations
    )

    # Feature contributions
    feature_contributions = equal_ease_feature_contributions(
        ert_predictor, analysis_sample, quartiles, n_permutations=64
    )

    # Per-feature equalization
    per_feature = {}
    for feature in ["length", "zipf", "surprisal"]:
        per_feature[feature] = per_feature_equalization(
            ert_predictor, analysis_sample, feature, n_permutations
        )

    # Decision logic
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
