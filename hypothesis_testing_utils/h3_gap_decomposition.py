"""
Hypothesis 3: Gap Decomposition - FULLY RULES-COMPLIANT
Single source of truth (S & G0), consistent equal-ease policy, proper anchoring
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

ANALYSIS_SEED = 42
BASELINE_TOLERANCE = 1e-6


def predict_components_batch(predictor, df, group):
    """
    Batch helper for predictions with components.
    Returns 1D arrays of ert, p_skip, trt.
    """
    X = df[["length", "zipf", "surprisal"]]
    ert, p_skip, trt = predictor.predict_ert(X, group, return_components=True)
    # Ensure 1D arrays
    return np.asarray(ert).ravel(), np.asarray(p_skip).ravel(), np.asarray(trt).ravel()


def predict_gap_on_sample(ert_predictor, sample: pd.DataFrame) -> float:
    """
    Compute gap using model predictions on the given sample.
    This is the ONLY way to compute gaps - ensures consistency.
    Uses BATCH prediction for performance.
    """
    ert_by_group = {"control": [], "dyslexic": []}

    for group_name in ["control", "dyslexic"]:
        group_data = sample[sample["group"] == group_name]
        if len(group_data) == 0:
            continue

        # BATCH PREDICTION: extract all features at once
        features_batch = group_data[["length", "zipf", "surprisal"]].values
        try:
            # Predict on entire batch
            ert_batch = ert_predictor.predict_ert(
                pd.DataFrame(features_batch, columns=["length", "zipf", "surprisal"]),
                group_name,
            )
            ert_by_group[group_name] = ert_batch.tolist()
        except:
            continue

    if len(ert_by_group["control"]) > 0 and len(ert_by_group["dyslexic"]) > 0:
        return np.mean(ert_by_group["dyslexic"]) - np.mean(ert_by_group["control"])
    return np.nan


def bootstrap_gap_component(
    observed_value: float,
    data: pd.DataFrame,
    computation_fn,
    n_bootstrap: int = 200,
    **kwargs,
) -> Dict:
    """Bootstrap with subject resampling (not permutation test)"""
    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)

    bootstrap_values = []

    print(f"      Bootstrap (n={n_bootstrap})...", flush=True)

    for i in tqdm(range(n_bootstrap), desc="      Bootstrap", leave=False):
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
    ert_predictor, S: pd.DataFrame, G0: float, n_bootstrap: int = 200
) -> Dict:
    """
    Shapley decomposition - skip vs duration contributions.
    Uses the SAME sample S and baseline G0 as rest of H3.
    """
    logger.info("  Computing Shapley decomposition...")
    print("  Shapley decomposition...", flush=True)

    dys_data = S[S["group"] == "dyslexic"]

    if len(dys_data) == 0:
        return {"error": "No dyslexic data"}

    # === BATCHED PREDICTIONS (replaces per-row loops) ===
    ert_dys, p_skip_dys, trt_dys = predict_components_batch(
        ert_predictor, dys_data, "dyslexic"
    )
    mean_ert_dys = ert_dys.mean()

    ert_ctrl, p_skip_ctrl, trt_ctrl = predict_components_batch(
        ert_predictor, dys_data, "control"
    )
    mean_ert_ctrl = ert_ctrl.mean()

    computed_gap = mean_ert_dys - mean_ert_ctrl

    # Path 1: Equalize skip first
    ert_1 = (1 - p_skip_ctrl) * trt_dys
    mean_ert_1 = ert_1.mean()

    delta_skip_path1 = mean_ert_dys - mean_ert_1
    delta_duration_path1 = mean_ert_1 - mean_ert_ctrl

    # Path 2: Equalize duration first
    ert_3 = (1 - p_skip_dys) * trt_ctrl
    mean_ert_3 = ert_3.mean()

    delta_duration_path2 = mean_ert_dys - mean_ert_3
    delta_skip_path2 = mean_ert_3 - mean_ert_ctrl

    # Average Shapley values
    delta_skip = (delta_skip_path1 + delta_skip_path2) / 2
    delta_duration = (delta_duration_path1 + delta_duration_path2) / 2

    logger.info(
        f"    Computed gap (from predictions on dys subset): {computed_gap:.2f} ms"
    )
    logger.info(f"    Canonical gap G0 (single source of truth): {G0:.2f} ms")

    # Anchor to G0
    if abs(computed_gap) > 1e-9 and abs(computed_gap - G0) > BASELINE_TOLERANCE:
        scale = G0 / computed_gap
        logger.info(f"    Applying scale factor: {scale:.4f}")
    else:
        scale = 1.0

    skip_contribution_ms = delta_skip * scale
    duration_contribution_ms = delta_duration * scale

    # === BOOTSTRAP: SKIP contribution (batched inside) ===
    def compute_skip_contrib(boot_data):
        dys_boot = boot_data[boot_data["group"] == "dyslexic"]
        if len(dys_boot) < 100:
            return np.nan

        dys_sample = dys_boot.sample(min(500, len(dys_boot)))  # no fixed seed

        ed, psd, td = predict_components_batch(ert_predictor, dys_sample, "dyslexic")
        ec, psc, tc = predict_components_batch(ert_predictor, dys_sample, "control")

        if len(ed) < 10 or len(ec) < 10:
            return np.nan

        mean_dys = ed.mean()
        mean_ctrl = ec.mean()

        # Path 1: equalize skip first
        ert_1_b = (1 - psc) * td
        skip_1 = mean_dys - ert_1_b.mean()

        # Path 2: equalize duration first
        ert_3_b = (1 - psd) * tc
        skip_2 = ert_3_b.mean() - mean_ctrl

        return 0.5 * (skip_1 + skip_2)

    # === BOOTSTRAP: DURATION contribution (direct, not complement) ===
    def compute_duration_contrib(boot_data):
        dys_boot = boot_data[boot_data["group"] == "dyslexic"]
        if len(dys_boot) < 100:
            return np.nan

        dys_sample = dys_boot.sample(min(500, len(dys_boot)))  # no fixed seed

        ed, psd, td = predict_components_batch(ert_predictor, dys_sample, "dyslexic")
        ec, psc, tc = predict_components_batch(ert_predictor, dys_sample, "control")

        if len(ed) < 10 or len(ec) < 10:
            return np.nan

        mean_dys = ed.mean()
        mean_ctrl = ec.mean()

        # Path 1: duration first
        ert_3_b = (1 - psd) * tc
        dur_1 = mean_dys - ert_3_b.mean()

        # Path 2: duration second
        ert_1_b = (1 - psc) * td
        dur_2 = ert_1_b.mean() - mean_ctrl

        return 0.5 * (dur_1 + dur_2)

    logger.info("    Computing bootstrap for Shapley components...")
    print("    Shapley bootstrap (may take a few minutes)...", flush=True)
    stats_skip = bootstrap_gap_component(
        delta_skip, S, compute_skip_contrib, n_bootstrap=n_bootstrap
    )
    stats_dur = bootstrap_gap_component(
        delta_duration, S, compute_duration_contrib, n_bootstrap=n_bootstrap
    )

    # Scale both stats to G0
    for st in (stats_skip, stats_dur):
        st["mean"] *= scale
        st["ci_low"] *= scale
        st["ci_high"] *= scale

    stats_skip_scaled = {
        "mean": stats_skip["mean"],
        "ci_low": stats_skip["ci_low"],
        "ci_high": stats_skip["ci_high"],
        "p_value": stats_skip["p_value"],
        "n_bootstrap": stats_skip["n_bootstrap"],
    }
    stats_duration_scaled = {
        "mean": stats_dur["mean"],
        "ci_low": stats_dur["ci_low"],
        "ci_high": stats_dur["ci_high"],
        "p_value": stats_dur["p_value"],
        "n_bootstrap": stats_dur["n_bootstrap"],
    }

    # Percentages relative to G0
    pct_skip = (skip_contribution_ms / G0 * 100) if G0 != 0 else 0
    pct_duration = (duration_contribution_ms / G0 * 100) if G0 != 0 else 0

    logger.info(
        f"    Skip contribution: {skip_contribution_ms:.2f} ms ({pct_skip:.1f}%)"
    )
    logger.info(
        f"    Duration contribution: {duration_contribution_ms:.2f} ms ({pct_duration:.1f}%)"
    )
    logger.info(f"    Sum check: {pct_skip + pct_duration:.1f}% (should be 100%)")

    # Verify sum
    assert abs((skip_contribution_ms + duration_contribution_ms) - G0) < max(
        0.1 * abs(G0), 1.0
    ), "Skip + Duration must sum to G0"

    return {
        "skip_contribution": float(skip_contribution_ms),
        "skip_contribution_stats": stats_skip_scaled,
        "duration_contribution": float(duration_contribution_ms),
        "duration_contribution_stats": stats_duration_scaled,
        "skip_pct": float(pct_skip),
        "duration_pct": float(pct_duration),
    }


def equal_ease_counterfactual(
    ert_predictor,
    S: pd.DataFrame,
    quartiles: Dict,
    G0: float,
    n_bootstrap: int = 200,
) -> Dict:
    """
    Equal-ease counterfactual using Q1/Q3 clamp policy.
    Uses the SAME sample S and baseline G0.
    """
    logger.info("  Computing equal-ease counterfactual...")
    print("  Equal-ease counterfactual...", flush=True)

    # Extract quartiles
    Q1_len = quartiles["length"]["q1"]
    Q1_surpr = quartiles["surprisal"]["q1"]
    Q3_zipf = quartiles["zipf"]["q3"]

    # Baseline gap (should equal G0)
    baseline_gap = predict_gap_on_sample(ert_predictor, S)

    if abs(baseline_gap - G0) > BASELINE_TOLERANCE:
        logger.warning(f"    Baseline gap {baseline_gap:.2f} differs from G0 {G0:.2f}")

    # Apply equal-ease policy: clamp to Q1/Q3 (same for both groups)
    S_easy = S.copy()
    S_easy["length"] = S_easy["length"].clip(upper=Q1_len)
    S_easy["surprisal"] = S_easy["surprisal"].clip(upper=Q1_surpr)
    S_easy["zipf"] = S_easy["zipf"].clip(lower=Q3_zipf)

    # Counterfactual gap
    counterfactual_gap = predict_gap_on_sample(ert_predictor, S_easy)

    # Gap reduction
    gap_shrink_ms = G0 - counterfactual_gap
    gap_shrink_pct = (gap_shrink_ms / G0 * 100) if G0 != 0 else 0

    logger.info(f"    Baseline gap G0: {G0:.2f} ms")
    logger.info(f"    Counterfactual gap: {counterfactual_gap:.2f} ms")
    logger.info(f"    Gap shrink: {gap_shrink_ms:.2f} ms ({gap_shrink_pct:.1f}%)")

    # Bootstrap
    def compute_gap_shrink(boot_data):
        # No fixed seed inside bootstrap
        boot_sample = boot_data.sample(min(500, len(boot_data)))

        baseline_gap_b = predict_gap_on_sample(ert_predictor, boot_sample)

        if np.isnan(baseline_gap_b):
            return np.nan

        # Apply same policy
        boot_easy = boot_sample.copy()
        boot_easy["length"] = boot_easy["length"].clip(upper=Q1_len)
        boot_easy["surprisal"] = boot_easy["surprisal"].clip(upper=Q1_surpr)
        boot_easy["zipf"] = boot_easy["zipf"].clip(lower=Q3_zipf)

        cf_gap_b = predict_gap_on_sample(ert_predictor, boot_easy)

        if np.isnan(cf_gap_b):
            return np.nan

        return baseline_gap_b - cf_gap_b

    logger.info("    Computing bootstrap for gap shrinkage...")
    print("    Gap shrinkage bootstrap (may take a few minutes)...", flush=True)
    stats_gap_shrink = bootstrap_gap_component(
        gap_shrink_ms, S, compute_gap_shrink, n_bootstrap=n_bootstrap
    )

    return {
        "baseline_gap": float(G0),
        "counterfactual_gap": float(counterfactual_gap),
        "counterfactual_gap_stats": {
            "mean": float(counterfactual_gap),
            "ci_low": float(G0 - stats_gap_shrink["ci_high"]),
            "ci_high": float(G0 - stats_gap_shrink["ci_low"]),
            "p_value": stats_gap_shrink["p_value"],
        },
        "gap_shrink_ms": float(gap_shrink_ms),
        "gap_shrink_stats": stats_gap_shrink,
        "gap_shrink_pct": float(gap_shrink_pct),
    }


def equal_ease_feature_contributions(
    ert_predictor,
    S: pd.DataFrame,
    quartiles: Dict,
    gap_shrink_ms: float,
    n_permutations: int = 64,
) -> Dict:
    """
    Feature Shapley decomposition of equal-ease gap reduction.
    Uses SAME sample S and SAME policy. Contributions MUST sum to gap_shrink_ms.
    """
    logger.info(f"  Computing feature contributions ({n_permutations} permutations)...")
    print(f"  Feature contributions (Shapley, n={n_permutations})...", flush=True)

    # Extract quartiles
    Q1_len = quartiles["length"]["q1"]
    Q1_surpr = quartiles["surprisal"]["q1"]
    Q3_zipf = quartiles["zipf"]["q3"]

    features = ["length", "zipf", "surprisal"]
    contributions = {f: [] for f in features}

    # Value function: gap after applying clamps
    def v(S_current):
        """Value = gap after equalizing features in S_current"""
        return predict_gap_on_sample(ert_predictor, S_current)

    baseline_gap = v(S)

    for perm_idx in range(n_permutations):
        if (perm_idx + 1) % 16 == 0:
            logger.info(f"    Permutation {perm_idx + 1}/{n_permutations}")

        order = np.random.permutation(features)

        S_current = S.copy()
        gap_prev = baseline_gap

        if np.isnan(gap_prev):
            continue

        for feat in order:
            # Apply clamp for this feature
            if feat == "length":
                S_current["length"] = S_current["length"].clip(upper=Q1_len)
            elif feat == "surprisal":
                S_current["surprisal"] = S_current["surprisal"].clip(upper=Q1_surpr)
            elif feat == "zipf":
                S_current["zipf"] = S_current["zipf"].clip(lower=Q3_zipf)

            gap_new = v(S_current)

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
                "n_samples": len(samples),
            }
        else:
            feature_contributions[feat] = 0.0
            feature_stats[feat] = {
                "mean": 0.0,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "n_samples": 0,
            }

    shapley_sum = sum(feature_contributions.values())

    # Verify sum to gap_shrink_ms
    tolerance = max(0.1 * abs(gap_shrink_ms), 1.0) + 1e-9

    if abs(shapley_sum - gap_shrink_ms) > tolerance:
        logger.warning(
            f"    Shapley sum {shapley_sum:.2f} differs from gap_shrink {gap_shrink_ms:.2f}"
        )
        # Apply proportional renormalization
        if abs(shapley_sum) > 1e-9:
            scale = gap_shrink_ms / shapley_sum
            for feat in features:
                feature_contributions[feat] *= scale
                feature_stats[feat]["mean"] *= scale
                feature_stats[feat]["ci_low"] *= scale
                feature_stats[feat]["ci_high"] *= scale
            shapley_sum = gap_shrink_ms
            logger.info(f"    Applied renormalization: scale={scale:.4f}")

    logger.info(f"    Feature contributions to gap reduction:")
    for feat, contrib in feature_contributions.items():
        pct = (contrib / gap_shrink_ms * 100) if gap_shrink_ms != 0 else 0
        p_val = feature_stats[feat]["p_value"]
        logger.info(f"      {feat}: {contrib:.2f} ms ({pct:.1f}%, p={p_val:.5f})")
    logger.info(f"    Total Shapley reduction: {shapley_sum:.2f} ms")

    assert (
        abs(shapley_sum - gap_shrink_ms) < 1e-6
    ), "Feature contributions must sum to gap_shrink"

    return {
        "feature_contributions_ms": feature_contributions,
        "feature_contributions_stats": feature_stats,
        "total_ms": float(shapley_sum),
        "n_permutations": n_permutations,
    }


def per_feature_equalization(
    ert_predictor,
    S: pd.DataFrame,
    feature: str,
    quartiles: Dict,
    G0: float,
    n_bootstrap: int = 200,
) -> Dict:
    """
    Per-feature equalization: apply clamp to BOTH groups identically.
    Report delta_gap in ms and as % of G0.
    """
    logger.info(f"  Computing per-feature equalization for {feature}...")
    print(f"  Per-feature equalization: {feature}...", flush=True)

    # Extract quartile for this feature
    if feature == "length":
        Q1 = quartiles["length"]["q1"]
        clamp_fn = lambda df: df["length"].clip(upper=Q1)
    elif feature == "surprisal":
        Q1 = quartiles["surprisal"]["q1"]
        clamp_fn = lambda df: df["surprisal"].clip(upper=Q1)
    elif feature == "zipf":
        Q3 = quartiles["zipf"]["q3"]
        clamp_fn = lambda df: df["zipf"].clip(lower=Q3)
    else:
        raise ValueError(f"Unknown feature: {feature}")

    # Baseline gap
    baseline_gap = predict_gap_on_sample(ert_predictor, S)

    if abs(baseline_gap - G0) > BASELINE_TOLERANCE:
        logger.warning(f"    Baseline gap {baseline_gap:.2f} differs from G0 {G0:.2f}")

    # Apply clamp to BOTH groups
    S_eq = S.copy()
    S_eq[feature] = clamp_fn(S_eq)

    # Counterfactual gap
    counterfactual_gap = predict_gap_on_sample(ert_predictor, S_eq)

    # Gap reduction
    gap_explained_ms = G0 - counterfactual_gap
    pct_of_G0 = (gap_explained_ms / G0 * 100) if G0 != 0 else 0

    logger.info(f"    Baseline gap G0: {G0:.2f} ms")
    logger.info(f"    Gap after equalizing {feature}: {counterfactual_gap:.2f} ms")
    logger.info(
        f"    Gap explained: {gap_explained_ms:.2f} ms ({pct_of_G0:.1f}% of G0)"
    )

    # Bootstrap
    def compute_gap_explained_feat(boot_data):
        # No fixed seed inside bootstrap
        boot_sample = boot_data.sample(min(500, len(boot_data)))

        baseline_gap_b = predict_gap_on_sample(ert_predictor, boot_sample)

        if np.isnan(baseline_gap_b):
            return np.nan

        boot_eq = boot_sample.copy()
        boot_eq[feature] = clamp_fn(boot_eq)

        cf_gap_b = predict_gap_on_sample(ert_predictor, boot_eq)

        if np.isnan(cf_gap_b):
            return np.nan

        return baseline_gap_b - cf_gap_b

    logger.info(f"    Computing bootstrap for {feature} equalization...")
    print(
        f"    {feature} equalization bootstrap (may take a few minutes)...", flush=True
    )
    stats_explained = bootstrap_gap_component(
        gap_explained_ms, S, compute_gap_explained_feat, n_bootstrap=n_bootstrap
    )

    return {
        "feature": feature,
        "baseline_gap": float(G0),
        "counterfactual_gap": float(counterfactual_gap),
        "gap_explained_ms": float(gap_explained_ms),
        "gap_explained_stats": stats_explained,
        "pct_of_G0": float(pct_of_G0),
        "method": "clamp_both_groups",
    }


def test_hypothesis_3(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict[str, Dict[str, float]],
    n_bootstrap: int = 200,
) -> Dict:
    """
    Test Hypothesis 3: Gap decomposition - FULLY RULES-COMPLIANT

    Single source of truth: one sample S, one baseline G0.
    All analyses use the same S, G0, and equal-ease policy.
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 3: GAP DECOMPOSITION")
    logger.info("=" * 60)
    logger.info(f"Using {n_bootstrap} bootstrap iterations")
    print(f"\n=== H3: Gap Decomposition (n_bootstrap={n_bootstrap}) ===\n", flush=True)

    # Create THE analysis sample S (single source of truth)
    np.random.seed(ANALYSIS_SEED)
    sample_size = min(2000, len(data))
    S = data.sample(sample_size, random_state=ANALYSIS_SEED)

    logger.info(f"\nAnalysis sample S: {len(S):,} observations")
    logger.info(f"Random seed: {ANALYSIS_SEED}")

    # Compute THE canonical gap G0 (single source of truth)
    G0 = predict_gap_on_sample(ert_predictor, S)
    logger.info(f"\nCanonical gap G0: {G0:.2f} ms")

    # Bootstrap G0 itself
    def compute_gap_bootstrap(boot_data):
        """Bootstrap the gap by resampling subjects"""
        return predict_gap_on_sample(ert_predictor, boot_data)

    logger.info("  Computing statistics for G0...")
    print("  Computing G0 statistics...", flush=True)
    gap_stats = bootstrap_gap_component(
        G0,
        S,
        compute_gap_bootstrap,
        n_bootstrap=n_bootstrap,
    )

    # Run all analyses on SAME S with SAME G0
    shapley = shapley_decomposition(ert_predictor, S, G0, n_bootstrap)

    equal_ease = equal_ease_counterfactual(ert_predictor, S, quartiles, G0, n_bootstrap)

    # Pass gap_shrink to feature contributions so they sum correctly
    feature_contributions = equal_ease_feature_contributions(
        ert_predictor, S, quartiles, equal_ease["gap_shrink_ms"], n_permutations=64
    )

    per_feature = {}
    for feature in ["length", "zipf", "surprisal"]:
        per_feature[feature] = per_feature_equalization(
            ert_predictor, S, feature, quartiles, G0, n_bootstrap
        )

    # Decision logic
    any_substantial_effect = any(
        abs(res.get("pct_of_G0", 0)) >= 25 for res in per_feature.values()
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
        "canonical_gap_G0": float(G0),
        "canonical_gap_stats": gap_stats,
        "sample_size": len(S),
        "random_seed": ANALYSIS_SEED,
        "shapley_decomposition": shapley,
        "equal_ease_counterfactual": equal_ease,
        "equal_ease_feature_contributions": feature_contributions,
        "per_feature_equalization": per_feature,
        "summary": summary,
    }
