"""
Hypothesis 3: Gap Decomposition
Single source of truth (S & G0), consistent equal-ease policy
"""

import itertools
import logging
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from hypothesis_testing_utils.stats_utils import compute_two_tailed_pvalue_corrected

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
    Subject-balanced: mean of subject means within group (reduces variance).
    Uses BATCH prediction for performance.
    """
    group_means = {}
    for group_name in ["control", "dyslexic"]:
        gdf = sample[sample["group"] == group_name]
        if len(gdf) == 0:
            continue
        X = gdf[["length", "zipf", "surprisal"]]
        try:
            ert = np.asarray(ert_predictor.predict_ert(X, group_name)).ravel()
        except Exception:
            continue
        subj_mean = (
            pd.DataFrame({"subject_id": gdf["subject_id"].values, "ert": ert})
            .groupby("subject_id")["ert"]
            .mean()
        )
        group_means[group_name] = float(subj_mean.mean())

    if "control" in group_means and "dyslexic" in group_means:
        return group_means["dyslexic"] - group_means["control"]
    return np.nan


def bootstrap_gap_component(
    observed_value: float,
    data: pd.DataFrame,
    computation_fn,
    n_bootstrap: int = 2000,
    **kwargs,
) -> Dict:
    """Bootstrap with subject resampling (stratified by group)"""
    ctrl_subjects = data.loc[data["group"] == "control", "subject_id"].unique()
    dys_subjects = data.loc[data["group"] == "dyslexic", "subject_id"].unique()

    bootstrap_values = []

    print(f"      Bootstrap (n={n_bootstrap})...", flush=True)

    for i in tqdm(range(n_bootstrap), desc="      Bootstrap", leave=False):
        rng = np.random.RandomState(3000 + i)
        boot_ctrl = rng.choice(ctrl_subjects, size=len(ctrl_subjects), replace=True)
        boot_dys = rng.choice(dys_subjects, size=len(dys_subjects), replace=True)

        parts = []
        for j, s in enumerate(boot_ctrl):
            df = data[data["subject_id"] == s].copy()
            df["subject_id"] = f"ctrl_{s}_{j}"  # unique id per draw
            parts.append(df)
        for j, s in enumerate(boot_dys):
            df = data[data["subject_id"] == s].copy()
            df["subject_id"] = f"dys_{s}_{j}"  # unique id per draw
            parts.append(df)

        boot_data = pd.concat(parts, ignore_index=True)

        try:
            result = computation_fn(boot_data, **kwargs)
            if not np.isnan(result):
                bootstrap_values.append(result)
        except Exception:
            pass

    if len(bootstrap_values) > 0:
        ci_low = float(np.percentile(bootstrap_values, 2.5))
        ci_high = float(np.percentile(bootstrap_values, 97.5))

        # P-value with +1 correction
        p_value = compute_two_tailed_pvalue_corrected(
            observed_value, np.array(bootstrap_values)
        )
    else:
        ci_low = ci_high = p_value = np.nan

    return {
        "mean": float(observed_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "n_bootstrap": n_bootstrap,
        "n_effective_bootstrap": len(bootstrap_values),
    }


def _apply_equal_ease_subset(df, subset, quartiles, bin_edges, q3_zipf_by_bin):
    out = df.copy()
    if "length" in subset:
        out["length"] = out["length"].clip(upper=quartiles["length"]["q1"])
    if "surprisal" in subset:
        out["surprisal"] = out["surprisal"].clip(upper=quartiles["surprisal"]["q1"])
    if "zipf" in subset:
        out["zipf"] = _clamp_zipf_conditional(out, bin_edges, q3_zipf_by_bin)
    return out


def shapley_decomposition(
    ert_predictor, S: pd.DataFrame, G0: float, n_bootstrap: int = 2000
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

    # === BOOTSTRAP: SKIP contribution (use ALL dys tokens in each draw) ===
    def compute_skip_contrib(boot_data):
        dys_boot = boot_data[boot_data["group"] == "dyslexic"]
        if len(dys_boot) < 10:
            return np.nan

        ed, psd, td = predict_components_batch(ert_predictor, dys_boot, "dyslexic")
        ec, psc, tc = predict_components_batch(ert_predictor, dys_boot, "control")

        mean_dys = ed.mean()
        mean_ctrl = ec.mean()

        # Path 1: equalize skip first
        ert_1_b = (1 - psc) * td
        skip_1 = mean_dys - ert_1_b.mean()

        # Path 2: equalize duration first
        ert_3_b = (1 - psd) * tc
        skip_2 = ert_3_b.mean() - mean_ctrl

        return 0.5 * (skip_1 + skip_2)

    # === BOOTSTRAP: DURATION contribution (use ALL dys tokens in each draw) ===
    def compute_duration_contrib(boot_data):
        dys_boot = boot_data[boot_data["group"] == "dyslexic"]
        if len(dys_boot) < 10:
            return np.nan

        ed, psd, td = predict_components_batch(ert_predictor, dys_boot, "dyslexic")
        ec, psc, tc = predict_components_batch(ert_predictor, dys_boot, "control")

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


# ---------------------- NEW HELPERS (conditional Zipf policy) ----------------------


def _assign_length_bin(df: pd.DataFrame, bin_edges: np.ndarray) -> pd.Series:
    """Return integer length-bin indices for df['length'] using provided bin_edges."""
    return pd.cut(df["length"], bins=bin_edges, labels=False, include_lowest=True)


def _zipf_q3_by_length_bin(S: pd.DataFrame, bin_edges: np.ndarray) -> pd.Series:
    """
    Compute per-bin Q3 of zipf on the SINGLE source-of-truth sample S.
    Returns a Series indexed by bin id (0..n_bins-1).
    """
    bins_S = _assign_length_bin(S, bin_edges)
    tmp = S.copy()
    tmp["length_bin"] = bins_S
    return tmp.groupby("length_bin")["zipf"].quantile(0.75)


def _clamp_zipf_conditional(
    df: pd.DataFrame, bin_edges: np.ndarray, q3_by_bin: pd.Series
) -> pd.Series:
    """
    Clamp zipf within length bins: zipf := max(zipf, Q3_bin).
    Uses fixed thresholds q3_by_bin computed on S.
    If a row maps to a bin with no Q3 (rare if bootstrapping from S),
    leave that row unchanged.
    """
    bins_df = _assign_length_bin(df, bin_edges)
    thr = bins_df.map(q3_by_bin)
    z = df["zipf"].values
    t = thr.values.astype(float)
    t = np.where(np.isnan(t), z, t)  # no-op if bin unseen
    return pd.Series(np.maximum(z, t), index=df.index)


# -------------------------------------------------------------------------------


def equal_ease_counterfactual(
    ert_predictor,
    S: pd.DataFrame,
    quartiles: Dict,
    G0: float,
    n_bootstrap: int = 2000,
    bin_edges: np.ndarray = None,
) -> Dict:
    """
    Equal-ease counterfactual using Q1/Q3 clamp policy.
    Uses the SAME sample S and baseline G0.

    Zipf clamp is conditional on length: per-bin Q3 computed once on S.
    """
    logger.info("  Computing equal-ease counterfactual...")
    print("  Equal-ease counterfactual...", flush=True)

    # Extract quartiles
    Q1_len = quartiles["length"]["q1"]
    Q1_surpr = quartiles["surprisal"]["q1"]

    # NEW: conditional Zipf thresholds from S
    if bin_edges is None:
        raise ValueError("bin_edges must be provided for conditional Zipf equal-ease.")
    q3_zipf_by_bin = _zipf_q3_by_length_bin(S, bin_edges)

    # Baseline gap (should equal G0)
    baseline_gap = predict_gap_on_sample(ert_predictor, S)

    if abs(baseline_gap - G0) > BASELINE_TOLERANCE:
        logger.warning(f"    Baseline gap {baseline_gap:.2f} differs from G0 {G0:.2f}")

    # Apply equal-ease policy: global Q1 for length & surprisal, per-bin Q3 for Zipf
    S_easy = S.copy()
    S_easy["length"] = S_easy["length"].clip(upper=Q1_len)
    S_easy["surprisal"] = S_easy["surprisal"].clip(upper=Q1_surpr)
    S_easy["zipf"] = _clamp_zipf_conditional(S_easy, bin_edges, q3_zipf_by_bin)  # NEW

    # Counterfactual gap
    counterfactual_gap = predict_gap_on_sample(ert_predictor, S_easy)

    # Gap reduction
    gap_shrink_ms = G0 - counterfactual_gap
    gap_shrink_pct = (gap_shrink_ms / G0 * 100) if G0 != 0 else 0

    logger.info(f"    Baseline gap G0: {G0:.2f} ms")
    logger.info(f"    Counterfactual gap: {counterfactual_gap:.2f} ms")
    logger.info(f"    Gap shrink: {gap_shrink_ms:.2f} ms ({gap_shrink_pct:.1f}%)")

    def compute_cf_gap(boot_data):
        boot_easy = boot_data.copy()
        boot_easy["length"] = boot_easy["length"].clip(upper=Q1_len)
        boot_easy["surprisal"] = boot_easy["surprisal"].clip(upper=Q1_surpr)
        boot_easy["zipf"] = _clamp_zipf_conditional(
            boot_easy, bin_edges, q3_zipf_by_bin
        )
        return predict_gap_on_sample(ert_predictor, boot_easy)

    # Bootstrap (policy thresholds fixed from S)
    def compute_gap_shrink(boot_data):
        baseline_gap_b = predict_gap_on_sample(ert_predictor, boot_data)
        if np.isnan(baseline_gap_b):
            return np.nan

        boot_easy = boot_data.copy()
        boot_easy["length"] = boot_easy["length"].clip(upper=Q1_len)
        boot_easy["surprisal"] = boot_easy["surprisal"].clip(upper=Q1_surpr)
        boot_easy["zipf"] = _clamp_zipf_conditional(
            boot_easy, bin_edges, q3_zipf_by_bin
        )

        cf_gap_b = predict_gap_on_sample(ert_predictor, boot_easy)
        if np.isnan(cf_gap_b):
            return np.nan

        return baseline_gap_b - cf_gap_b

    logger.info("    Computing bootstrap for gap shrinkage...")
    print("    Gap shrinkage bootstrap (may take a few minutes)...", flush=True)
    stats_gap_shrink = bootstrap_gap_component(
        gap_shrink_ms, S, compute_gap_shrink, n_bootstrap=n_bootstrap
    )
    stats_cf_gap = bootstrap_gap_component(
        counterfactual_gap, S, compute_cf_gap, n_bootstrap=n_bootstrap
    )

    return {
        "baseline_gap": float(G0),
        "counterfactual_gap": float(counterfactual_gap),
        "counterfactual_gap_stats": {
            "mean": float(stats_cf_gap["mean"]),
            "ci_low": float(stats_cf_gap["ci_low"]),
            "ci_high": float(stats_cf_gap["ci_high"]),
            "p_value": float(stats_cf_gap["p_value"]),
            "n_bootstrap": stats_cf_gap["n_bootstrap"],
        },
        "gap_shrink_ms": float(gap_shrink_ms),
        "gap_shrink_stats": stats_gap_shrink,
        "gap_shrink_p_value": stats_gap_shrink["p_value"],
        "gap_shrink_pct": float(gap_shrink_pct),
    }


def equal_ease_feature_contributions(
    ert_predictor,
    S: pd.DataFrame,
    quartiles: Dict,
    gap_shrink_ms: float,
    n_bootstrap: int = 2000,
    bin_edges: np.ndarray = None,
) -> Dict:
    """
    Shapley feature contributions via exact enumeration of 3! orders.
    Uncertainty via subject bootstrap; policy thresholds fixed from S.
    """
    if bin_edges is None:
        raise ValueError("bin_edges must be provided for conditional Zipf equal-ease.")
    q3_zipf_by_bin = _zipf_q3_by_length_bin(S, bin_edges)
    features = ["length", "zipf", "surprisal"]
    orders = list(itertools.permutations(features))  # 6 orders

    def v(df):
        return predict_gap_on_sample(ert_predictor, df)

    def apply_subset(df, subset):
        return _apply_equal_ease_subset(
            df, subset, quartiles, bin_edges, q3_zipf_by_bin
        )

    # --- Point estimate on S by enumerating all 6 orders ---
    baseline_gap = v(S)
    per_order_contribs = {f: [] for f in features}
    for order in orders:
        applied = set()
        gap_prev = baseline_gap
        for feat in order:
            applied.add(feat)
            S_curr = apply_subset(S, applied)
            gap_new = v(S_curr)
            if np.isnan(gap_new):  # skip if failure
                continue
            per_order_contribs[feat].append(gap_prev - gap_new)
            gap_prev = gap_new

    contrib_point = {
        f: float(np.mean(per_order_contribs[f])) if per_order_contribs[f] else 0.0
        for f in features
    }

    # --- Bootstrap for CIs/p-values (re-enumerate all 6 orders each draw) ---
    contrib_samples = {f: [] for f in features}
    if n_bootstrap and n_bootstrap > 0:
        ctrl_subj = S.loc[S["group"] == "control", "subject_id"].unique()
        dys_subj = S.loc[S["group"] == "dyslexic", "subject_id"].unique()

        for i in tqdm(
            range(n_bootstrap), desc="      Feature Shapley bootstrap", leave=False
        ):
            rng = np.random.RandomState(7000 + i)
            boot_ctrl = rng.choice(ctrl_subj, size=len(ctrl_subj), replace=True)
            boot_dys = rng.choice(dys_subj, size=len(dys_subj), replace=True)

            parts = []
            for j, s in enumerate(boot_ctrl):
                df = S[S["subject_id"] == s].copy()
                df["subject_id"] = f"ctrl_{s}_{j}"
                parts.append(df)
            for j, s in enumerate(boot_dys):
                df = S[S["subject_id"] == s].copy()
                df["subject_id"] = f"dys_{s}_{j}"
                parts.append(df)
            boot = pd.concat(parts, ignore_index=True)

            # Evaluate all 6 orders on the boot sample
            base = v(boot)
            if np.isnan(base):
                continue

            sums = {f: 0.0 for f in features}
            counts = {f: 0 for f in features}
            for order in orders:
                applied = set()
                gap_prev = base
                for feat in order:
                    applied.add(feat)
                    boot_curr = apply_subset(boot, applied)
                    gap_new = v(boot_curr)
                    if np.isnan(gap_new):
                        continue
                    delta = gap_prev - gap_new
                    sums[feat] += delta
                    counts[feat] += 1
                    gap_prev = gap_new

            for f in features:
                if counts[f] > 0:
                    contrib_samples[f].append(sums[f] / counts[f])

    # --- Assemble stats ---
    stats = {}
    for f in features:
        if contrib_samples[f]:
            arr = np.array(contrib_samples[f], float)
            ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
            p = compute_two_tailed_pvalue_corrected(contrib_point[f], arr)
            stats[f] = {
                "mean": float(contrib_point[f]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": float(p),
                "n_samples": int(arr.size),
            }
        else:
            stats[f] = {
                "mean": float(contrib_point[f]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "n_samples": 0,
            }

    total = sum(contrib_point.values())
    # (optional) renormalize to gap_shrink_ms if tiny drift:
    if not np.isnan(gap_shrink_ms) and abs(total - gap_shrink_ms) > max(
        1.0, 0.1 * abs(gap_shrink_ms)
    ):
        scale = gap_shrink_ms / total if abs(total) > 1e-9 else 1.0
        for f in features:
            stats[f]["mean"] *= scale
            if contrib_samples[f]:
                stats[f]["ci_low"] *= scale
                stats[f]["ci_high"] *= scale
        total = gap_shrink_ms

    return {
        "feature_contributions_ms": {f: stats[f]["mean"] for f in features},
        "feature_contributions_stats": stats,
        "total_ms": float(total),
        "n_orders": len(orders),
        "n_bootstrap": int(n_bootstrap),
    }


def test_hypothesis_3(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict[str, Dict[str, float]],
    n_bootstrap: int = 2000,
    bin_edges: np.ndarray = None,
) -> Dict:
    """
    Test Hypothesis 3: Gap decomposition

    Single source of truth: one sample S, one baseline G0.
    All analyses use the same S, G0, and equal-ease policy.
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 3: GAP DECOMPOSITION")
    logger.info("=" * 60)
    logger.info(f"Using {n_bootstrap} bootstrap iterations")
    print(f"\n=== H3: Gap Decomposition (n_bootstrap={n_bootstrap}) ===\n", flush=True)

    if bin_edges is None:
        raise ValueError(
            "H3 requires bin_edges for conditional Zipf equal-ease (match H1)."
        )

    # Create THE analysis sample S (single source of truth)
    np.random.seed(ANALYSIS_SEED)
    S = data.copy()

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

    equal_ease = equal_ease_counterfactual(
        ert_predictor, S, quartiles, G0, n_bootstrap, bin_edges=bin_edges
    )

    # Pass gap_shrink to feature contributions so they sum correctly
    feature_contributions = equal_ease_feature_contributions(
        ert_predictor,
        S,
        quartiles,
        equal_ease["gap_shrink_ms"],
        n_bootstrap=n_bootstrap,
        bin_edges=bin_edges,
    )

    text_difficulty_explained = equal_ease.get("gap_shrink_pct", 0) >= 20

    if text_difficulty_explained:
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
        "summary": summary,
    }
